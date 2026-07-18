from asgiref.sync import async_to_sync
from channels.layers import get_channel_layer
from django.db.models import Avg, Count, Q
from django.utils import timezone

from .models import Conversation, Message, PredictionMetric
from .services import get_bot


def get_user_key(request=None, explicit_user_id=None):
    if explicit_user_id:
        return str(explicit_user_id)
    if request and getattr(request, "user", None) and request.user.is_authenticated:
        return f"user:{request.user.pk}"
    if request:
        if not request.session.session_key:
            request.session.save()
        return f"session:{request.session.session_key}"
    return "anonymous"


def get_or_create_conversation(user_id):
    conversation, _ = Conversation.objects.get_or_create(user_id=user_id)
    return conversation


def get_recent_history(conversation, limit=1):
    recent = conversation.messages.order_by("-timestamp")[:limit]
    history = []
    for message in reversed(list(recent)):
        history.append(message.user_text)
    return history


def process_chat_message(user_text, user_id="anonymous", lang="en"):
    text = (user_text or "").strip()
    if not text:
        raise ValueError("text is required")

    conversation = get_or_create_conversation(user_id)
    history = get_recent_history(conversation)
    bot = get_bot()
    result = bot.answer_with_followup_suggestions(text, history)

    message = Message.objects.create(
        conversation=conversation,
        user_text=text,
        bot_response=result["reply"],
        intent_tag=result["tag"],
        confidence=result["confidence"],
        uncertainty=result["uncertainty"],
        is_uncertain=result["is_uncertain"],
    )
    conversation.updated_at = timezone.now()
    conversation.save(update_fields=["updated_at"])
    PredictionMetric.objects.create(
        conversation=conversation,
        intent=result["tag"],
        confidence=result["confidence"],
        uncertainty=result["uncertainty"],
    )

    payload = {
        "message_id": message.id,
        "conversation_id": conversation.id,
        "reply": result["reply"],
        "tag": result["tag"],
        "confidence": result["confidence"],
        "uncertainty": result["uncertainty"],
        "is_uncertain": result["is_uncertain"],
        "suggested_followups": result["suggested_followups"],
        "top3": result["top3"],
        "lang": lang,
        "timestamp": message.timestamp.isoformat(),
    }
    broadcast_metrics_update()
    return payload


def metrics_summary():
    total_predictions = PredictionMetric.objects.count()
    avg_confidence = PredictionMetric.objects.aggregate(value=Avg("confidence"))["value"] or 0.0
    avg_uncertainty = PredictionMetric.objects.aggregate(value=Avg("uncertainty"))["value"] or 0.0
    satisfied_count = PredictionMetric.objects.filter(user_satisfied=True).count()
    rated_count = PredictionMetric.objects.filter(user_satisfied__isnull=False).count()
    active_sessions = Conversation.objects.filter(
        updated_at__gte=timezone.now() - timezone.timedelta(minutes=30)
    ).count()
    top_intents = list(
        PredictionMetric.objects.values("intent")
        .annotate(count=Count("id"), avg_confidence=Avg("confidence"))
        .order_by("-count")[:5]
    )
    uncertain_count = PredictionMetric.objects.filter(Q(uncertainty__gte=0.7) | Q(confidence__lt=0.35)).count()

    return {
        "total_predictions": total_predictions,
        "avg_confidence": float(avg_confidence),
        "avg_uncertainty": float(avg_uncertainty),
        "success_rate": float(satisfied_count / rated_count) if rated_count else None,
        "active_sessions": active_sessions,
        "uncertain_predictions": uncertain_count,
        "top_intents": top_intents,
    }


def broadcast_metrics_update():
    channel_layer = get_channel_layer()
    if not channel_layer:
        return
    async_to_sync(channel_layer.group_send)(
        "metrics",
        {
            "type": "metrics_update",
            "data": metrics_summary(),
        },
    )
