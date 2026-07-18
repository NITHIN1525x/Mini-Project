# chat/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from django.db.models import Count, F

from .engine import broadcast_metrics_update, get_user_key, metrics_summary, process_chat_message
from .models import Message, PredictionMetric, UserFeedback

class HealthView(APIView):
    """Simple health check - no ML models loaded"""
    def get(self, request):
        return Response({"status": "ok", "service": "chatbot-api"})

@method_decorator(csrf_exempt, name='dispatch')
class PredictView(APIView):
    def post(self, request):
        from .services import get_bot
        text = (request.data.get("text") or "").strip()
        if not text:
            return Response({"error": "text is required"}, status=status.HTTP_400_BAD_REQUEST)
        bot = get_bot()
        out = bot.predict_intent(text)
        top3 = bot.top_k(text, 3)
        return Response({
            "tag": out["tag"],
            "confidence": out["confidence"],
            "top3": top3
        })

@method_decorator(csrf_exempt, name='dispatch')
class ChatView(APIView):
    """
    POST /api/chat/

    Request:
        {"text": "What are the college timings?", "lang": "en"}

    Response includes reply, intent, confidence, uncertainty, message_id,
    conversation_id, top3 predictions, and optional follow-up suggestions.
    """
    def post(self, request):
        text = (request.data.get("text") or "").strip()
        user_lang = (request.data.get("lang") or "en").strip()  # optional, for logging/analytics
        if not text:
            return Response({"error": "text is required"}, status=status.HTTP_400_BAD_REQUEST)
        payload = process_chat_message(text, get_user_key(request), user_lang)
        return Response(payload)


@method_decorator(csrf_exempt, name='dispatch')
class FeedbackView(APIView):
    """Capture user ratings and corrected intents for active learning."""
    def post(self, request):
        message_id = request.data.get("message_id")
        corrected_intent = (request.data.get("corrected_intent") or "").strip()
        rating = int(request.data.get("rating", 3))
        feedback_text = (request.data.get("feedback_text") or "").strip()

        if rating not in [1, 2, 3, 4]:
            return Response({"error": "rating must be between 1 and 4"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            message = Message.objects.get(id=message_id)
        except Message.DoesNotExist:
            return Response({"error": "Message not found"}, status=status.HTTP_404_NOT_FOUND)

        feedback = UserFeedback.objects.create(
            message=message,
            predicted_intent=message.intent_tag,
            corrected_intent=corrected_intent,
            user_rating=rating,
            feedback_text=feedback_text,
        )
        latest_metric = PredictionMetric.objects.filter(
            conversation=message.conversation,
            intent=message.intent_tag,
        ).order_by("-timestamp").first()
        if latest_metric:
            latest_metric.user_satisfied = rating >= 3
            latest_metric.save(update_fields=["user_satisfied"])
        broadcast_metrics_update()

        return Response({
            "status": "feedback_saved",
            "feedback_id": feedback.id,
            "message": "Feedback saved. These corrections can be used for retraining.",
        })


class MetricsView(APIView):
    """Return production-style prediction analytics for dashboards."""
    def get(self, request):
        return Response(metrics_summary())


class ActiveLearningView(APIView):
    """Show the most common corrected-intent pairs for retraining priority."""
    def get(self, request):
        mistakes = (
            UserFeedback.objects.exclude(corrected_intent="")
            .exclude(predicted_intent=F("corrected_intent"))
            .values("predicted_intent", "corrected_intent")
            .annotate(count=Count("id"))
            .order_by("-count")[:10]
        )
        return Response({"misclassified_examples": list(mistakes)})
        
# Cleaned up templates. Backend is now REST API-only.
