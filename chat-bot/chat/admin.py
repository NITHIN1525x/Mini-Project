from django.contrib import admin

from .models import Conversation, Message, PredictionMetric, UserFeedback


@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("id", "user_id", "created_at", "updated_at")
    search_fields = ("user_id",)


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("id", "conversation", "intent_tag", "confidence", "uncertainty", "timestamp")
    list_filter = ("intent_tag", "is_uncertain")
    search_fields = ("user_text", "bot_response", "intent_tag")


@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ("id", "message", "predicted_intent", "corrected_intent", "user_rating", "created_at")
    list_filter = ("user_rating", "predicted_intent", "corrected_intent")
    search_fields = ("feedback_text", "predicted_intent", "corrected_intent")


@admin.register(PredictionMetric)
class PredictionMetricAdmin(admin.ModelAdmin):
    list_display = ("id", "intent", "confidence", "uncertainty", "user_satisfied", "timestamp")
    list_filter = ("intent", "user_satisfied")
