# chat/urls.py
from django.urls import path
from .views import (
    ActiveLearningView,
    ChatView,
    FeedbackView,
    HealthView,
    MetricsView,
    PredictView,
)


urlpatterns = [
    path("", HealthView.as_view(), name="health"), 
    path("health/", HealthView.as_view(), name="health"),
    path("predict/", PredictView.as_view(), name="predict"),
    path("chat/", ChatView.as_view(), name="chat-api"),
    path("feedback/", FeedbackView.as_view(), name="feedback-api"),
    path("metrics/", MetricsView.as_view(), name="metrics-api"),
    path("active-learning/", ActiveLearningView.as_view(), name="active-learning-api"),
]
