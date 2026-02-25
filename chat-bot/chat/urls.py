# chat/urls.py
from django.urls import path
from .views import HealthView, PredictView, ChatView,ChatPageView


urlpatterns = [
    path("", ChatPageView.as_view(), name="chat_page"), 
    path("health/", HealthView.as_view(), name="health"),
    path("predict/", PredictView.as_view(), name="predict"),
    path("chat/", ChatView.as_view(), name="chat"),
]
