# chat/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .services import get_bot

class HealthView(APIView):
    def get(self, request):
        return Response({"status": "ok"})

class PredictView(APIView):
    def post(self, request):
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

class ChatView(APIView):
    def post(self, request):
        text = (request.data.get("text") or "").strip()
        user_lang = (request.data.get("lang") or "en").strip()  # optional, for logging/analytics
        if not text:
            return Response({"error": "text is required"}, status=status.HTTP_400_BAD_REQUEST)
        bot = get_bot()
        out = bot.predict_intent(text)
        reply = bot.answer(text)
        return Response({
            "reply": reply,
            "tag": out["tag"],
            "confidence": out["confidence"],
            "lang": user_lang
        })
        
# Template page for the chat UI
from django.views.generic import TemplateView

class ChatPageView(TemplateView):
    template_name = "chat.html"

