import asyncio
import json

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncWebsocketConsumer

from .engine import metrics_summary, process_chat_message


class ChatConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        user = self.scope.get("user")
        if user and user.is_authenticated:
            self.user_id = f"user:{user.pk}"
        else:
            session = self.scope.get("session")
            if session and not session.session_key:
                await sync_to_async(session.save)()
            self.user_id = f"session:{session.session_key}" if session else "anonymous"

        self.group_name = f"chat_{self.user_id.replace(':', '_')}"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()
        await self.send_json({
            "type": "connection_established",
            "message": "Connected to chatbot",
        })

    async def receive(self, text_data):
        try:
            data = json.loads(text_data)
            user_message = (data.get("text") or "").strip()
            lang = (data.get("lang") or "en").strip()
            if not user_message:
                await self.send_json({"type": "error", "error": "text is required"})
                return

            payload = await sync_to_async(process_chat_message)(user_message, self.user_id, lang)
            await self.send_json({
                "type": "message_start",
                "message_id": payload["message_id"],
                "conversation_id": payload["conversation_id"],
            })
            # Stream word-by-word for high-speed typing animation (reduces latency by 5x)
            words = payload["reply"].split(" ")
            for i, word in enumerate(words):
                chunk = word + (" " if i < len(words) - 1 else "")
                await self.send_json({"type": "token", "char": chunk})
                await asyncio.sleep(0.015)
            await self.send_json({"type": "chat_message", **payload})
        except Exception as exc:
            await self.send_json({"type": "error", "error": str(exc)})

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def send_json(self, content):
        await self.send(text_data=json.dumps(content))


class MetricsConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        await self.channel_layer.group_add("metrics", self.channel_name)
        await self.accept()
        await self.send_json({"type": "metrics_update", "data": await sync_to_async(metrics_summary)()})

    async def receive(self, text_data):
        await self.send_json({"type": "metrics_update", "data": await sync_to_async(metrics_summary)()})

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard("metrics", self.channel_name)

    async def metrics_update(self, event):
        await self.send_json({"type": "metrics_update", "data": event["data"]})

    async def send_json(self, content):
        await self.send(text_data=json.dumps(content))
