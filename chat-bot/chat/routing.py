from django.urls import re_path

from . import consumers


websocket_urlpatterns = [
    re_path(r"ws/chat/$", consumers.ChatConsumer.as_asgi()),
    re_path(r"ws/metrics/$", consumers.MetricsConsumer.as_asgi()),
]
