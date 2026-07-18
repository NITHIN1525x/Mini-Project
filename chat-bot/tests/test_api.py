
import pytest
from django.urls import reverse
from rest_framework.test import APIClient
from chat.models import Conversation, Message

@pytest.fixture
def api_client():
    return APIClient()

@pytest.mark.django_db
def test_chat_api_endpoint(api_client):
    """Test the POST /api/chat/ endpoint."""
    url = reverse('chat-api')
    data = {"text": "hello"}
    response = api_client.post(url, data, format='json')
    
    assert response.status_code == 200
    assert "reply" in response.data
    assert "tag" in response.data

@pytest.mark.django_db
def test_feedback_api_endpoint(api_client):
    """Test submitting user feedback via API."""
    # First create a message to give feedback on
    conv = Conversation.objects.create(user_id="test_user")
    msg = Message.objects.create(
        conversation=conv, 
        user_text="hi", 
        bot_response="hello", 
        intent_tag="greetings", 
        confidence=1.0
    )
    
    url = reverse('feedback-api')
    data = {
        "message_id": msg.id,
        "corrected_intent": "greetings",
        "rating": 4,
        "feedback_text": "Perfect answer"
    }
    response = api_client.post(url, data, format='json')
    
    assert response.status_code == 200
    assert "status" in response.data

@pytest.mark.django_db
def test_metrics_api_endpoint(api_client):
    """Test the metrics summary endpoint."""
    url = reverse('metrics-api')
    response = api_client.get(url)
    
    assert response.status_code == 200
    assert "total_predictions" in response.data
    assert "avg_confidence" in response.data

@pytest.mark.django_db
def test_active_learning_endpoint(api_client):
    """Test the active learning mistakes endpoint."""
    url = reverse('active-learning-api')
    response = api_client.get(url)
    
    assert response.status_code == 200
    assert "misclassified_examples" in response.data
