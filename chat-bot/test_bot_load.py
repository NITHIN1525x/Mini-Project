
import os
import django
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from chat.services import get_bot

try:
    print("Attempting to load bot...")
    bot = get_bot()
    print(f"Bot loaded: {type(bot).__name__}")
    
    test_text = "What are the college timings?"
    print(f"Testing prediction for: '{test_text}'")
    result = bot.answer_with_followup_suggestions(test_text)
    print("Result strategy: " + ("SBERT+TF" if "IntentPredictor" in str(type(bot)) else "Lightweight"))
    print(f"Reply: {result['reply']}")
    print(f"Tag: {result['tag']} (Confidence: {result['confidence']:.2f})")
    
except Exception as e:
    print(f"Error loading/testing bot: {e}")
    import traceback
    traceback.print_exc()
