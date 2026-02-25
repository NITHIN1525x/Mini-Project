# chat/services.py
from src.utils import IntentPredictor

_bot = None

def get_bot():
    """
    Lazy singleton: loads SBERT + TF model once per process.
    """
    global _bot
    if _bot is None:
        _bot = IntentPredictor(threshold=0.35, nn_sim_threshold=0.55)
    return _bot
