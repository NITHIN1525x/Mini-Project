# chat/services.py
# Lazy import - only load when actually needed
_bot = None

def get_bot():
    """
    Lazy singleton: loads SBERT + TF model once per process.
    Only imports when first called to avoid startup memory issues.
    """
    global _bot
    if _bot is None:
        from src.utils import IntentPredictor
        _bot = IntentPredictor(threshold=0.35, nn_sim_threshold=0.55)
    return _bot
