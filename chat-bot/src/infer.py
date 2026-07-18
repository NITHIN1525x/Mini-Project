import sys
from pathlib import Path

# Add src directory to system path for importing utils
sys.path.append(str(Path(__file__).resolve().parent))

from utils import IntentPredictor

if __name__ == "__main__":
    predictor = IntentPredictor()
    while True:
        q = input("You: ").strip()
        if not q or q.lower() in {"quit","exit"}:
            break
        print(predictor.predict_intent(q))
        