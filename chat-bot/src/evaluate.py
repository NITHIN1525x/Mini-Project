
import os
import django
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
PLOTS_DIR = ROOT / "static" / "evaluation"
DATA_PATH = ROOT / "data" / "intents.json"
PLOTS_DIR.mkdir(exist_ok=True, parents=True)

class ModelEvaluator:
    def __init__(self, y_true, y_pred_probs, classes):
        self.y_true = np.array(y_true)
        self.y_pred_probs = np.array(y_pred_probs)
        self.y_pred = np.argmax(self.y_pred_probs, axis=1)
        self.classes = classes

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        plt.figure(figsize=(15, 12))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=self.classes, yticklabels=self.classes)
        plt.title('Intent Classifier Confusion Matrix')
        plt.ylabel('Ground Truth')
        plt.xlabel('Predicted Intent')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "confusion_matrix.png", dpi=150)
        print(f"Confusion matrix saved to {PLOTS_DIR / 'confusion_matrix.png'}")

    def plot_precision_recall(self):
        plt.figure(figsize=(12, 8))
        for i, cls in enumerate(self.classes):
            # Only plot top 10 class or any classes with significant presence for readability
            y_true_binary = (self.y_true == i).astype(int)
            if np.sum(y_true_binary) == 0: continue
            
            y_scores = self.y_pred_probs[:, i]
            precision, recall, _ = precision_recall_curve(y_true_binary, y_scores)
            plt.plot(recall, precision, label=cls)
            
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves per Intent')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "pr_curves.png", dpi=150)
        print(f"PR curves saved to {PLOTS_DIR / 'pr_curves.png'}")

    def generate_report(self):
        report = classification_report(self.y_true, self.y_pred, target_names=self.classes, output_dict=True)
        with open(PLOTS_DIR / "classification_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(classification_report(self.y_true, self.y_pred, target_names=self.classes))

if __name__ == "__main__":
    # pyrefly: ignore [missing-import]
    from chat.services import get_bot
    import json
    
    bot = get_bot()
    if hasattr(bot, "classes"):
        classes = bot.classes
        
        # Load patterns from intents.json
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            intents_json = json.load(f)
            
        y_true = []
        y_pred_probs = []
        
        print("Evaluating classifier on intent patterns dataset...")
        for intent in intents_json.get("intents", []):
            tag = intent["tag"]
            if tag not in classes:
                continue
            tag_idx = classes.index(tag)
            for pattern in intent.get("patterns", []):
                if not pattern.strip():
                    continue
                pred = bot.predict_intent(pattern)
                y_true.append(tag_idx)
                
                # Get probability vector in classes order
                probs = [pred["probs_by_tag"].get(c, 0.0) for c in classes]
                y_pred_probs.append(probs)
                
        if len(y_true) > 0:
            evaluator = ModelEvaluator(y_true, y_pred_probs, classes)
            evaluator.plot_confusion_matrix()
            evaluator.plot_precision_recall()
            evaluator.generate_report()
            print("Evaluation completed successfully on patterns dataset!")
        else:
            print("No patterns found to evaluate.")
    else:
        print("Model is not trained. LightweightIntentPredictor is active, skipping advanced evaluation.")
