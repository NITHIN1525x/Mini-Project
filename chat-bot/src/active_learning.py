
import os
import django
import json
from django.db.models import Count, F
from pathlib import Path

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

# pyrefly: ignore [missing-import]
from chat.models import UserFeedback

class ActiveLearner:
    """
    Identifies misclassified examples from user feedback and 
    prepares them for retraining.
    """
    def __init__(self, data_path="data/intents.json"):
        self.data_path = Path(data_path)

    def group_mistakes(self):
        """Find patterns where the model often confuses two specific intents."""
        mistakes = (
            UserFeedback.objects.exclude(corrected_intent="")
            .exclude(predicted_intent=F("corrected_intent"))
            .values("predicted_intent", "corrected_intent")
            .annotate(count=Count("id"))
            .order_by("-count")
        )
        return list(mistakes)

    def export_correction_dataset(self, output_path="data/corrections.jsonl"):
        """Exports user-corrected messages for manual review or data augmentation."""
        corrections = UserFeedback.objects.exclude(corrected_intent="")
        count = 0
        with open(output_path, "w", encoding="utf-8") as f:
            for c in corrections:
                entry = {
                    "text": c.message.user_text,
                    "original_intent": c.predicted_intent,
                    "corrected_intent": c.corrected_intent,
                    "rating": c.user_rating,
                    "created_at": c.created_at.isoformat()
                }
                f.write(json.dumps(entry) + "\n")
                count += 1
        print(f"Exported {count} corrections to {output_path}")

    def update_intents_json(self):
        """
        Dynamically adds user corrections as new patterns to intents.json.
        Note: In production, this usually goes to a staging queue for review.
        """
        if not self.data_path.exists():
            print(f"Error: {self.data_path} not found.")
            return

        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        corrections = UserFeedback.objects.filter(user_rating__gte=3).exclude(corrected_intent="")
        added_count = 0
        
        # Map tag to intent object
        intent_map = {i["tag"]: i for i in data["intents"]}
        
        for c in corrections:
            tag = c.corrected_intent
            text = c.message.user_text
            if tag in intent_map:
                if text not in intent_map[tag]["patterns"]:
                    intent_map[tag]["patterns"].append(text)
                    added_count += 1
            else:
                # Create new intent if it doesn't exist
                data["intents"].append({
                    "tag": tag,
                    "patterns": [text],
                    "responses": [f"New intent {tag} learned from feedback. Needs response template."]
                })
                added_count += 1
                # Refresh map
                intent_map = {i["tag"]: i for i in data["intents"]}

        if added_count > 0:
            with open(self.data_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            print(f"Successfully added {added_count} new patterns to intents.json")
        else:
            print("No high-quality corrections found to update.")

if __name__ == "__main__":
    learner = ActiveLearner()
    print("Mistakes summary:")
    for m in learner.group_mistakes():
        print(f"Predicted '{m['predicted_intent']}' but user said '{m['corrected_intent']}' ({m['count']} times)")
    
    learner.export_correction_dataset()
    learner.update_intents_json()
