# src/chat.py
"""
Flask web server for the chatbot interface (English-only).
This simplified server removes translation logic and processes text as English.
"""
from pathlib import Path
from flask import Flask, request, jsonify, render_template
from utils import IntentPredictor

# Initialize Flask app with proper template and static paths
app = Flask(__name__,
           template_folder=Path(__file__).resolve().parents[1] / "templates",
           static_folder=Path(__file__).resolve().parents[1] / "static")

# Initialize chatbot
predictor = IntentPredictor(threshold=0.45)

@app.route("/")
def home():
    return render_template("chat.html")

@app.route("/api/chat/", methods=["POST"])
def chat():
    data = request.json
    text = data.get("text", "").strip()
    # Always treat input as English-only (no translation)
    if not text:
        return jsonify({"error": "Empty message"})

    try:
        pred = predictor.predict_intent(text)
        response = predictor.answer(text)
        return jsonify({
            "reply": response,
            "tag": pred["tag"],
            "confidence": pred["confidence"]
        })
    except Exception as e:
        return jsonify({"error": "Internal server error", "details": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
