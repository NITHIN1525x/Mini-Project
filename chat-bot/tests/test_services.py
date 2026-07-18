"""
tests/test_services.py
-----------------------
Tests for chat/services.py — the Django backend predictor.
Runs without Django (no DB needed).
"""
import sys, os
# Make sure the chat package is importable without running Django
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from chat.services import LightweightIntentPredictor, _normalize_text, _score_pair


@pytest.fixture(scope="session")
def bot():
    return LightweightIntentPredictor(threshold=0.40)


# ── Normalization ─────────────────────────────────────────────────────────────

def test_normalize_lowercase():
    assert _normalize_text("Hello WORLD") == "hello world"


def test_normalize_punctuation():
    assert _normalize_text("Who is the HOD?") == "who is the hod"


def test_normalize_repeated_letters():
    result = _normalize_text("Heyyy there")
    assert "heyy" in result or "hey" in result  # collapsed


def test_normalize_extra_spaces():
    assert _normalize_text("  too   many   spaces  ") == "too many spaces"


# ── Scoring ───────────────────────────────────────────────────────────────────

def test_score_exact_match():
    score = _score_pair("Who is the HOD of computer science?", "Who is the HOD of computer science?")
    assert score >= 0.95


def test_score_case_insensitive():
    score = _score_pair("who is hod of cse", "Who is the HOD of CSE?")
    assert score >= 0.5


def test_score_unrelated_low():
    score = _score_pair("pizza recipe", "Who is the HOD of computer science?")
    assert score < 0.4


# ── HOD Prediction ───────────────────────────────────────────────────────────

def test_hod_prediction_full(bot):
    pred = bot.predict_intent("Who is the HOD of computer science?")
    assert pred["tag"] == "hod_cs", f"Got {pred['tag']} conf={pred['confidence']:.3f}"
    assert pred["confidence"] >= 0.9


def test_hod_prediction_short(bot):
    pred = bot.predict_intent("CSE HOD")
    assert pred["tag"] == "hod_cs"


def test_hod_prediction_lowercase(bot):
    pred = bot.predict_intent("who is hod of computer science")
    assert pred["tag"] == "hod_cs"


def test_hod_answer_name(bot):
    reply = bot.answer("Who is the HOD of computer science?")
    assert "mustafa" in reply.lower() or "basthikodi" in reply.lower(), (
        f"Name missing from reply: {reply}"
    )


# ── Faculty Info (separate from HOD) ─────────────────────────────────────────

def test_faculty_info(bot):
    pred = bot.predict_intent("List of faculty members")
    assert pred["tag"] == "faculty_info"


# ── Common Intents ────────────────────────────────────────────────────────────

def test_greeting(bot):
    pred = bot.predict_intent("Hi")
    assert pred["tag"] == "greetings"


def test_fees(bot):
    pred = bot.predict_intent("What is the fee structure?")
    assert pred["tag"] == "fees"


def test_hostel(bot):
    pred = bot.predict_intent("Is hostel available?")
    assert pred["tag"] in ("hostel", "hostel_registration", "hostel_rules")


def test_canteen(bot):
    pred = bot.predict_intent("Is there a canteen?")
    assert pred["tag"] == "canteen"


def test_placement(bot):
    pred = bot.predict_intent("Placement details please")
    assert pred["tag"] in ("placement", "placement_stats", "placement_training")


def test_library(bot):
    pred = bot.predict_intent("What are the library timings?")
    assert pred["tag"] == "library"


def test_college_timing(bot):
    pred = bot.predict_intent("What are the college timings?")
    assert pred["tag"] in ("college_timing", "academics_yearwise_timings")


# ── Fallback ──────────────────────────────────────────────────────────────────

def test_fallback_gibberish(bot):
    reply = bot.answer("asdfghjklqwerty")
    assert "sorry" in reply.lower() or "rephrase" in reply.lower()


# ── answer_with_followup_suggestions ─────────────────────────────────────────

def test_followup_structure(bot):
    out = bot.answer_with_followup_suggestions("Who is the HOD of computer science?")
    assert "reply" in out
    assert "tag" in out
    assert "confidence" in out
    assert "top3" in out
    assert out["tag"] == "hod_cs"


def test_history_doesnt_corrupt_hod(bot):
    """Conversation history about placement must NOT contaminate a clear HOD question."""
    history = [
        "What companies visit campus?",
        "Infosys, TCS, Wipro, Accenture visit us."
    ]
    out = bot.answer_with_followup_suggestions(
        "Who is the HOD of computer science?",
        conversation_history=history
    )
    assert out["tag"] == "hod_cs", (
        f"History contaminated the result — got {out['tag']} conf={out['confidence']:.3f}"
    )
