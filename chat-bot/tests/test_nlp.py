"""
tests/test_nlp.py
------------------
Tests for the core NLP intent predictor (src/utils.py).
"""
import pytest
from src.utils import IntentPredictor


@pytest.fixture(scope="session")
def predictor():
    """Load predictor once for the whole test session (expensive)."""
    return IntentPredictor(threshold=0.40)


# ── Greeting / Goodbye / Thanks ───────────────────────────────────────────────

def test_greeting_intent(predictor):
    result = predictor.predict_intent("Hello there")
    assert result["tag"] == "greetings"
    assert result["confidence"] > 0.4


def test_greeting_hi(predictor):
    result = predictor.predict_intent("Hi")
    assert result["tag"] == "greetings"


def test_goodbye_intent(predictor):
    result = predictor.predict_intent("Goodbye")
    assert result["tag"] == "goodbye"


def test_thanks_intent(predictor):
    result = predictor.predict_intent("Thank you so much")
    assert result["tag"] == "thanks"


# ── HOD / Faculty ─────────────────────────────────────────────────────────────

def test_hod_cs_full_question(predictor):
    """Classic HOD question must resolve to hod_cs, not the generic faculty tag."""
    result = predictor.predict_intent("Who is the HOD of computer science?")
    assert result["tag"] == "hod_cs", (
        f"Expected hod_cs, got {result['tag']} (conf={result['confidence']:.3f})"
    )
    assert result["confidence"] >= 0.9


def test_hod_cs_short(predictor):
    result = predictor.predict_intent("CSE HOD")
    assert result["tag"] == "hod_cs"
    assert result["confidence"] >= 0.9


def test_hod_cs_lowercase(predictor):
    result = predictor.predict_intent("who is hod of computer science")
    assert result["tag"] == "hod_cs"


def test_hod_cs_head_phrasing(predictor):
    result = predictor.predict_intent("who is the head of computer science department")
    assert result["tag"] == "hod_cs"


def test_hod_cs_answer_contains_name(predictor):
    """The answer must mention Dr. Mustafa Basthikodi."""
    ans = predictor.answer("Who is the HOD of computer science?")
    assert "mustafa" in ans.lower() or "basthikodi" in ans.lower(), (
        f"Name not found in answer: {ans}"
    )


def test_faculty_info(predictor):
    result = predictor.predict_intent("List of faculty members")
    assert result["tag"] == "faculty_info"


# ── College Timing ────────────────────────────────────────────────────────────

def test_college_timing_intent(predictor):
    result = predictor.predict_intent("What time does college open?")
    assert result["tag"] in ("college_timing", "academics_yearwise_timings")


def test_first_year_timing(predictor):
    result = predictor.predict_intent("What are the first year timings?")
    assert result["tag"] in ("first_year_timings", "academics_yearwise_timings")


# ── Fees ───────────────────────────────────────────────────────────────────────

def test_fee_structure(predictor):
    result = predictor.predict_intent("What is the fee structure?")
    assert result["tag"] == "fees"


def test_comedk_fees(predictor):
    result = predictor.predict_intent("What are COMEDK fees?")
    assert result["tag"] == "fees"


# ── Placement ─────────────────────────────────────────────────────────────────

def test_placement_intent(predictor):
    result = predictor.predict_intent("Placement details please")
    assert result["tag"] in ("placement", "placement_stats", "placement_training")


def test_highest_package(predictor):
    result = predictor.predict_intent("What is the highest package offered?")
    assert result["tag"] == "placement_stats"


# ── Hostel ────────────────────────────────────────────────────────────────────

def test_hostel_intent(predictor):
    result = predictor.predict_intent("Is hostel available?")
    assert result["tag"] in ("hostel", "hostel_registration", "hostel_rules")


def test_hostel_rules(predictor):
    result = predictor.predict_intent("What are the hostel rules?")
    assert result["tag"] == "hostel_rules"


# ── Library ───────────────────────────────────────────────────────────────────

def test_library_intent(predictor):
    result = predictor.predict_intent("What are the library timings?")
    assert result["tag"] == "library"


# ── Canteen ───────────────────────────────────────────────────────────────────

def test_canteen_intent(predictor):
    result = predictor.predict_intent("Is there a canteen?")
    assert result["tag"] == "canteen"


# ── Admissions ────────────────────────────────────────────────────────────────

def test_admissions_intent(predictor):
    result = predictor.predict_intent("What is the admission process?")
    assert result["tag"] == "admissions"


def test_attendance_rules(predictor):
    result = predictor.predict_intent("What is the minimum attendance required?")
    assert result["tag"] == "attendance_rules"


# ── Exams ─────────────────────────────────────────────────────────────────────

def test_exam_schedule(predictor):
    result = predictor.predict_intent("When is the exam?")
    assert result["tag"] in ("exam_schedule", "internal_exams", "semester_exams")


def test_revaluation(predictor):
    result = predictor.predict_intent("How can I apply for revaluation?")
    assert result["tag"] == "revaluation_process"


# ── Fallback ──────────────────────────────────────────────────────────────────

def test_fallback_logic(predictor):
    """Gibberish should trigger low confidence or uncertainty flag."""
    result = predictor.get_prediction_with_uncertainty("zxvmnbzxmcvbjk123")
    assert result["confidence"] < 0.5 or result["is_uncertain"] is True


def test_answer_is_string(predictor):
    answer = predictor.answer("hi")
    assert isinstance(answer, str) and len(answer) > 0


def test_answer_with_followup_suggestions_structure(predictor):
    out = predictor.answer_with_followup_suggestions("Who is the HOD of computer science?")
    assert "reply" in out
    assert "tag" in out
    assert "confidence" in out
    assert "top3" in out
    assert out["tag"] == "hod_cs"
