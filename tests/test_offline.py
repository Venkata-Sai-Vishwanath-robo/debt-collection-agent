"""
tests/test_offline.py
---------------------
Offline tests — no API keys needed. Validates all logic modules.
Run with: pytest tests/test_offline.py -v
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.conversation_manager import ConversationManager
from agent.negotiation_engine import NegotiationEngine
from ml.sentiment_analysis import SentimentAnalyzer
from ml.risk_model import RiskPredictor


@pytest.fixture
def customer():
    return {"name": "John Smith", "debt_amount": 420.0, "account_number": "ACC-9823", "days_overdue": 45}

@pytest.fixture
def cm(customer):
    return ConversationManager(customer)

@pytest.fixture
def ne(customer):
    return NegotiationEngine(customer)

@pytest.fixture
def sa():
    return SentimentAnalyzer()

@pytest.fixture
def rp():
    return RiskPredictor()


# ── ConversationManager ───────────────────────────────────────────────────────

class TestConversationManager:
    def test_initial_state(self, cm):
        s = cm.get_state()
        assert s["customer_name"] == "John Smith"
        assert s["call_outcome"] == "in_progress"
        assert s["payment_intent"] is False

    def test_promise_to_pay(self, cm):
        cm.add_transcript_turn("customer", "I'll pay this tomorrow for sure.")
        assert cm.get_state()["call_outcome"] == "promise_to_pay"
        assert cm.get_state()["payment_intent"] is True

    def test_refusal(self, cm):
        cm.add_transcript_turn("customer", "I'm not paying this.")
        cm.add_transcript_turn("customer", "I won't pay. Goodbye.")
        assert cm.get_state()["refusal_count"] >= 2
        assert cm.get_state()["call_outcome"] == "refused"

    def test_dispute(self, cm):
        cm.add_transcript_turn("customer", "I don't owe this money, this is wrong.")
        assert cm.get_state()["dispute_raised"] is True
        assert cm.get_state()["call_outcome"] == "dispute"

    def test_already_paid(self, cm):
        cm.add_transcript_turn("customer", "I already paid this last month.")
        assert cm.get_state()["already_paid_claimed"] is True
        assert cm.get_state()["call_outcome"] == "already_paid"

    def test_opt_out(self, cm):
        cm.add_transcript_turn("customer", "Please stop calling me.")
        assert cm.get_state()["opt_out_requested"] is True

    def test_sentiment_update(self, cm):
        cm.update_sentiment({"label": "anger", "score": 0.92})
        assert cm.get_state()["current_sentiment"] == "anger"
        assert cm.get_state()["anger_escalations"] == 1

    def test_summary_generation(self, cm):
        cm.add_transcript_turn("customer", "I'll pay tomorrow.")
        summary = cm.generate_summary(risk_score=0.82)
        assert summary["call_outcome"] == "promise_to_pay"
        assert "risk_score" in summary
        assert "risk_label" in summary

    def test_call_dropped(self, cm):
        cm.mark_call_dropped()
        assert cm.get_state()["call_outcome"] == "call_dropped"


# ── NegotiationEngine ─────────────────────────────────────────────────────────

class TestNegotiationEngine:
    def test_initial_tier(self, ne):
        assert ne.current_tier == 1
        assert ne.get_current_strategy()["name"] == "full_payment"

    def test_escalation(self, ne):
        ne.escalate("cannot pay in full")
        assert ne.current_tier == 2
        assert ne.get_current_strategy()["name"] == "partial_payment"

    def test_amounts(self, ne):
        assert ne.amounts["full"] == 420.0
        assert ne.amounts["partial_50"] == 210.0
        assert abs(ne.amounts["installment_3m"] - 140.0) < 0.01

    def test_anger_jump(self, ne):
        for _ in range(3):
            ne.handle_sentiment_signal("anger")
        assert ne.current_tier >= 4

    def test_exhaustion(self, ne):
        for _ in range(10):
            ne.escalate()
        assert ne.is_exhausted()


# ── SentimentAnalyzer ─────────────────────────────────────────────────────────

class TestSentimentAnalyzer:
    def test_empty(self, sa):
        assert sa.analyze("")["label"] == "neutral"

    def test_anger_keywords(self, sa):
        r = sa.analyze("Stop harassing me, this is ridiculous!")
        assert r["label"] == "anger"

    def test_cooperative(self, sa):
        r = sa.analyze("Okay sure, I understand. Thank you.")
        assert r["label"] in ("cooperative", "neutral")

    def test_batch(self, sa):
        results = sa.batch_analyze(["I'm angry!", "Okay sounds good.", ""])
        assert len(results) == 3


# ── RiskPredictor ─────────────────────────────────────────────────────────────

class TestRiskPredictor:
    def test_loads(self, rp):
        assert rp.model is not None

    def test_range(self, rp):
        state = {
            "dominant_sentiment": "cooperative",
            "sentiment_history": [{"label": "cooperative", "score": 0.9}],
            "refusal_count": 0,
            "payment_intent": True,
            "dispute_raised": False,
            "already_paid_claimed": False,
            "call_duration_seconds": 120,
            "turn_count": 10,
            "opt_out_requested": False,
        }
        score = rp.predict(state, [])
        assert 0.0 <= score <= 1.0

    def test_high_risk(self, rp):
        state = {
            "dominant_sentiment": "anger",
            "sentiment_history": [{"label": "anger", "score": 0.95}] * 3,
            "refusal_count": 4,
            "payment_intent": False,
            "dispute_raised": True,
            "already_paid_claimed": False,
            "call_duration_seconds": 60,
            "turn_count": 5,
            "opt_out_requested": True,
        }
        assert rp.predict(state, []) < 0.5

    def test_low_risk(self, rp):
        state = {
            "dominant_sentiment": "cooperative",
            "sentiment_history": [{"label": "cooperative", "score": 0.92}] * 3,
            "refusal_count": 0,
            "payment_intent": True,
            "dispute_raised": False,
            "already_paid_claimed": False,
            "call_duration_seconds": 180,
            "turn_count": 14,
            "opt_out_requested": False,
        }
        assert rp.predict(state, []) > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
