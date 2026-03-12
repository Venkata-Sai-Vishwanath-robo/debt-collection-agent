"""
conversation_manager.py
------------------------
Tracks conversation state, sentiment history, transcript turns,
and derives call outcomes from conversation signals.
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


# Keyword maps for outcome detection
OUTCOME_SIGNALS = {
    "promise_to_pay": [
        "i'll pay", "i will pay", "i can pay", "i'll send", "i'll transfer",
        "yes i'll do it", "set up a payment", "pay tomorrow", "pay friday",
        "pay next week", "payment plan", "installment", "pay in full",
    ],
    "refused": [
        "i'm not paying", "won't pay", "refuse to pay", "i don't owe",
        "you're not getting", "can't help you", "goodbye", "i'm hanging up",
        "stop calling me", "never paying",
    ],
    "dispute": [
        "i don't owe", "this is wrong", "incorrect amount", "not my debt",
        "never had an account", "fraud", "identity theft", "dispute",
        "that's not right", "wrong amount", "erroneous",
    ],
    "already_paid": [
        "already paid", "paid this", "paid last", "paid it off",
        "sent payment", "cleared this", "i paid", "payment was made",
    ],
    "call_dropped": [],  # Detected by connection events, not text
}


class ConversationManager:
    """
    Manages all stateful aspects of a debt collection conversation.
    Tracks outcomes, sentiment, negotiation signals, and builds transcripts.
    """

    def __init__(self, customer_data: dict, on_update: Optional[callable] = None):
        self.customer_data = customer_data
        self.start_time = datetime.now()
        self.on_update = on_update

        # Transcript storage
        self.transcript: List[Dict[str, Any]] = []

        # Conversation state
        self.state = {
            "customer_name": customer_data.get("name", "Unknown"),
            "debt_amount": customer_data.get("debt_amount", 0.0),
            "call_outcome": "in_progress",
            "payment_intent": False,
            "payment_promise_date": None,
            "payment_amount_agreed": None,
            "plan_type": None,  # full | partial | installment | extension
            "refusal_count": 0,
            "dispute_raised": False,
            "already_paid_claimed": False,
            "opt_out_requested": False,
            "sentiment_history": [],
            "current_sentiment": "neutral",
            "dominant_sentiment": "neutral",
            "turn_count": 0,
            "customer_turns": 0,
            "agent_turns": 0,
            "call_duration_seconds": 0,
            "anger_escalations": 0,
            "cooperative_signals": 0,
        }

    def add_transcript_turn(self, role: str, text: str, sentiment: Optional[Dict] = None):
        """Add a single conversation turn to the transcript."""
        turn = {
            "turn": len(self.transcript) + 1,
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "sentiment": sentiment,
        }
        self.transcript.append(turn)

        self.state["turn_count"] += 1
        if role == "customer":
            self.state["customer_turns"] += 1
            self._analyze_customer_text(text)
        elif role == "agent":
            self.state["agent_turns"] += 1

        logger.debug(f"Transcript turn added: [{role}] {text[:60]}...")

        if self.on_update:
            asyncio.create_task(
                self.on_update({"type": "transcript_update", "turn": turn})
            )

    def update_sentiment(self, sentiment: Dict):
        """Update running sentiment state from latest analysis."""
        self.state["current_sentiment"] = sentiment.get("label", "neutral")
        self.state["sentiment_history"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "label": sentiment.get("label"),
                "score": sentiment.get("score"),
            }
        )

        # Track escalations
        if sentiment.get("label") in ("anger", "frustration"):
            self.state["anger_escalations"] += 1
        elif sentiment.get("label") in ("neutral", "cooperative"):
            self.state["cooperative_signals"] += 1

        # Update dominant sentiment
        self._recalculate_dominant_sentiment()

        if self.on_update:
            asyncio.create_task(
                self.on_update({
                    "type": "sentiment_update",
                    "sentiment": sentiment,
                    "label": sentiment.get("label"),
                    "score": sentiment.get("score"),
                })
            )

    def _recalculate_dominant_sentiment(self):
        """Find the most frequent sentiment label in history."""
        history = self.state["sentiment_history"]
        if not history:
            return
        counts: Dict[str, int] = {}
        for entry in history:
            label = entry.get("label", "neutral")
            counts[label] = counts.get(label, 0) + 1
        self.state["dominant_sentiment"] = max(counts, key=counts.get)

    def _analyze_customer_text(self, text: str):
        """Scan customer text for outcome signals and update state."""
        text_lower = text.lower()

        for outcome, keywords in OUTCOME_SIGNALS.items():
            for kw in keywords:
                if kw in text_lower:
                    if outcome == "refused":
                        self.state["refusal_count"] += 1
                        if self.state["refusal_count"] >= 2:
                            self._set_outcome("refused")
                    elif outcome == "dispute":
                        self.state["dispute_raised"] = True
                        self._set_outcome("dispute")
                    elif outcome == "already_paid":
                        self.state["already_paid_claimed"] = True
                        self._set_outcome("already_paid")
                    elif outcome == "promise_to_pay":
                        self.state["payment_intent"] = True
                        if self.state["call_outcome"] == "in_progress":
                            self._set_outcome("promise_to_pay")
                    break

        # Opt-out detection
        opt_out_phrases = ["stop calling", "do not call", "remove me", "opt out", "take me off"]
        for phrase in opt_out_phrases:
            if phrase in text_lower:
                self.state["opt_out_requested"] = True
                logger.warning(f"OPT-OUT requested by customer: '{text}'")
                break

    def _set_outcome(self, outcome: str):
        """Set the call outcome if not already finalized."""
        finalized = {"promise_to_pay", "refused", "dispute", "already_paid"}
        if self.state["call_outcome"] not in finalized:
            self.state["call_outcome"] = outcome
            logger.info(f"Call outcome set to: {outcome}")

    def set_payment_details(
        self,
        plan_type: str,
        amount: Optional[float] = None,
        promise_date: Optional[str] = None,
    ):
        """Record specific payment arrangement details."""
        self.state["plan_type"] = plan_type
        self.state["payment_amount_agreed"] = amount
        self.state["payment_promise_date"] = promise_date
        self.state["payment_intent"] = True
        self._set_outcome("promise_to_pay")

    def mark_call_dropped(self):
        """Mark call as dropped due to connection loss."""
        if self.state["call_outcome"] == "in_progress":
            self._set_outcome("call_dropped")

    def get_state(self) -> Dict[str, Any]:
        """Return current conversation state snapshot."""
        self.state["call_duration_seconds"] = (
            datetime.now() - self.start_time
        ).total_seconds()
        return dict(self.state)

    def get_full_transcript(self) -> List[Dict[str, Any]]:
        """Return complete transcript."""
        return self.transcript

    def generate_summary(self, risk_score: float) -> Dict[str, Any]:
        """
        Generate a structured call summary report.

        Args:
            risk_score: Predicted repayment probability (0.0–1.0)
        """
        state = self.get_state()

        # Build human-readable notes
        notes_parts = []
        if state["dispute_raised"]:
            notes_parts.append("Customer raised a debt dispute.")
        if state["already_paid_claimed"]:
            notes_parts.append("Customer claims debt was already paid.")
        if state["opt_out_requested"]:
            notes_parts.append("Customer requested opt-out from calls.")
        if state["anger_escalations"] > 2:
            notes_parts.append(f"Customer showed anger {state['anger_escalations']} times.")
        if state["plan_type"]:
            notes_parts.append(f"Payment plan agreed: {state['plan_type']}.")
        if state["payment_promise_date"]:
            notes_parts.append(f"Customer promised payment by: {state['payment_promise_date']}.")

        return {
            "session_id": None,  # filled by caller
            "customer_name": state["customer_name"],
            "debt_amount": f"${state['debt_amount']:.2f}",
            "call_outcome": state["call_outcome"],
            "plan_type": state["plan_type"],
            "payment_amount_agreed": state["payment_amount_agreed"],
            "payment_promise_date": state["payment_promise_date"],
            "sentiment": state["dominant_sentiment"],
            "risk_score": round(risk_score, 4),
            "risk_label": self._risk_label(risk_score),
            "call_duration_seconds": round(state["call_duration_seconds"], 1),
            "turn_count": state["turn_count"],
            "refusal_count": state["refusal_count"],
            "anger_escalations": state["anger_escalations"],
            "opt_out_requested": state["opt_out_requested"],
            "notes": " ".join(notes_parts) if notes_parts else "Call completed without notable events.",
            "payment_likelihood_pct": round(risk_score * 100, 1),
            "callback_suggestion": self.suggest_callback(state, risk_score),
            "generated_at": datetime.now().isoformat(),
        }

    @staticmethod
    def _risk_label(score: float) -> str:
        if score >= 0.75:
            return "LOW_RISK"
        elif score >= 0.45:
            return "MEDIUM_RISK"
        else:
            return "HIGH_RISK"

    @staticmethod
    def suggest_callback(state: Dict[str, Any], risk_score: float) -> Dict[str, Any]:
        """
        Suggest whether to call back, when, and at what time of day.
        Returns a dict with: should_callback, days_to_wait, best_days, best_time, reason.
        """
        outcome = state.get("call_outcome", "in_progress")
        sentiment = state.get("dominant_sentiment", "neutral")
        anger = state.get("anger_escalations", 0)
        refusals = state.get("refusal_count", 0)
        opt_out = state.get("opt_out_requested", False)

        # Hard stops — do not call back
        if opt_out:
            return {
                "should_callback": False,
                "days_to_wait": None,
                "best_days": [],
                "best_time": None,
                "reason": "Customer opted out. Do not call.",
            }
        if outcome == "dispute":
            return {
                "should_callback": False,
                "days_to_wait": None,
                "best_days": [],
                "best_time": None,
                "reason": "Dispute raised. Pause all contact until resolved.",
            }
        if outcome == "already_paid":
            return {
                "should_callback": False,
                "days_to_wait": None,
                "best_days": [],
                "best_time": None,
                "reason": "Payment claimed. Verify with accounts team before any follow-up.",
            }
        if outcome == "promise_to_pay":
            return {
                "should_callback": True,
                "days_to_wait": 7,
                "best_days": ["Tuesday", "Wednesday", "Thursday"],
                "best_time": "10:00 AM – 12:00 PM",
                "reason": "Payment promised. Follow up in 7 days to confirm receipt.",
            }

        # Determine days to wait based on risk + sentiment
        if risk_score >= 0.70:
            days = 3
            time_slot = "10:00 AM – 12:00 PM"
            reason = "High payment likelihood. Strike while engagement is warm."
        elif risk_score >= 0.50:
            days = 5
            time_slot = "5:00 PM – 7:00 PM"
            reason = "Moderate likelihood. Give breathing room, then follow up."
        else:
            days = 7 if anger < 2 else 10
            time_slot = "10:00 AM – 11:00 AM"
            reason = "Low likelihood. Allow time to cool off before re-engaging."

        if refusals >= 3:
            days = max(days, 10)
            reason += " Multiple refusals — wait longer before retry."

        # Best days to call (avoid Mondays and Fridays)
        best_days = ["Tuesday", "Wednesday", "Thursday"]
        if sentiment == "cooperative":
            best_days = ["Tuesday", "Wednesday"]
            days = max(days - 1, 1)

        return {
            "should_callback": True,
            "days_to_wait": days,
            "best_days": best_days,
            "best_time": time_slot,
            "reason": reason,
        }
