"""
negotiation_engine.py
---------------------
Manages negotiation strategy and offer sequencing.
Adapts based on sentiment signals and customer responses.
"""

import logging
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)


class NegotiationEngine:
    """
    Drives the negotiation flow during a debt collection call.

    Strategy tiers (applied in order as resistance increases):
        1. Full payment request
        2. Partial payment (50%+)
        3. Extended due date (30–60 days)
        4. Installment plan (3–6 months)
        5. Hardship arrangement (longer term, lower amounts)
    """

    STRATEGIES = [
        {
            "tier": 1,
            "name": "full_payment",
            "label": "Full Payment",
            "description": "Request full payment of the outstanding balance.",
            "prompt_hint": (
                "Ask the customer if they can pay the full balance of ${amount} today or by a specific date. "
                "Keep it simple and direct."
            ),
        },
        {
            "tier": 2,
            "name": "partial_payment",
            "label": "Partial Payment",
            "description": "Offer to accept 50% of the balance now.",
            "prompt_hint": (
                "Offer to accept ${partial_amount} (50% of the balance) as an immediate payment "
                "to show good faith, with the remainder due in 30 days."
            ),
        },
        {
            "tier": 3,
            "name": "extension",
            "label": "Payment Extension",
            "description": "Offer a 30-day extension on the due date.",
            "prompt_hint": (
                "Offer a 30-day extension — the customer would have until {extension_date} to pay "
                "the full amount of ${amount} without any additional fees."
            ),
        },
        {
            "tier": 4,
            "name": "installment_plan",
            "label": "Installment Plan",
            "description": "Break the debt into 3 or 6 monthly payments.",
            "prompt_hint": (
                "Offer a 3-month installment plan: ${installment_3m}/month, "
                "or a 6-month plan at ${installment_6m}/month. "
                "Ask which works better for their budget."
            ),
        },
        {
            "tier": 5,
            "name": "hardship_arrangement",
            "label": "Hardship Arrangement",
            "description": "Long-term low-payment arrangement for extreme cases.",
            "prompt_hint": (
                "Acknowledge the customer's difficulty. Offer a 12-month hardship plan "
                "at ${installment_12m}/month. Express that this is the most flexible option available."
            ),
        },
    ]

    def __init__(self, customer_data: dict):
        self.customer_data = customer_data
        self.debt_amount = float(customer_data.get("debt_amount", 0))

        self.current_tier = 1
        self.offers_made: List[str] = []
        self.last_rejection_reason: Optional[str] = None
        self.sentiment_pressure = 0  # increases with anger/frustration

        self._precompute_amounts()

    def _precompute_amounts(self):
        """Precompute offer amounts based on total debt."""
        self.amounts = {
            "full": self.debt_amount,
            "partial_50": round(self.debt_amount * 0.50, 2),
            "installment_3m": round(self.debt_amount / 3, 2),
            "installment_6m": round(self.debt_amount / 6, 2),
            "installment_12m": round(self.debt_amount / 12, 2),
        }

    def get_current_strategy(self) -> Dict[str, Any]:
        """Return the current negotiation strategy tier."""
        idx = min(self.current_tier - 1, len(self.STRATEGIES) - 1)
        strategy = dict(self.STRATEGIES[idx])

        # Hydrate prompt_hint with actual amounts
        from datetime import datetime, timedelta
        ext_date = (datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")

        strategy["prompt_hint"] = strategy["prompt_hint"].format(
            amount=self.debt_amount,
            partial_amount=self.amounts["partial_50"],
            extension_date=ext_date,
            installment_3m=self.amounts["installment_3m"],
            installment_6m=self.amounts["installment_6m"],
            installment_12m=self.amounts["installment_12m"],
        )
        return strategy

    def escalate(self, rejection_reason: Optional[str] = None):
        """
        Move to the next negotiation tier after a rejection.

        Args:
            rejection_reason: Why the customer rejected the current offer
        """
        self.last_rejection_reason = rejection_reason
        if self.current_tier < len(self.STRATEGIES):
            prev = self.current_tier
            self.current_tier += 1
            strategy = self.get_current_strategy()
            self.offers_made.append(strategy["name"])
            logger.info(
                f"Negotiation escalated: tier {prev} → {self.current_tier} "
                f"({strategy['label']}) | Reason: {rejection_reason}"
            )
        else:
            logger.info("Negotiation exhausted: all tiers offered.")

    def handle_sentiment_signal(self, sentiment_label: str):
        """
        Adjust strategy based on incoming sentiment.
        High anger can cause skipping aggressive tiers.
        """
        if sentiment_label in ("anger", "frustration"):
            self.sentiment_pressure += 1
            # Skip to more accommodating offers under high pressure
            if self.sentiment_pressure >= 3 and self.current_tier < 4:
                logger.info("High sentiment pressure: jumping to installment plan tier.")
                self.current_tier = 4
        elif sentiment_label == "cooperative":
            self.sentiment_pressure = max(0, self.sentiment_pressure - 1)

    def is_exhausted(self) -> bool:
        """Return True if all negotiation tiers have been tried."""
        return self.current_tier > len(self.STRATEGIES)

    def get_context(self) -> Dict[str, Any]:
        """Return current negotiation context for LLM prompt injection."""
        return {
            "current_tier": self.current_tier,
            "current_strategy": self.get_current_strategy() if not self.is_exhausted() else None,
            "offers_made": self.offers_made,
            "debt_amounts": self.amounts,
            "sentiment_pressure": self.sentiment_pressure,
            "exhausted": self.is_exhausted(),
        }

    def get_tone_instruction(self) -> str:
        """
        Return a tone instruction for the LLM based on current sentiment pressure.
        """
        if self.sentiment_pressure >= 4:
            return (
                "The customer is very frustrated. Be extremely empathetic, apologize sincerely, "
                "and focus entirely on finding a solution that works for them. Do not push for payment."
            )
        elif self.sentiment_pressure >= 2:
            return (
                "The customer is showing frustration. Soften your tone, acknowledge their feelings, "
                "and pivot toward a more flexible offer."
            )
        else:
            return "Maintain a professional, friendly tone. Be direct but warm."
