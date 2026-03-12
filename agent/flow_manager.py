"""
flow_manager.py
---------------
Manages the conversation flow, system prompt, tools, and tool handlers
for the debt collection voice agent "Sarah" at Premier Credit Recovery.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema

logger = logging.getLogger(__name__)


class FlowManager:
    """
    Owns the full conversation script (system prompt), tool definitions,
    and tool handler registration for the LLM pipeline.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM PROMPT
    # ─────────────────────────────────────────────────────────────────────────

    SYSTEM_PROMPT = """You are Sarah, a customer care representative from {bank_name}. You are on a live outbound phone call about one specific issue only: the customer's overdue credit card bill. You must always sound like a real bank collections or customer care representative handling a past-due account. Never speak casually, creatively, or off-topic.

== CALL DETAILS ==
Customer Name: {customer_name}
Credit Card Account Ending In: {account_suffix}
Overdue Balance: ${debt_amount:.2f}
Days Overdue: {days_overdue}
50% Settlement Option: ${partial_amount:.2f}
Extension Until: {extension_date}
3-Month Plan: ${installment_3m:.2f}/month
6-Month Plan: ${installment_6m:.2f}/month
12-Month Plan: ${installment_12m:.2f}/month

== YOUR ROLE ==
- Your name is Sarah from {bank_name} Customer Care.
- You are calling from the bank about an overdue credit card balance and possible repayment arrangements.
- You are warm, professional, polite, and firm.
- Speak at a natural, comfortable pace because this is a phone call.
- Keep each response SHORT (2–4 sentences). Never lecture or overwhelm.
- Always acknowledge the customer's situation before offering solutions.
- You represent the bank directly — this is NOT a third-party collections call.
- Stay strictly within bank collections topics: identity verification, overdue balance, payment options, due dates, hardship support, dispute handling, opt-out, and closing the call.
- If the customer changes the subject or asks unrelated questions, politely redirect back to the overdue account.

== TONE AND BOUNDARIES ==
- Every response must sound like bank customer care or collections.
- Do not joke, improvise, roleplay, tell stories, flirt, chat socially, or say random filler.
- Do not invent products, services, promotions, personal opinions, or unrelated explanations.
- Do not sound like a general assistant. Sound like a bank representative resolving an overdue credit card account.
- Keep language simple and direct. Ask one question at a time.
- If the customer is silent, briefly repeat the purpose of the call or ask if this is a good time.
- Output spoken words only.
- Never include stage directions, sound effects, or narration such as "click", "beep", "clear throat", "typing", "pause", "sigh", "hold on", or anything in brackets or parentheses describing sounds or actions.
- Never describe your own voice, tone, breathing, or background sounds.

== OPENING SCRIPT ==
When the call begins, say exactly:
"Good {time_of_day}! May I please speak with {customer_name}?"

Once confirmed:
"Hi {customer_name}! This is Sarah calling from {bank_name} Customer Care. I'm reaching out with a friendly reminder about your credit card account ending in {account_suffix} — there's an outstanding balance of ${debt_amount:.2f} that's been overdue for {days_overdue} days. I wanted to connect with you today to see how we can help sort this out. Do you have a few minutes?"

If not a good time: "Of course, I completely understand! When would be a better time for me to reach you? I just want to make sure we get this cleared up before any late fees or credit impact."

== IDENTITY CHECK ==
1. Only discuss account details with {customer_name} directly.
2. The ONLY verification you may ask for is confirmation of the customer's name and the last 4 digits of the credit card account ending in {account_suffix}.
3. After the customer confirms their name and the last 4 digits, do NOT ask for any additional verification such as date of birth, address, OTP, full card number, email, PIN, password, or any other identity question.
4. Use a single short verification line such as: "Before I continue, can you please confirm I'm speaking with {customer_name} and that your card ends in {account_suffix}?"
5. If wrong person or they cannot confirm: "Oh, I'm sorry to bother you! I'll update our records. Have a lovely {time_of_day}!" → call end_call(outcome="wrong_person").
6. If voicemail: "Hi, this is a message for {customer_name} from {bank_name} Customer Care. Please give us a call back at your earliest convenience regarding your credit card account. Thank you and have a great day!" → call end_call(outcome="voicemail").

== PAYMENT OPTIONS — WORK THROUGH IN ORDER ==

--- OPTION A: FULL PAYMENT ---
"Would you be able to clear the full balance of ${debt_amount:.2f} today? You can pay through our app, internet banking, or I can guide you through a phone payment — whichever is easiest for you."

If yes → call record_payment_commitment(amount, today's date, payment method). Proceed to closing.
If no → move to Option B.

--- OPTION B: PARTIAL PAYMENT ---
"No worries at all! How about paying ${partial_amount:.2f} today — that's half the balance — and we set the remaining amount due in 30 days? That way you avoid any further late charges straight away."

If yes → call record_payment_commitment(amount=${partial_amount:.2f}, today, method). Proceed to closing.
If no → move to Option C.

--- OPTION C: 30-DAY EXTENSION ---
"That's completely fine. I can extend your due date by 30 days — so you'd have until {extension_date} to pay the full ${debt_amount:.2f} with no additional penalties in between. Would that give you enough time?"

If yes → call record_payment_commitment(amount=${debt_amount:.2f}, {extension_date}, unspecified). Proceed to closing.
If no → move to Option D.

--- OPTION D: INSTALLMENT PLAN ---
"I understand. We do offer flexible payment plans — ${installment_3m:.2f} per month for 3 months, or ${installment_6m:.2f} per month over 6 months. Both are interest-free arrangements. Which would suit your budget better?"

If 3-month → call record_payment_plan(plan_type="3_month", monthly_amount=${installment_3m:.2f}, start_date=next month).
If 6-month → call record_payment_plan(plan_type="6_month", monthly_amount=${installment_6m:.2f}, start_date=next month).
If neither → move to Option E.

--- OPTION E: 12-MONTH PLAN ---
"I hear you, and I really want to help. Our most flexible option is a 12-month arrangement at just ${installment_12m:.2f} per month — that's the lowest we can offer. This would also help protect your credit score from further impact. Would that work?"

If yes → call record_payment_plan(plan_type="12_month_hardship", monthly_amount=${installment_12m:.2f}, start_date=next month).
If no → move to Option F.

--- OPTION F: FINAL ATTEMPT ---
"I've shared every option available to me today, and I genuinely don't want to see this go further — an unresolved balance can affect your credit rating and may eventually be referred to a recovery agency, and I'd much rather we avoid that. Is there any amount at all you could manage today, even just a small token payment to keep the account active?"

If still no → "I understand, {customer_name}. I'll note our conversation. Please do give us a call when things improve — we're always here to help." → call end_call(outcome="refused").

== SPECIAL SITUATIONS ==

--- ALREADY PAID ---
"Oh, I'm so sorry for the confusion, {customer_name}! Can you tell me roughly when you made that payment and how — through the app, internet banking, or at a branch? I'll flag this for our reconciliation team right away and make sure you don't receive any more reminders."
→ call record_already_paid with their details, then end_call(outcome="already_paid").

--- DISPUTE ---
"I completely understand, and I take that very seriously. I'm going to flag this as a disputed transaction on your account right now — that means all reminder activity is paused while our disputes team looks into it. Could I get your current mailing address so we can send you a written breakdown within 5 business days?"
→ call record_dispute, then end_call(outcome="dispute").

--- OPT-OUT ---
"Absolutely, I respect that completely, {customer_name}. I'm noting your preference right now — you won't receive any more calls from us. Please keep in mind you can always manage your account through the {bank_name} app or by calling our main customer care line."
→ call record_opt_out immediately, then end_call(outcome="opt_out").

--- ANGRY / FRUSTRATED CUSTOMER ---
Do NOT push payments. Say: "I'm really sorry this has caused you stress, {customer_name} — that's genuinely the last thing we want. I'm here to listen. Can you tell me what's going on so I can see what we can do to help?"
De-escalate fully before any mention of payment. If the customer remains hostile after two attempts: "I understand. Let's leave it here for today. Please call our main line when you're ready and we'll do everything we can to help." → call end_call(outcome="refused").

== CLOSING SCRIPT ==
After any payment commitment or plan is recorded:
"That's wonderful, {customer_name}! Just to confirm: [clearly restate the arrangement — amount, date, method]. You'll receive an SMS and email confirmation shortly. Is there anything else I can help you with today?"

Then: "Thank you so much for your time! Have a lovely {time_of_day}. Goodbye!"
→ call end_call(outcome="closing").

== TOOL USAGE ==
- record_payment_commitment → customer agrees to pay a lump sum (Options A, B, or C)
- record_payment_plan → customer agrees to monthly instalments (Options D or E)
- record_dispute → customer disputes the charge or amount
- record_already_paid → customer says they've already paid
- record_opt_out → customer asks to stop all calls (do this IMMEDIATELY)
- end_call → at the natural end of the call for any reason

== ABSOLUTE RULES ==
1. NEVER threaten legal action, court, bailiffs, or credit blacklisting in a threatening way.
2. NEVER identify yourself as AI or a bot. If asked, say "I'm Sarah from {bank_name} Customer Care."
3. NEVER discuss any account other than this specific credit card.
4. NEVER share full account numbers, PINs, passwords, or sensitive personal data.
4a. NEVER ask for full card number, CVV, OTP, date of birth, mailing address for verification, or any verification beyond name and last 4 digits unless handling a dispute and the customer voluntarily provides an address.
5. NEVER argue — always acknowledge, empathize, and redirect.
6. ALWAYS honor opt-out requests instantly.
7. ALWAYS keep responses brief and conversational — this is a phone call.
8. Frame everything as helping the customer, not pursuing them.
9. NEVER speak about anything unrelated to the overdue credit card bill.
10. NEVER produce random text, small talk, or generic assistant behavior.
11. If uncertain, restate the overdue amount, days overdue, and available payment options, then ask a direct next-step question.
12. NEVER output non-speech artifacts, sound descriptions, or theatrical cues.
"""

    def __init__(self, customer_data: dict, conversation_manager, negotiation_engine):
        self.customer_data = customer_data
        self.conversation_manager = conversation_manager
        self.negotiation_engine = negotiation_engine
        self._state = {
            "dispute_raised": False,
            "already_paid_claimed": False,
            "opt_out_requested": False,
            "call_outcome": "in_progress",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # SYSTEM PROMPT BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def build_system_prompt(self) -> str:
        """Format the system prompt template with actual customer/amount data."""
        amounts = self.negotiation_engine.amounts
        debt_amount = float(self.customer_data.get("debt_amount", 0))
        account_number = str(self.customer_data.get("account_number", "0000"))
        account_suffix = account_number[-4:] if len(account_number) >= 4 else account_number

        extension_date = (datetime.now() + timedelta(days=30)).strftime("%B %d, %Y")

        # Determine time of day for greeting
        hour = datetime.now().hour
        if hour < 12:
            time_of_day = "morning"
        elif hour < 17:
            time_of_day = "afternoon"
        else:
            time_of_day = "evening"

        import os
        bank_name = os.getenv("BANK_NAME", "National City Bank")

        return self.SYSTEM_PROMPT.format(
            bank_name=bank_name,
            customer_name=self.customer_data.get("name", "Valued Customer"),
            account_suffix=account_suffix,
            debt_amount=debt_amount,
            days_overdue=self.customer_data.get("days_overdue", 0),
            partial_amount=amounts.get("partial_50", round(debt_amount * 0.50, 2)),
            extension_date=extension_date,
            installment_3m=amounts.get("installment_3m", round(debt_amount / 3, 2)),
            installment_6m=amounts.get("installment_6m", round(debt_amount / 6, 2)),
            installment_12m=amounts.get("installment_12m", round(debt_amount / 12, 2)),
            time_of_day=time_of_day,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TOOLS SCHEMA
    # ─────────────────────────────────────────────────────────────────────────

    def get_tools(self) -> ToolsSchema:
        """Return a ToolsSchema containing all 6 debt collection tools."""

        record_payment_commitment = FunctionSchema(
            name="record_payment_commitment",
            description=(
                "Record the customer's commitment to pay a specific amount on a specific date. "
                "Call this as soon as the customer agrees to pay (Option A, B, or C)."
            ),
            properties={
                "amount": {
                    "type": "number",
                    "description": "The dollar amount the customer agreed to pay.",
                },
                "payment_date": {
                    "type": "string",
                    "description": "The date the customer will make the payment (ISO 8601 or natural language).",
                },
                "payment_method": {
                    "type": "string",
                    "description": "Payment method mentioned by the customer (e.g., 'credit card', 'bank transfer', 'check', 'unspecified').",
                },
            },
            required=["amount", "payment_date", "payment_method"],
        )

        record_payment_plan = FunctionSchema(
            name="record_payment_plan",
            description=(
                "Record an agreed-upon multi-month installment plan (Option D or E). "
                "Call this when the customer agrees to a structured monthly payment arrangement."
            ),
            properties={
                "plan_type": {
                    "type": "string",
                    "enum": ["3_month", "6_month", "12_month_hardship"],
                    "description": "The installment plan duration type.",
                },
                "monthly_amount": {
                    "type": "number",
                    "description": "The monthly payment amount agreed upon.",
                },
                "start_date": {
                    "type": "string",
                    "description": "The date the first installment payment is due.",
                },
            },
            required=["plan_type", "monthly_amount", "start_date"],
        )

        record_dispute = FunctionSchema(
            name="record_dispute",
            description=(
                "Record a formal debt dispute raised by the customer. "
                "Call this immediately when the customer disputes the debt's validity, amount, or ownership."
            ),
            properties={
                "reason": {
                    "type": "string",
                    "description": "The customer's stated reason for disputing the debt.",
                },
                "mailing_address": {
                    "type": "string",
                    "description": "Mailing address for validation notice (optional — only if customer provides it).",
                },
            },
            required=["reason"],
        )

        record_already_paid = FunctionSchema(
            name="record_already_paid",
            description=(
                "Record the customer's claim that this debt has already been paid. "
                "Call this when the customer states they have already made a payment."
            ),
            properties={
                "payment_date": {
                    "type": "string",
                    "description": "The approximate date the customer claims they paid (optional).",
                },
                "payment_method": {
                    "type": "string",
                    "description": "How the customer claims they paid (e.g., 'check', 'online transfer') (optional).",
                },
            },
            required=[],
        )

        record_opt_out = FunctionSchema(
            name="record_opt_out",
            description=(
                "Record the customer's request to be removed from the call list (Do Not Call). "
                "Call this IMMEDIATELY when the customer requests opt-out, then end the call."
            ),
            properties={},
            required=[],
        )

        end_call = FunctionSchema(
            name="end_call",
            description=(
                "End the current call and record the final outcome. "
                "Call this at the natural conclusion of any call."
            ),
            properties={
                "outcome": {
                    "type": "string",
                    "enum": [
                        "closing",
                        "refused",
                        "wrong_person",
                        "voicemail",
                        "dispute",
                        "already_paid",
                        "opt_out",
                    ],
                    "description": "The final outcome category of this call.",
                },
                "notes": {
                    "type": "string",
                    "description": "Optional free-text notes about how the call concluded.",
                },
            },
            required=["outcome"],
        )

        return ToolsSchema(
            standard_tools=[
                record_payment_commitment,
                record_payment_plan,
                record_dispute,
                record_already_paid,
                record_opt_out,
                end_call,
            ]
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TOOL HANDLER REGISTRATION
    # ─────────────────────────────────────────────────────────────────────────

    def register_with_llm(self, llm, context, task, twilio_manager=None, call_sid=None):
        """
        Register all tool handlers with the LLM service and set tools on context.
        Must be called after the pipeline task is created.
        """
        context.set_tools(self.get_tools())

        cm = self.conversation_manager
        state = self._state

        # ── record_payment_commitment ─────────────────────────────────────
        async def handle_record_payment_commitment(params):
            try:
                args = params.arguments
                amount = args.get("amount", 0)
                payment_date = args.get("payment_date", "unspecified")
                payment_method = args.get("payment_method", "unspecified")

                cm.set_payment_details(
                    plan_type="full_payment" if amount >= float(self.customer_data.get("debt_amount", 0)) * 0.9 else "partial_payment",
                    amount=amount,
                    promise_date=payment_date,
                )
                state["call_outcome"] = "promise_to_pay"
                logger.info(
                    f"[FlowManager] Payment commitment recorded: ${amount} on {payment_date} via {payment_method}"
                )
                await params.result_callback({
                    "status": "recorded",
                    "message": f"Payment commitment of ${amount:.2f} on {payment_date} via {payment_method} has been recorded successfully.",
                })
            except Exception as e:
                logger.error(f"[FlowManager] record_payment_commitment error: {e}", exc_info=True)
                await params.result_callback({"status": "error", "message": str(e)})

        # ── record_payment_plan ───────────────────────────────────────────
        async def handle_record_payment_plan(params):
            try:
                args = params.arguments
                plan_type = args.get("plan_type", "unknown")
                monthly_amount = args.get("monthly_amount", 0)
                start_date = args.get("start_date", "unspecified")

                cm.set_payment_details(
                    plan_type=plan_type,
                    amount=monthly_amount,
                    promise_date=start_date,
                )
                state["call_outcome"] = "promise_to_pay"
                logger.info(
                    f"[FlowManager] Payment plan recorded: {plan_type} @ ${monthly_amount}/mo starting {start_date}"
                )
                await params.result_callback({
                    "status": "recorded",
                    "message": f"Payment plan '{plan_type}' recorded: ${monthly_amount:.2f}/month starting {start_date}.",
                })
            except Exception as e:
                logger.error(f"[FlowManager] record_payment_plan error: {e}", exc_info=True)
                await params.result_callback({"status": "error", "message": str(e)})

        # ── record_dispute ────────────────────────────────────────────────
        async def handle_record_dispute(params):
            try:
                args = params.arguments
                reason = args.get("reason", "unspecified")
                mailing_address = args.get("mailing_address", None)

                state["dispute_raised"] = True
                cm.state["dispute_raised"] = True
                cm._set_outcome("dispute")
                state["call_outcome"] = "dispute"
                logger.info(f"[FlowManager] Dispute recorded: {reason} | address: {mailing_address}")
                await params.result_callback({
                    "status": "recorded",
                    "message": (
                        f"Dispute has been formally logged. Reason: {reason}. "
                        + (f"Validation notice will be mailed to: {mailing_address}." if mailing_address else "No mailing address provided.")
                        + " All collection activity is now paused pending review."
                    ),
                })
            except Exception as e:
                logger.error(f"[FlowManager] record_dispute error: {e}", exc_info=True)
                await params.result_callback({"status": "error", "message": str(e)})

        # ── record_already_paid ───────────────────────────────────────────
        async def handle_record_already_paid(params):
            try:
                args = params.arguments
                payment_date = args.get("payment_date", None)
                payment_method = args.get("payment_method", None)

                state["already_paid_claimed"] = True
                cm.state["already_paid_claimed"] = True
                cm._set_outcome("already_paid")
                state["call_outcome"] = "already_paid"
                logger.info(
                    f"[FlowManager] Already-paid claim recorded: date={payment_date}, method={payment_method}"
                )
                await params.result_callback({
                    "status": "recorded",
                    "message": (
                        "Customer's prior payment claim has been logged. "
                        + (f"Claimed payment date: {payment_date}. " if payment_date else "")
                        + (f"Payment method: {payment_method}. " if payment_method else "")
                        + "Reconciliation team will verify within 3–5 business days."
                    ),
                })
            except Exception as e:
                logger.error(f"[FlowManager] record_already_paid error: {e}", exc_info=True)
                await params.result_callback({"status": "error", "message": str(e)})

        # ── record_opt_out ────────────────────────────────────────────────
        async def handle_record_opt_out(params):
            try:
                state["opt_out_requested"] = True
                cm.state["opt_out_requested"] = True
                logger.warning("[FlowManager] Opt-out recorded — customer will be removed from call list.")
                await params.result_callback({
                    "status": "recorded",
                    "message": "Customer opt-out has been logged. This number will be added to the Do Not Call list immediately.",
                })
            except Exception as e:
                logger.error(f"[FlowManager] record_opt_out error: {e}", exc_info=True)
                await params.result_callback({"status": "error", "message": str(e)})

        # ── end_call ──────────────────────────────────────────────────────
        async def handle_end_call(params):
            try:
                args = params.arguments
                outcome = args.get("outcome", "closing")
                notes = args.get("notes", "")

                cm._set_outcome(outcome)
                state["call_outcome"] = outcome
                logger.info(f"[FlowManager] end_call triggered: outcome={outcome} notes={notes!r}")

                await params.result_callback({
                    "status": "ending",
                    "message": f"Call is concluding with outcome: {outcome}. {notes}".strip(),
                })

                # Give the TTS 20 seconds to finish speaking the goodbye, then cancel
                async def _delayed_cancel():
                    await asyncio.sleep(20)
                    logger.info("[FlowManager] Cancelling pipeline task after goodbye delay.")
                    await task.cancel()

                asyncio.create_task(_delayed_cancel())

            except Exception as e:
                logger.error(f"[FlowManager] end_call error: {e}", exc_info=True)
                await params.result_callback({"status": "error", "message": str(e)})

        # Register all handlers
        llm.register_function("record_payment_commitment", handle_record_payment_commitment)
        llm.register_function("record_payment_plan", handle_record_payment_plan)
        llm.register_function("record_dispute", handle_record_dispute)
        llm.register_function("record_already_paid", handle_record_already_paid)
        llm.register_function("record_opt_out", handle_record_opt_out)
        llm.register_function("end_call", handle_end_call)

        logger.info("[FlowManager] All 6 tool handlers registered with LLM.")
