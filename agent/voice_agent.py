"""
voice_agent.py
--------------
Pipecat voice agent for debt collection calls.
Uses Claude claude-sonnet-4-6 (Anthropic), Deepgram STT, and Cartesia TTS.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.anthropic.llm import AnthropicLLMService, AnthropicLLMContext, AnthropicLLMSettings
from pipecat.transports.websocket.fastapi import FastAPIWebsocketTransport, FastAPIWebsocketParams
from pipecat.serializers.twilio import TwilioFrameSerializer
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.frames.frames import Frame, LLMRunFrame, TranscriptionFrame

from agent.conversation_manager import ConversationManager
from agent.negotiation_engine import NegotiationEngine
from agent.flow_manager import FlowManager

logger = logging.getLogger(__name__)


class SentimentMonitorProcessor(FrameProcessor):
    """
    Intercepts transcription frames, runs sentiment analysis,
    updates the conversation manager, and optionally broadcasts
    transcript/sentiment updates via the on_transcript_update callback.
    """

    def __init__(self, conversation_manager: ConversationManager, sentiment_analyzer, on_transcript_update=None):
        super().__init__()
        self.conversation_manager = conversation_manager
        self.sentiment_analyzer = sentiment_analyzer
        self.on_transcript_update = on_transcript_update

    async def process_frame(self, frame: Frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame) and frame.text:
            sentiment = self.sentiment_analyzer.analyze(frame.text)
            self.conversation_manager.update_sentiment(sentiment)
            self.conversation_manager.add_transcript_turn("customer", frame.text, sentiment)
            logger.info(f"[CUSTOMER] {frame.text} | sentiment={sentiment.get('label')}")

            if self.on_transcript_update:
                try:
                    import asyncio
                    asyncio.create_task(
                        self.on_transcript_update({
                            "type": "transcript_update",
                            "role": "customer",
                            "text": frame.text,
                            "sentiment": sentiment,
                        })
                    )
                except Exception as exc:
                    logger.debug(f"[SentimentMonitor] on_transcript_update error: {exc}")

        await self.push_frame(frame, direction)


class DebtCollectionVoiceAgent:
    """
    Orchestrates the full Pipecat pipeline for one outbound debt-collection call.
    """

    def __init__(
        self,
        customer_data: dict,
        call_sid: str,
        stream_sid: str,
        sentiment_analyzer=None,
        risk_predictor=None,
        twilio_manager=None,
        on_update: Optional[callable] = None,
    ):
        self.customer_data = customer_data
        self.call_sid = call_sid
        self.stream_sid = stream_sid
        self.twilio_manager = twilio_manager
        self.on_update = on_update
        self.session_id = f"{call_sid}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Core components
        self.conversation_manager = ConversationManager(customer_data)
        self.negotiation_engine = NegotiationEngine(customer_data)
        self.flow_manager = FlowManager(customer_data, self.conversation_manager, self.negotiation_engine)

        # ML components (lazy-load if not provided)
        if sentiment_analyzer is None:
            from ml.sentiment_analysis import SentimentAnalyzer
            sentiment_analyzer = SentimentAnalyzer()
        if risk_predictor is None:
            from ml.risk_model import RiskPredictor
            risk_predictor = RiskPredictor()

        self.sentiment_analyzer = sentiment_analyzer
        self.risk_predictor = risk_predictor

        # Ensure output directories exist
        os.makedirs("transcripts", exist_ok=True)
        os.makedirs("recordings", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        logger.info(
            f"[{self.session_id}] DebtCollectionVoiceAgent initialized for "
            f"{customer_data.get('name', 'Unknown')} | "
            f"debt=${customer_data.get('debt_amount', 0):.2f}"
        )

    async def run(self, websocket):
        """Build and run the full Pipecat pipeline for this call."""
        logger.info(f"[{self.session_id}] Starting Pipecat pipeline...")

        # ── Transport ────────────────────────────────────────────────────
        serializer = TwilioFrameSerializer(
            stream_sid=self.stream_sid,
            call_sid=self.call_sid,
            account_sid=os.getenv("TWILIO_ACCOUNT_SID"),
            auth_token=os.getenv("TWILIO_AUTH_TOKEN"),
        )
        transport = FastAPIWebsocketTransport(
            websocket=websocket,
            params=FastAPIWebsocketParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer(),
                serializer=serializer,
            ),
        )

        # ── STT ─────────────────────────────────────────────────────────
        stt = DeepgramSTTService(
            api_key=os.getenv("DEEPGRAM_API_KEY"),
            live_options=LiveOptions(
                model="nova-2-phonecall",
                language="en-US",
                punctuate=True,
                endpointing=300,
            ),
        )

        # ── LLM ─────────────────────────────────────────────────────────
        llm = AnthropicLLMService(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            settings=AnthropicLLMSettings(
                model="claude-sonnet-4-6",
                max_tokens=1024,
            ),
        )

        # ── TTS ─────────────────────────────────────────────────────────
        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id=os.getenv("CARTESIA_VOICE_ID", "a0e99841-438c-4a64-b679-ae501e7d6091"),
        )

        # ── Sentiment Monitor ────────────────────────────────────────────
        # Pass on_update from conversation_manager if set
        on_transcript_update = getattr(self.conversation_manager, "on_update", None) or self.on_update
        sentiment_monitor = SentimentMonitorProcessor(
            conversation_manager=self.conversation_manager,
            sentiment_analyzer=self.sentiment_analyzer,
            on_transcript_update=on_transcript_update,
        )

        # ── LLM Context ─────────────────────────────────────────────────
        system_prompt = self.flow_manager.build_system_prompt()
        context = AnthropicLLMContext(
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The phone call has just been answered. Begin the call now. "
                        "Stay strictly in your role as a bank representative calling about an overdue credit card bill. "
                        "Your first spoken line must follow the opening script exactly."
                    ),
                }
            ],
            system=system_prompt,
        )
        context_aggregator = llm.create_context_aggregator(context)

        # ── Pipeline ─────────────────────────────────────────────────────
        pipeline = Pipeline([
            transport.input(),
            stt,
            sentiment_monitor,
            context_aggregator.user(),
            llm,
            tts,
            context_aggregator.assistant(),
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(allow_interruptions=True),
        )

        # ── Register tool handlers ────────────────────────────────────────
        self.flow_manager.register_with_llm(
            llm=llm,
            context=context,
            task=task,
            twilio_manager=self.twilio_manager,
            call_sid=self.call_sid,
        )

        # Kick off the opening line immediately after the call connects.
        await task.queue_frame(LLMRunFrame())

        # ── Transport event: disconnected ─────────────────────────────────
        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(t, client):
            logger.info(f"[{self.session_id}] Client disconnected — cancelling task.")
            await task.cancel()

        # ── Run ───────────────────────────────────────────────────────────
        runner = PipelineRunner(handle_sigint=False)
        await runner.run(task)
        await self._finalize_call()

    async def _finalize_call(self):
        """Save transcript, generate summary, and log final outcome."""
        transcript = self.conversation_manager.get_full_transcript()
        state = self.conversation_manager.get_state()

        # Persist full transcript
        transcript_path = f"transcripts/{self.session_id}.json"
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "session_id": self.session_id,
                    "customer": self.customer_data,
                    "transcript": transcript,
                    "state": state,
                    "timestamp": datetime.now().isoformat(),
                },
                f,
                indent=2,
            )

        # Generate and persist summary
        risk_score = self.risk_predictor.predict(state, transcript)
        summary = self.conversation_manager.generate_summary(risk_score)
        summary["session_id"] = self.session_id

        summary_path = f"transcripts/{self.session_id}_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            f"[{self.session_id}] Call finalized — "
            f"outcome={summary['call_outcome']} | "
            f"risk={summary['risk_label']} | "
            f"duration={summary['call_duration_seconds']}s"
        )

        # Broadcast final update if callback set
        if self.on_update:
            try:
                import asyncio
                asyncio.create_task(
                    self.on_update({
                        "type": "call_finalized",
                        "session_id": self.session_id,
                        "summary": summary,
                    })
                )
            except Exception as exc:
                logger.debug(f"[{self.session_id}] on_update (finalize) error: {exc}")

        return summary
