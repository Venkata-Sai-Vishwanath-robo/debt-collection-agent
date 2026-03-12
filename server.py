"""
server.py
---------
FastAPI server that:
  - Accepts POST /call/initiate  → places outbound Twilio call
  - Serves POST /twiml           → returns TwiML connecting call to WebSocket
  - Handles ws://.../ws/{sid}    → runs Pipecat voice agent
  - Accepts status webhooks from Twilio
  - Serves the operator portal at GET /portal
  - Streams live events via SSE at GET /api/stream
  - Exposes JSON APIs: /api/sessions, /api/sessions/{id}, /api/stats
"""

import asyncio
import csv
import io
import json
import logging
import os
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional

import openpyxl

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, UploadFile, File, Form
from fastapi.responses import PlainTextResponse, JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.voice_agent import DebtCollectionVoiceAgent
from telephony.twilio_call import TwilioCallManager

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────
active_sessions: dict = {}
twilio_manager: Optional[TwilioCallManager] = None

# SSE subscriber queues
sse_queues: list = []

# ── Bulk Call Queue ────────────────────────────────────────────────────────────
call_queue: list = []          # list of queue item dicts
queue_running: bool = False    # True while background processor is active
queue_task: Optional[asyncio.Task] = None

QUEUE_DELAY_SECONDS = int(os.getenv("QUEUE_DELAY_SECONDS", "10"))
MAX_CONCURRENT_CALLS = int(os.getenv("MAX_CONCURRENT_CALLS", "3"))

# Global pre-loaded instances
sentiment_analyzer_instance = None
risk_predictor_instance = None


# ── SSE Broadcast Helper ──────────────────────────────────────────────────────

async def broadcast_event(event: dict):
    """Put an event dict into every active SSE subscriber queue."""
    dead = []
    for q in sse_queues:
        try:
            await q.put(event)
        except Exception:
            dead.append(q)
    for q in dead:
        try:
            sse_queues.remove(q)
        except ValueError:
            pass


# ── Bulk Queue Helpers ────────────────────────────────────────────────────────

_PHONE_COLS   = {"phone", "phone_number", "number", "to_number", "mobile", "contact", "telephone"}
_NAME_COLS    = {"name", "customer_name", "full_name", "debtor_name", "client_name"}
_AMOUNT_COLS  = {"amount", "debt_amount", "balance", "outstanding", "overdue_amount", "due_amount"}
_ACCOUNT_COLS = {"account", "account_number", "account_no", "acc_no", "account_id"}
_DAYS_COLS    = {"days", "days_overdue", "overdue_days", "days_past_due", "days_due"}


def _pick(row: dict, candidates: set):
    for key in candidates:
        val = row.get(key)
        if val not in (None, ""):
            return str(val).strip()
    return None


def _normalize_phone(phone: str) -> str:
    cleaned = phone.replace(" ", "").replace("-", "").replace("(", "").replace(")", "").replace(".", "")
    if not cleaned.startswith("+"):
        cleaned = "+" + cleaned
    return cleaned


def parse_upload(content: bytes, filename: str) -> list:
    """
    Parse an Excel (.xlsx / .xls) or CSV file.
    Returns a list of dicts ready for OutboundCallRequest.
    Required columns: phone, name, amount.
    Optional: account_number, days_overdue.
    """
    raw_rows = []

    if filename.lower().endswith(".csv"):
        text = content.decode("utf-8-sig", errors="replace")
        reader = csv.DictReader(io.StringIO(text))
        for row in reader:
            raw_rows.append({k.lower().strip(): v for k, v in row.items() if k})
    else:
        wb = openpyxl.load_workbook(filename=io.BytesIO(content), read_only=True, data_only=True)
        ws = wb.active
        headers = None
        for i, row in enumerate(ws.iter_rows(values_only=True)):
            if i == 0:
                headers = [str(c).lower().strip() if c is not None else f"col{i}" for i, c in enumerate(row)]
            else:
                if all(c is None for c in row):
                    continue
                raw_rows.append(dict(zip(headers, [str(c).strip() if c is not None else "" for c in row])))
        wb.close()

    result = []
    for idx, row in enumerate(raw_rows):
        phone   = _pick(row, _PHONE_COLS)
        name    = _pick(row, _NAME_COLS)
        amount  = _pick(row, _AMOUNT_COLS)
        account = _pick(row, _ACCOUNT_COLS) or f"ACC-{idx+1:04d}"
        days    = _pick(row, _DAYS_COLS)

        if not phone or not name or not amount:
            continue  # skip rows missing required fields
        try:
            amount_f = float(str(amount).replace(",", "").replace("$", "").replace("₹", ""))
        except ValueError:
            continue

        try:
            days_i = int(float(days)) if days else 30
        except ValueError:
            days_i = 30

        result.append({
            "to_number":     _normalize_phone(phone),
            "customer_name": name,
            "debt_amount":   round(amount_f, 2),
            "account_number": account,
            "days_overdue":  days_i,
        })

    return result


def _queue_summary() -> dict:
    counts = {"pending": 0, "calling": 0, "initiated": 0, "failed": 0, "skipped": 0}
    for item in call_queue:
        counts[item.get("status", "pending")] = counts.get(item.get("status", "pending"), 0) + 1
    return {
        "total": len(call_queue),
        "running": queue_running,
        "counts": counts,
        "items": [
            {
                "id":            item["id"],
                "to_number":     item["to_number"],
                "customer_name": item["customer_name"],
                "debt_amount":   item["debt_amount"],
                "status":        item.get("status", "pending"),
                "session_id":    item.get("session_id"),
                "error":         item.get("error"),
            }
            for item in call_queue
        ],
    }


async def _process_queue():
    global queue_running
    queue_running = True
    logger.info(f"[Queue] Started. {sum(1 for i in call_queue if i['status']=='pending')} pending items.")

    try:
        for item in call_queue:
            if not queue_running:
                break
            if item["status"] != "pending":
                continue

            # Respect max concurrent limit
            while True:
                active = sum(1 for i in call_queue if i["status"] in ("calling", "initiated"))
                if active < MAX_CONCURRENT_CALLS:
                    break
                await asyncio.sleep(2)

            item["status"] = "calling"
            await broadcast_event({"type": "queue_update", **_queue_summary()})

            try:
                req = OutboundCallRequest(
                    to_number=item["to_number"],
                    customer_name=item["customer_name"],
                    debt_amount=item["debt_amount"],
                    account_number=item["account_number"],
                    days_overdue=item["days_overdue"],
                )
                result = await initiate_call(req)
                item["session_id"] = result["session_id"]
                item["status"] = "initiated"
                logger.info(f"[Queue] Initiated call for {item['customer_name']} → {item['to_number']}")
            except Exception as e:
                item["status"] = "failed"
                item["error"] = str(e)
                logger.error(f"[Queue] Failed call for {item['customer_name']}: {e}")

            await broadcast_event({"type": "queue_update", **_queue_summary()})
            await asyncio.sleep(QUEUE_DELAY_SECONDS)

    finally:
        queue_running = False
        logger.info("[Queue] Processing finished.")
        await broadcast_event({"type": "queue_update", **_queue_summary()})


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global twilio_manager, sentiment_analyzer_instance, risk_predictor_instance

    os.makedirs("recordings", exist_ok=True)
    os.makedirs("transcripts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    # Pre-load ML models at startup so calls don't lag
    logger.info("Pre-loading ML models...")
    from ml.sentiment_analysis import SentimentAnalyzer
    from ml.risk_model import RiskPredictor
    sentiment_analyzer_instance = SentimentAnalyzer()
    risk_predictor_instance = RiskPredictor()
    logger.info("ML models ready.")

    try:
        twilio_manager = TwilioCallManager()
        logger.info("Twilio manager ready.")
    except Exception as e:
        logger.error(f"Twilio init failed: {e}")

    yield
    logger.info("Server shutting down.")


app = FastAPI(
    title="Debt Collection Voice Agent",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (serves the portal dashboard)
app.mount("/static", StaticFiles(directory="static", html=True), name="static")


# ── Models ────────────────────────────────────────────────────────────────────

class OutboundCallRequest(BaseModel):
    to_number: str       # E.164 e.g. +15551234567
    customer_name: str
    debt_amount: float
    account_number: str
    days_overdue: int = 30


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": len(active_sessions),
        "twilio_ready": twilio_manager is not None,
    }


# ── Portal ────────────────────────────────────────────────────────────────────

@app.get("/portal")
async def portal():
    """Serve the operator dashboard SPA."""
    index_path = os.path.join("static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="Portal not found. Ensure static/index.html exists.")
    return FileResponse(index_path)


# ── SSE — Live Event Stream ───────────────────────────────────────────────────

@app.get("/api/stream")
async def sse_stream(request: Request):
    """
    Server-Sent Events endpoint.
    Clients connect here to receive live updates about sessions,
    sentiment, transcripts, and call completions.
    """
    queue: asyncio.Queue = asyncio.Queue()
    sse_queues.append(queue)

    async def event_generator():
        # Send an initial snapshot of all current sessions
        try:
            snapshot = _build_sessions_list()
            yield f"event: snapshot\ndata: {json.dumps({'sessions': snapshot})}\n\n"
        except Exception as e:
            logger.warning(f"[SSE] Snapshot error: {e}")

        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    event_type = event.get("type", "update")
                    yield f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    # Send a keepalive ping
                    yield "event: ping\ndata: {}\n\n"
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(f"[SSE] Generator error: {e}")
        finally:
            try:
                sse_queues.remove(queue)
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ── API: Sessions List ────────────────────────────────────────────────────────

def _build_sessions_list():
    """Build a serializable list of all sessions for the API."""
    result = []
    for session_id, session in active_sessions.items():
        customer = session.get("customer_data", {})
        agent = session.get("agent")

        # Try to get summary data if available
        summary = None
        summary_path = f"transcripts/{session_id}_summary.json"
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
            except Exception:
                pass

        sentiment = "neutral"
        risk_score = None
        call_outcome = session.get("call_outcome", "in_progress")
        created_at = session.get("created_at", "")

        if agent:
            state = agent.conversation_manager.get_state()
            sentiment = state.get("dominant_sentiment", "neutral")
            call_outcome = state.get("call_outcome", call_outcome)

        if summary:
            sentiment = summary.get("sentiment", sentiment)
            risk_score = summary.get("risk_score")
            call_outcome = summary.get("call_outcome", call_outcome)

        result.append({
            "session_id": session_id,
            "status": session.get("status", "unknown"),
            "call_sid": session.get("call_sid"),
            "customer_name": customer.get("name", "Unknown"),
            "debt_amount": customer.get("debt_amount", 0),
            "call_outcome": call_outcome,
            "sentiment": sentiment,
            "created_at": created_at,
            "risk_score": risk_score,
        })
    return result


@app.get("/api/sessions")
async def api_sessions():
    """Return JSON list of all sessions."""
    return JSONResponse(_build_sessions_list())


@app.get("/api/sessions/{session_id}")
async def api_session_detail(session_id: str):
    """Return full session data including transcript turns, state, and summary."""
    session = active_sessions.get(session_id)
    if not session:
        # Try loading from disk if session is no longer in memory
        transcript_path = f"transcripts/{session_id}.json"
        summary_path = f"transcripts/{session_id}_summary.json"
        if not os.path.exists(transcript_path):
            raise HTTPException(status_code=404, detail="Session not found.")
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript_data = json.load(f)
        summary = None
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        return JSONResponse({
            "session_id": session_id,
            "status": "completed",
            "customer": transcript_data.get("customer"),
            "transcript": transcript_data.get("transcript", []),
            "state": transcript_data.get("state", {}),
            "summary": summary,
        })

    customer = session.get("customer_data", {})
    agent = session.get("agent")
    transcript = []
    state = {}
    summary = None

    if agent:
        transcript = agent.conversation_manager.get_full_transcript()
        state = agent.conversation_manager.get_state()

    summary_path = f"transcripts/{session_id}_summary.json"
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
        except Exception:
            pass

    return JSONResponse({
        "session_id": session_id,
        "status": session.get("status", "unknown"),
        "call_sid": session.get("call_sid"),
        "customer": customer,
        "transcript": transcript,
        "state": state,
        "summary": summary,
        "created_at": session.get("created_at"),
    })


# ── API: Stats ────────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def api_stats():
    """Return aggregate statistics across all sessions."""
    total_sessions = len(active_sessions)
    active_count = sum(
        1 for s in active_sessions.values()
        if s.get("status") in ("ringing", "answered", "in_progress")
    )
    completed_sessions = sum(
        1 for s in active_sessions.values()
        if s.get("status") == "completed"
    )

    outcomes: dict = {}
    promise_count = 0
    risk_scores = []

    for session_id, session in active_sessions.items():
        agent = session.get("agent")
        outcome = "in_progress"

        if agent:
            state = agent.conversation_manager.get_state()
            outcome = state.get("call_outcome", "in_progress")

        # Also check summary file for completed sessions
        summary_path = f"transcripts/{session_id}_summary.json"
        if os.path.exists(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = json.load(f)
                outcome = summary.get("call_outcome", outcome)
                rs = summary.get("risk_score")
                if rs is not None:
                    risk_scores.append(float(rs))
            except Exception:
                pass

        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        if outcome == "promise_to_pay":
            promise_count += 1

    payment_rate = round(promise_count / completed_sessions, 4) if completed_sessions > 0 else 0.0
    avg_risk_score = round(sum(risk_scores) / len(risk_scores), 4) if risk_scores else None

    return JSONResponse({
        "total_sessions": total_sessions,
        "active_sessions": active_count,
        "completed_sessions": completed_sessions,
        "payment_rate": payment_rate,
        "avg_risk_score": avg_risk_score,
        "outcomes": outcomes,
    })


# ── Bulk Upload & Queue Endpoints ─────────────────────────────────────────────

@app.post("/api/bulk-upload")
async def bulk_upload(file: UploadFile = File(...)):
    """
    Parse an Excel (.xlsx) or CSV file and preview parsed rows.
    Does NOT start calling — just returns what was parsed.
    """
    content = await file.read()
    filename = file.filename or "upload.xlsx"

    try:
        rows = parse_upload(content, filename)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {e}")

    if not rows:
        raise HTTPException(
            status_code=422,
            detail="No valid rows found. Ensure columns: phone, name, amount (optional: account_number, days_overdue)."
        )

    return JSONResponse({"parsed": len(rows), "rows": rows})


class QueueAddRequest(BaseModel):
    rows: list
    delay_seconds: int = 10
    max_concurrent: int = 3


@app.post("/api/queue/add")
async def queue_add(req: QueueAddRequest):
    """Add a list of call records to the queue (call after /api/bulk-upload)."""
    global call_queue, QUEUE_DELAY_SECONDS, MAX_CONCURRENT_CALLS
    QUEUE_DELAY_SECONDS = max(5, req.delay_seconds)
    MAX_CONCURRENT_CALLS = max(1, req.max_concurrent)
    added = 0
    for row in req.rows:
        # rows may use "phone"/"name" (from bulk-upload) or "to_number"/"customer_name"
        to_number = row.get("to_number") or row.get("phone", "")
        customer_name = row.get("customer_name") or row.get("name", "Unknown")
        call_queue.append({
            "id":            uuid.uuid4().hex[:8],
            "to_number":     to_number,
            "customer_name": customer_name,
            "debt_amount":   float(row.get("debt_amount", 0)),
            "account_number": row.get("account_number", "ACC-0000"),
            "days_overdue":  int(row.get("days_overdue", 30)),
            "status":        "pending",
            "session_id":    None,
            "error":         None,
        })
        added += 1

    await broadcast_event({"type": "queue_update", **_queue_summary()})
    logger.info(f"[Queue] {added} items added. Total: {len(call_queue)}")
    return JSONResponse({"added": added, "total": len(call_queue), "queue": _queue_summary()})


@app.post("/api/queue/start")
async def queue_start():
    """Start processing the call queue."""
    global queue_running, queue_task
    if not twilio_manager:
        raise HTTPException(status_code=503, detail="Twilio not configured.")
    if queue_running:
        return JSONResponse({"status": "already_running"})
    pending = sum(1 for i in call_queue if i["status"] == "pending")
    if pending == 0:
        raise HTTPException(status_code=400, detail="No pending items in the queue.")
    queue_task = asyncio.create_task(_process_queue())
    return JSONResponse({"status": "started", "pending": pending, "queue": _queue_summary()})


@app.post("/api/queue/pause")
async def queue_pause():
    """Pause queue processing (current call in progress will still complete)."""
    global queue_running
    queue_running = False
    await broadcast_event({"type": "queue_update", **_queue_summary()})
    return JSONResponse({"status": "paused", "queue": _queue_summary()})


@app.post("/api/queue/clear")
async def queue_clear():
    """Remove all PENDING items from the queue."""
    global call_queue
    before = len(call_queue)
    call_queue = [i for i in call_queue if i["status"] != "pending"]
    removed = before - len(call_queue)
    await broadcast_event({"type": "queue_update", **_queue_summary()})
    return JSONResponse({"removed": removed, "remaining": len(call_queue), "queue": _queue_summary()})


@app.get("/api/queue")
async def get_queue():
    """Return current queue status and items."""
    return JSONResponse(_queue_summary())


# ── Initiate Call ─────────────────────────────────────────────────────────────

@app.post("/call/initiate")
async def initiate_call(req: OutboundCallRequest):
    """
    Place an outbound debt collection call.

    Example:
        curl -X POST http://localhost:8000/call/initiate \\
          -H "Content-Type: application/json" \\
          -d '{"to_number":"+15551234567","customer_name":"John Smith",
               "debt_amount":420.00,"account_number":"ACC-9823","days_overdue":45}'
    """
    if not twilio_manager:
        raise HTTPException(status_code=503, detail="Twilio not configured. Check .env and restart.")

    session_id = f"sess_{uuid.uuid4().hex[:12]}"
    customer_data = {
        "name": req.customer_name,
        "debt_amount": req.debt_amount,
        "account_number": req.account_number,
        "days_overdue": req.days_overdue,
    }

    active_sessions[session_id] = {
        "customer_data": customer_data,
        "call_sid": None,
        "status": "initiating",
        "created_at": datetime.now().isoformat(),
        "call_outcome": "in_progress",
    }

    try:
        call_sid = twilio_manager.make_outbound_call(
            to_number=req.to_number,
            session_id=session_id,
        )
        active_sessions[session_id]["call_sid"] = call_sid
        active_sessions[session_id]["status"] = "ringing"

        logger.info(f"[{session_id}] Call placed → SID: {call_sid}")

        # Broadcast session_created event
        await broadcast_event({
            "type": "session_created",
            "session_id": session_id,
            "call_sid": call_sid,
            "status": "ringing",
            "customer_name": req.customer_name,
            "debt_amount": req.debt_amount,
            "created_at": active_sessions[session_id]["created_at"],
        })

        return {
            "session_id": session_id,
            "call_sid": call_sid,
            "status": "ringing",
            "message": f"Calling {req.to_number}. Answer your phone!",
        }
    except Exception as e:
        active_sessions.pop(session_id, None)
        logger.error(f"Failed to place call: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Quick Test Call ───────────────────────────────────────────────────────────

@app.post("/test-call")
async def test_call():
    """
    Quick test using .env values. Run this to test without Postman.

    Usage:
        curl -X POST http://localhost:8000/test-call
    """
    test_number = os.getenv("TEST_PHONE_NUMBER")
    if not test_number:
        raise HTTPException(
            status_code=400,
            detail="TEST_PHONE_NUMBER not set in .env. Add it and restart."
        )

    req = OutboundCallRequest(
        to_number=test_number,
        customer_name=os.getenv("TEST_CUSTOMER_NAME", "John Smith"),
        debt_amount=float(os.getenv("TEST_DEBT_AMOUNT", "420.00")),
        account_number=os.getenv("TEST_ACCOUNT_NUMBER", "ACC-9823"),
        days_overdue=int(os.getenv("TEST_DAYS_OVERDUE", "45")),
    )
    return await initiate_call(req)


# ── TwiML Webhook ─────────────────────────────────────────────────────────────

@app.post("/twiml")
async def twiml_webhook(request: Request):
    """
    Twilio fetches this URL when the outbound call is answered.
    Returns TwiML that connects the call audio to our WebSocket.
    """
    params = dict(request.query_params)
    form_data = await request.form()

    session_id = params.get("session_id") or form_data.get("session_id", "unknown")
    call_sid = form_data.get("CallSid", "unknown")

    if session_id in active_sessions:
        active_sessions[session_id]["call_sid"] = call_sid
        active_sessions[session_id]["status"] = "answered"

    twiml = twilio_manager.generate_twiml(session_id)
    logger.info(f"[{session_id}] TwiML served for call {call_sid}")
    return PlainTextResponse(content=twiml, media_type="application/xml")


# ── WebSocket — Twilio Media Stream ───────────────────────────────────────────

@app.websocket("/ws/{session_id}")
async def media_stream(websocket: WebSocket, session_id: str):
    """
    Twilio connects its audio stream here after receiving our TwiML.
    Pipecat voice agent runs for the full duration of the call.
    """
    await websocket.accept()
    logger.info(f"[{session_id}] WebSocket connected.")

    session = active_sessions.get(session_id)
    if not session:
        logger.error(f"[{session_id}] Unknown session. Closing WebSocket.")
        await websocket.close(code=1008)
        return

    # Wait for the Twilio "start" message to get stream_sid
    stream_sid = None
    try:
        for _ in range(10):  # check up to 10 messages for the start event
            raw = await asyncio.wait_for(websocket.receive_text(), timeout=5.0)
            data = json.loads(raw)
            event = data.get("event", "")
            if event == "start":
                stream_sid = data.get("start", {}).get("streamSid") or data.get("streamSid")
                logger.info(f"[{session_id}] Stream SID: {stream_sid}")
                break
            elif event == "connected":
                logger.info(f"[{session_id}] Twilio connected event received.")
                continue  # keep reading, start event comes next
    except asyncio.TimeoutError:
        logger.warning(f"[{session_id}] Timed out waiting for start event.")
    except Exception as e:
        logger.warning(f"[{session_id}] Error reading start event: {e}")

    stream_sid = stream_sid or session_id
    call_sid = session.get("call_sid", session_id)
    customer_data = session["customer_data"]

    agent = DebtCollectionVoiceAgent(
        customer_data=customer_data,
        call_sid=call_sid,
        stream_sid=stream_sid,
        sentiment_analyzer=sentiment_analyzer_instance,
        risk_predictor=risk_predictor_instance,
        twilio_manager=twilio_manager,
    )
    session["agent"] = agent
    session["status"] = "in_progress"

    # Wire up SSE broadcast so live events flow to the portal
    session_id_for_closure = session_id

    async def _on_update(data):
        await broadcast_event({"session_id": session_id_for_closure, **data})

    agent.conversation_manager.on_update = _on_update

    try:
        await agent.run(websocket)
    except WebSocketDisconnect:
        logger.info(f"[{session_id}] WebSocket disconnected (call ended).")
        agent.conversation_manager.mark_call_dropped()
    except Exception as e:
        logger.error(f"[{session_id}] Agent error: {e}", exc_info=True)
    finally:
        session["status"] = "completed"
        logger.info(f"[{session_id}] Session complete.")

        # Broadcast session_complete event
        try:
            final_state = agent.conversation_manager.get_state()
            await broadcast_event({
                "type": "session_complete",
                "session_id": session_id,
                "call_outcome": final_state.get("call_outcome", "unknown"),
                "sentiment": final_state.get("dominant_sentiment", "neutral"),
                "duration": final_state.get("call_duration_seconds", 0),
                "customer_name": customer_data.get("name", "Unknown"),
                "debt_amount": customer_data.get("debt_amount", 0),
            })
        except Exception as broadcast_err:
            logger.warning(f"[{session_id}] session_complete broadcast error: {broadcast_err}")


# ── Twilio Status Webhooks ────────────────────────────────────────────────────

@app.post("/call-status")
async def call_status(request: Request):
    form = await request.form()
    params = dict(request.query_params)
    session_id = params.get("session_id", "unknown")
    status = form.get("CallStatus", "unknown")
    sid = form.get("CallSid", "unknown")
    duration = form.get("CallDuration", "0")

    logger.info(f"[{session_id}] Status: {status} | SID: {sid} | Duration: {duration}s")
    if session_id in active_sessions:
        active_sessions[session_id]["status"] = status
    return PlainTextResponse("OK")


@app.post("/recording-ready")
async def recording_ready(request: Request):
    form = await request.form()
    call_sid = form.get("CallSid", "")
    recording_sid = form.get("RecordingSid", "")
    duration = form.get("RecordingDuration", "0")

    logger.info(f"Recording ready: {recording_sid} | {duration}s | call: {call_sid}")

    # Download in background
    for session_id, session in active_sessions.items():
        if session.get("call_sid") == call_sid:
            output_path = f"recordings/{session_id}.mp3"
            asyncio.create_task(_download_recording(call_sid, output_path))
            break

    return PlainTextResponse("OK")


async def _download_recording(call_sid: str, output_path: str):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, twilio_manager.download_recording, call_sid, output_path)


# ── Session Info (legacy routes) ──────────────────────────────────────────────

@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    result = {
        "session_id": session_id,
        "status": session.get("status"),
        "call_sid": session.get("call_sid"),
        "customer": session.get("customer_data"),
        "created_at": session.get("created_at"),
    }
    agent = session.get("agent")
    if agent:
        result["conversation_state"] = agent.conversation_manager.get_state()
    return JSONResponse(result)


@app.get("/sessions")
async def list_sessions():
    return {
        sid: {
            "status": s.get("status"),
            "customer": s.get("customer_data", {}).get("name"),
            "created_at": s.get("created_at"),
        }
        for sid, s in active_sessions.items()
    }


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False,
        log_level="info",
    )
