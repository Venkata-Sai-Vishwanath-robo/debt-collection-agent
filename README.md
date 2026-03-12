# 🤖 Debt Collection Voice Agent

AI-powered outbound debt collection agent. Makes real phone calls, negotiates payment plans, 
analyzes customer sentiment, and generates call summaries with ML risk scores.

**Stack:** Pipecat · Twilio · Deepgram · Cartesia · Claude 3.5 Sonnet · scikit-learn · HuggingFace

---

## ⚡ Quick Start (3 steps)

```bash
# 1. Clone and setup everything
git clone <repo-url>
cd debt-collection-agent
bash setup.sh

# 2. Fill in .env (see section below)
nano .env

# 3. Start ngrok + server, then trigger a call
ngrok http 8000                         # Terminal 1
python server.py                        # Terminal 2
curl -X POST http://localhost:8000/test-call  # Terminal 3
```

Your phone rings. Answer it. The AI does the rest.

---

## 📋 What You Need (API Keys)

You need **5 things**. Here's exactly where to get each one:

| Key | Where to Get | Free Tier |
|-----|-------------|-----------|
| `TWILIO_ACCOUNT_SID` + `TWILIO_AUTH_TOKEN` | [console.twilio.com](https://console.twilio.com) → Account Info | $15 trial credit |
| `TWILIO_PHONE_NUMBER` | Twilio Console → Phone Numbers → Buy a Number | ~$1.15/month |
| `DEEPGRAM_API_KEY` | [console.deepgram.com](https://console.deepgram.com) → Create API Key | $200 free credit |
| `CARTESIA_API_KEY` | [play.cartesia.ai](https://play.cartesia.ai) → Settings → API Keys | Free trial |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys | Pay-as-you-go |

---

## 🔧 Step-by-Step Setup

### 1. Install Prerequisites

**macOS:**
```bash
brew install python@3.11 ngrok
```

**Ubuntu/Debian:**
```bash
sudo apt update && sudo apt install python3.11 python3.11-venv python3-pip -y
# ngrok:
curl -sSL https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
sudo apt update && sudo apt install ngrok
```

**Authenticate ngrok** (free account at ngrok.com):
```bash
ngrok config add-authtoken YOUR_NGROK_AUTH_TOKEN
```

---

### 2. Run Setup Script

```bash
bash setup.sh
```

This automatically:
- Creates Python virtual environment
- Installs all dependencies  
- Creates `recordings/`, `transcripts/`, `logs/` directories
- Pre-trains the risk prediction model
- Creates `.env` from `.env.example`

---

### 3. Fill in .env

```bash
nano .env  # or open in your editor
```

**What to change:**

```bash
# ─── REQUIRED: Change all of these ───────────────────────────

# From Twilio Console → Account Info
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_PHONE_NUMBER=+15551234567   # The number you bought in Twilio

# From ngrok (update every time you restart ngrok)
NGROK_URL=https://abc123.ngrok-free.app

# From Deepgram console
DEEPGRAM_API_KEY=your_key_here

# From Cartesia dashboard
CARTESIA_API_KEY=your_key_here

# From Anthropic console
ANTHROPIC_API_KEY=sk-ant-your_key_here

# Your phone number (must be verified in Twilio trial accounts)
TEST_PHONE_NUMBER=+1YOUR_PHONE_NUMBER

# ─── OPTIONAL: Change these to test different scenarios ──────
TEST_CUSTOMER_NAME=John Smith
TEST_DEBT_AMOUNT=420.00
TEST_ACCOUNT_NUMBER=ACC-9823
TEST_DAYS_OVERDUE=45
```

---

### 4. Twilio Trial Account: Verify Your Phone Number

> ⚠️ **Twilio trial accounts can only call verified numbers.**

1. Go to [console.twilio.com](https://console.twilio.com)
2. Navigate to: **Phone Numbers → Verified Caller IDs**
3. Click **Add a new Caller ID**
4. Enter your personal phone number and verify via SMS/call

Set this verified number as `TEST_PHONE_NUMBER` in `.env`.

---

### 5. Run the Agent

**Terminal 1 — Start ngrok:**
```bash
ngrok http 8000
```
Copy the `https://xxxx.ngrok-free.app` URL → paste as `NGROK_URL` in `.env`

**Terminal 2 — Start the server:**
```bash
source venv/bin/activate
python server.py
```

You should see:
```
✅ Twilio manager ready.
INFO: Uvicorn running on http://0.0.0.0:8000
```

**Terminal 3 — Trigger a test call:**
```bash
curl -X POST http://localhost:8000/test-call
```

**Your phone will ring. Answer it. Talk to Sarah.**

---

## 📞 Making Custom Calls

```bash
curl -X POST http://localhost:8000/call/initiate \
  -H "Content-Type: application/json" \
  -d '{
    "to_number": "+15559876543",
    "customer_name": "Jane Doe",
    "debt_amount": 875.50,
    "account_number": "ACC-4421",
    "days_overdue": 60
  }'
```

**Response:**
```json
{
  "session_id": "sess_a1b2c3d4e5f6",
  "call_sid": "CAxxxx",
  "status": "ringing",
  "message": "Calling +15559876543. Answer your phone!"
}
```

---

## 📊 Viewing Results

After each call, check `transcripts/`:

```bash
# See all transcripts
ls transcripts/

# Read the call transcript (JSON with per-turn sentiment)
cat transcripts/sess_*_20241215*.json | python3 -m json.tool

# Read the call summary + risk score
cat transcripts/sess_*_summary.json | python3 -m json.tool
```

**Example summary output:**
```json
{
  "session_id": "sess_a1b2c3_20241215_143022",
  "customer_name": "John Smith",
  "debt_amount": "$420.00",
  "call_outcome": "promise_to_pay",
  "plan_type": "partial",
  "payment_amount_agreed": 210.0,
  "payment_promise_date": "2025-01-15",
  "sentiment": "cooperative",
  "risk_score": 0.8312,
  "risk_label": "LOW_RISK",
  "call_duration_seconds": 98.7,
  "turn_count": 14,
  "notes": "Payment plan agreed: partial. Customer promised payment by: 2025-01-15."
}
```

**Call outcomes:**
- `promise_to_pay` — Customer committed to pay
- `refused` — Customer refused all offers  
- `dispute` — Customer disputes the debt
- `already_paid` — Customer claims they paid
- `call_dropped` — Connection lost

**Risk labels:**
- `LOW_RISK` — Score ≥ 0.75 (likely to pay)
- `MEDIUM_RISK` — Score 0.45–0.74
- `HIGH_RISK` — Score < 0.45 (unlikely to pay)

---

## 🧪 Run Offline Tests (No API Keys Needed)

```bash
source venv/bin/activate
pytest tests/test_offline.py -v
```

Tests all business logic: conversation state, negotiation engine, sentiment analysis, risk model.

---

## 🏗️ Architecture

```
Your Terminal                    Customer's Phone
     │                                  │
     │  POST /call/initiate             │
     ▼                                  │
  FastAPI ──── Twilio API ─────────────▶ RING
  Server                                │
     │                                  │ (answered)
     │◀─── POST /twiml (webhook) ───────┘
     │
     │  Returns TwiML with WebSocket URL
     │
     ▼
  WebSocket /ws/{session_id}
     │
     │  Twilio streams mulaw audio
     ▼
  Pipecat Pipeline:
  ┌─────────────────────────────────┐
  │  Twilio Audio In                │
  │       ↓                        │
  │  Deepgram STT (nova-2-phonecall)│
  │       ↓                        │
  │  Sentiment Monitor              │ → updates ConversationManager
  │       ↓                        │
  │  Claude 3.5 Sonnet (LLM)       │ ← system prompt with customer data
  │       ↓                        │
  │  Cartesia TTS (female voice)   │
  │       ↓                        │
  │  Twilio Audio Out              │
  └─────────────────────────────────┘
          │
          │ (call ends)
          ▼
  Post-Call Processing:
  - Save transcript JSON
  - HuggingFace sentiment summary
  - sklearn risk prediction
  - Generate summary report
```

---

## 📁 Project Structure

```
debt-collection-agent/
│
├── agent/
│   ├── voice_agent.py          # Pipecat pipeline + call lifecycle
│   ├── conversation_manager.py # State tracking, outcome detection
│   └── negotiation_engine.py   # 5-tier adaptive negotiation
│
├── ml/
│   ├── sentiment_analysis.py   # HuggingFace real-time emotion
│   └── risk_model.py           # sklearn repayment probability
│
├── telephony/
│   └── twilio_call.py          # Outbound calls + TwiML
│
├── flows/
│   └── debt_collection_flow.yaml  # Conversation flow reference
│
├── recordings/                 # Auto-saved MP3s
├── transcripts/                # Auto-saved JSON transcripts
├── logs/                       # server.log, agent.log
├── tests/
│   └── test_offline.py         # 20+ unit tests
│
├── server.py                   # FastAPI entry point ← START HERE
├── setup.sh                    # One-command setup
├── .env.example                # Copy to .env
└── requirements.txt
```

---

## ❓ Troubleshooting

| Problem | Fix |
|---------|-----|
| `Twilio not configured` | Check TWILIO_* values in `.env`, restart server |
| `TEST_PHONE_NUMBER not set` | Add your number to `.env` |
| Call rings but no voice | Check DEEPGRAM_API_KEY and CARTESIA_API_KEY |
| `ngrok` errors | Restart ngrok, copy new URL to `.env`, restart server |
| `No module named 'anthropic'` | Run `pip install -r requirements.txt` in venv |
| Call goes to voicemail | Answer faster — Twilio has AMD timeout |
| Sentiment model slow | First run downloads ~500MB; subsequent runs use cache |
| Port 8000 in use | Change PORT in `.env` and update ngrok command |

> **ngrok reminder:** Free ngrok URLs change every restart. Always update `NGROK_URL` in `.env` after restarting ngrok.

---

## ⚖️ Legal Note

This is a technical demo. Before production use, consult a lawyer on FDCPA, TCPA, and your state's call recording laws. The agent's compliance guardrails (no threats, instant opt-out, dispute acknowledgment) are built in but do not substitute for legal review.
