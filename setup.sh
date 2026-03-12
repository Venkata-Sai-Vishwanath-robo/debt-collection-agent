#!/bin/bash
# ============================================================
#  setup.sh — One-command setup for the Debt Collection Agent
#  Usage: bash setup.sh
# ============================================================

set -e  # Exit on any error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo ""
echo -e "${BLUE}╔══════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   Debt Collection Voice Agent — Setup        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════╝${NC}"
echo ""

# ── Step 1: Python version check ─────────────────────────────
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
PYTHON=$(which python3 || which python)
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo -e "  Python: $PY_VERSION"
if [[ "$PY_VERSION" < "3.10" ]]; then
    echo -e "${RED}ERROR: Python 3.10+ required. Please upgrade.${NC}"
    exit 1
fi
echo -e "${GREEN}  ✓ Python OK${NC}"

# ── Step 2: Virtual environment ───────────────────────────────
echo ""
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    $PYTHON -m venv venv
    echo -e "${GREEN}  ✓ venv created${NC}"
else
    echo -e "  venv already exists, skipping."
fi

source venv/bin/activate
echo -e "${GREEN}  ✓ venv activated${NC}"

# ── Step 3: Install dependencies ──────────────────────────────
echo ""
echo -e "${YELLOW}[3/6] Installing Python dependencies...${NC}"
echo -e "  (This may take 2-5 minutes the first time)"
pip install --upgrade pip --quiet

# Install CPU-only torch first (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu --quiet
echo -e "${GREEN}  ✓ PyTorch (CPU) installed${NC}"

pip install -r requirements.txt --quiet
echo -e "${GREEN}  ✓ All dependencies installed${NC}"

# ── Step 4: Create directories ───────────────────────────────
echo ""
echo -e "${YELLOW}[4/6] Creating directories...${NC}"
mkdir -p recordings transcripts logs
echo -e "${GREEN}  ✓ recordings/, transcripts/, logs/ created${NC}"

# ── Step 5: Pre-train risk model ──────────────────────────────
echo ""
echo -e "${YELLOW}[5/6] Pre-training risk model (takes ~10 seconds)...${NC}"
$PYTHON -c "
from ml.risk_model import train_and_save_model
train_and_save_model()
print('Risk model trained and saved.')
" && echo -e "${GREEN}  ✓ Risk model ready${NC}" || echo -e "${YELLOW}  ⚠ Risk model will train on first call${NC}"

# ── Step 6: .env file ─────────────────────────────────────────
echo ""
echo -e "${YELLOW}[6/6] Checking .env file...${NC}"
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo -e "${YELLOW}  ⚠ .env created from .env.example${NC}"
    echo -e "${RED}  ★ YOU MUST EDIT .env before running the server!${NC}"
else
    echo -e "${GREEN}  ✓ .env already exists${NC}"
fi

# ── Done ──────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   Setup complete! Next steps:                ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  1. Fill in your API keys in ${YELLOW}.env${NC}"
echo ""
echo -e "  2. Start ngrok in a new terminal:"
echo -e "     ${BLUE}ngrok http 8000${NC}"
echo -e "     Copy the https URL → paste as NGROK_URL in .env"
echo ""
echo -e "  3. Start the server:"
echo -e "     ${BLUE}source venv/bin/activate${NC}"
echo -e "     ${BLUE}python server.py${NC}"
echo ""
echo -e "  4. Make a test call:"
echo -e "     ${BLUE}curl -X POST http://localhost:8000/test-call${NC}"
echo ""
echo -e "  5. View results:"
echo -e "     ${BLUE}ls transcripts/${NC}"
echo ""
