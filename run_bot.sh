#!/usr/bin/env bash
set -euo pipefail

# Move to script's directory
cd "$(dirname "$0")"

# Load environment variables from .env
if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
else
    echo "❌ .env file not found. Please create it first."
    exit 1
fi

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the bot
exec python3 bot.py

