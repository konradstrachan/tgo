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

# Activate venv if exists
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# Run the bot
exec python3 bot.py
