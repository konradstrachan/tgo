# Telegram Ollama Bot

# About

TgO private, locally hosted Telegram AI assistant that connects to your own Ollama instance.  
It only responds to queries from pre-approved Telegram users, and can answer in brief or detailed modes.  
It maintains per-user conversation memory, including rolling state summaries and a semantic vector store for relevant fact retrieval, enabling more context-aware responses.  
The bot supports model selection from your Ollama models, keeps all processing local for privacy, and never sends your queries to external services.

# Set up instructions

## 1. Create a Telegram Bot & Get Your Bot Token
1. Open **Telegram** (desktop, mobile, or web).
2. Search for **`@BotFather`** (this is the official Telegram bot manager).
3. Start a chat with BotFather and send: **/newbot**
4. Follow the prompts:
- **Bot Name:** Choose any display name (e.g., `Ollama Control Bot`).
- **Bot Username:** Must end in `bot` (e.g., `ollama_control_bot`).
5. BotFather will respond with something like:
Done! Congratulations...
Use this token to access the HTTP API:
123456789:ABCdefGhIJKlmNoPQRstUvWxYZ
6. Copy that token - you’ll need it in your `.env` file.

## 2. Get Your Telegram User ID

### Option A – Use a User Info Bot
1. In Telegram, search for **`@userinfobot`**.
2. Start it - it will display: ***Your ID: 987654321***
3. Copy this number for your `.env` file.

### Option B – Ask Your Bot
1. Temporarily modify your bot code to print:
```python
print(update.effective_user.id)
```
inside your /start or handle_message function.
2. Run your bot and send it a message.
3. Check your bot’s terminal output for your ID.
4. Create .env File

### Example .env file:

```
# Telegram bot settings
TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRstUvWxYZ
ALLOWED_USER_ID=987654321

# Ollama settings
OLLAMA_HOST=http://127.0.0.1:11434
OLLAMA_MODEL=llama3
OLLAMA_TIMEOUT_SECS=300
STREAM_EDIT_THROTTLE_SECS=0.6
```
## Install & Run

### Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

### Install dependencies
pip install -r requirements.txt

### Make run script executable
chmod +x run_bot.sh

### Run bot
./run_bot.sh

Once running:

Only the ALLOWED_USER_ID can interact with the bot. All other users will be ignored.