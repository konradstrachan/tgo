import asyncio
import json
import os
import time
from typing import AsyncGenerator, Optional, List

import aiohttp
from telegram import (
    Update,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)

# -------------------------
# Configuration (env-based)
# -------------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
# Comma-separated list of allowed user IDs
ALLOWED_USER_IDS = [
    int(uid.strip()) for uid in os.environ.get("ALLOWED_USER_IDS", "").split(",") if uid.strip()
]
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")  # default model name
OLLAMA_TIMEOUT_SECS = int(os.environ.get("OLLAMA_TIMEOUT_SECS", "300"))  # per request timeout
STREAM_EDIT_THROTTLE_SECS = float(os.environ.get("STREAM_EDIT_THROTTLE_SECS", "0.6"))  # edit rate

TELEGRAM_MAX_LEN = 4096  # Telegram hard limit per message

# System prompts / modes
SYSTEM_PROMPT_BRIEF = "Answer briefly in a couple of sentences."
SYSTEM_PROMPT_FULL = "Provide a clear, comprehensive, and accurate answer."
CURRENT_MODE = "brief"  # default: brief answers for normal messages

if not TELEGRAM_BOT_TOKEN or not ALLOWED_USER_IDS:
    raise SystemExit(
        "Please set TELEGRAM_BOT_TOKEN and ALLOWED_USER_IDS environment variables."
    )

# -------------------------
# Helpers
# -------------------------
def is_authorized(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id in ALLOWED_USER_IDS)

async def get_ollama_models(session: aiohttp.ClientSession, host: str) -> List[str]:
    """
    Query Ollama for available models via /api/tags and return a list of model names.
    """
    url = f"{host.rstrip('/')}/api/tags"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        resp.raise_for_status()
        data = await resp.json()
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        return sorted(dict.fromkeys(models))

def chunk_text(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    return [text[i:i+limit] for i in range(0, len(text), limit)]

async def stream_ollama_chat(
    session: aiohttp.ClientSession,
    prompt: str,
    model: str,
    host: str,
    timeout: int,
    system: Optional[str] = None,
) -> AsyncGenerator[str, None]:
    """
    Streams text chunks from Ollama's /api/chat endpoint.
    Yields incremental text pieces as they arrive.
    """
    url = f"{host.rstrip('/')}/api/chat"
    payload = {
        "model": model,
        "stream": True,
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        payload["messages"].insert(0, {"role": "system", "content": system})

    timeout_obj = aiohttp.ClientTimeout(total=timeout)
    async with session.post(url, json=payload, timeout=timeout_obj) as resp:
        resp.raise_for_status()
        async for line in resp.content:
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8").strip() or "{}")
            except json.JSONDecodeError:
                continue
            if "message" in data and data["message"] and "content" in data["message"]:
                chunk = data["message"]["content"]
                if chunk:
                    yield chunk
            if data.get("done"):
                break

async def send_or_edit_streamed(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text_stream: AsyncGenerator[str, None],
) -> None:
    """
    Sends a placeholder message and live-edits it as chunks arrive.
    If output grows beyond Telegram limit, sends additional messages in sequence.
    """
    chat_id = update.effective_chat.id
    sent = await context.bot.send_message(chat_id=chat_id, text="⏳ Running via Ollama…")
    accumulated = ""
    last_edit = 0.0

    try:
        async for piece in text_stream:
            accumulated += piece
            now = time.time()

            if len(accumulated) > TELEGRAM_MAX_LEN:
                chunks = chunk_text(accumulated, TELEGRAM_MAX_LEN)
                for c in chunks[:-1]:
                    try:
                        await sent.edit_text(
                            c[:TELEGRAM_MAX_LEN],
                            parse_mode=ParseMode.MARKDOWN,
                            disable_web_page_preview=True,
                        )
                    except Exception:
                        pass
                    sent = await context.bot.send_message(
                        chat_id=chat_id, text="…", disable_web_page_preview=True
                    )
                accumulated = chunks[-1]

            if now - last_edit >= STREAM_EDIT_THROTTLE_SECS:
                try:
                    await sent.edit_text(
                        accumulated if accumulated.strip() else "…",
                        parse_mode=ParseMode.MARKDOWN,
                        disable_web_page_preview=True,
                    )
                    last_edit = now
                except Exception:
                    pass

        if not accumulated.strip():
            await sent.edit_text("✅ Done (no output).")
            return

        chunks = chunk_text(accumulated, TELEGRAM_MAX_LEN)
        try:
            await sent.edit_text(
                chunks[0],
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
        except Exception:
            await sent.edit_text(chunks[0])
        for c in chunks[1:]:
            await context.bot.send_message(
                chat_id=chat_id,
                text=c,
                parse_mode=ParseMode.MARKDOWN,
                disable_web_page_preview=True,
            )
    except Exception as e:
        await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {e}")

def build_models_keyboard(models: List[str], per_row: int = 2) -> InlineKeyboardMarkup:
    buttons: List[List[InlineKeyboardButton]] = []
    row: List[InlineKeyboardButton] = []
    for name in models:
        row.append(InlineKeyboardButton(text=name, callback_data=f"setmodel:{name}"))
        if len(row) >= per_row:
            buttons.append(row)
            row = []
    if row:
        buttons.append(row)
    buttons.append([InlineKeyboardButton(text="🔄 Refresh", callback_data="models:refresh")])
    return InlineKeyboardMarkup(buttons)

def _log_query(user_id: int, prompt: str) -> None:
    # Log to console whenever someone queries something
    print(f"{user_id} : {prompt}")

# -------------------------
# Handlers
# -------------------------
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    await update.message.reply_text(
        "🤖 Ready. Send me a prompt and I’ll run it on your local Ollama instance.\n"
        f"Current model: `{OLLAMA_MODEL}`\n"
        "Commands:\n"
        "• /models — list models and pick one\n"
        "• /model <name> — set model by name\n"
        "• /whoami — returns your Telegram user ID\n"
        "• /ask <prompt> — brief answer (couple of sentences)\n"
        "• /full [prompt] — detailed answer; with no prompt switches default mode to full",
        parse_mode=ParseMode.MARKDOWN,
    )

async def whoami_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Unauthenticated: respond with the ID of the calling user
    user = update.effective_user
    if user:
        await update.message.reply_text(f"Your Telegram user ID is: {user.id}")
    else:
        await update.message.reply_text("Could not determine your Telegram user ID.")

async def ask_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Brief one-off query (authorized users only)
    if not is_authorized(update):
        return
    user = update.effective_user
    prompt = " ".join(context.args).strip()
    if not prompt:
        await update.message.reply_text("Usage: /ask <your question>")
        return

    _log_query(user.id, prompt)
    async with aiohttp.ClientSession() as session:
        text_stream = stream_ollama_chat(
            session=session,
            prompt=prompt,
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT_SECS,
            system=SYSTEM_PROMPT_BRIEF,
        )
        await send_or_edit_streamed(update, context, text_stream)

async def full_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Detailed query or toggle default mode to full
    if not is_authorized(update):
        return
    global CURRENT_MODE
    user = update.effective_user
    prompt = " ".join(context.args).strip()

    if not prompt:
        CURRENT_MODE = "full"
        await update.message.reply_text("✅ Default mode set to: full (normal messages will be detailed).")
        return

    _log_query(user.id, prompt)
    async with aiohttp.ClientSession() as session:
        text_stream = stream_ollama_chat(
            session=session,
            prompt=prompt,
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT_SECS,
            system=SYSTEM_PROMPT_FULL,
        )
        await send_or_edit_streamed(update, context, text_stream)

async def model_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    global OLLAMA_MODEL
    if not context.args:
        await update.message.reply_text(
            f"Current model: `{OLLAMA_MODEL}`\nUse /models to pick from a list.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return
    OLLAMA_MODEL = " ".join(context.args).strip()
    await update.message.reply_text(
        f"✅ Model set to: `{OLLAMA_MODEL}`", parse_mode=ParseMode.MARKDOWN
    )

async def models_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authorized(update):
        return
    async with aiohttp.ClientSession() as session:
        try:
            models = await get_ollama_models(session, OLLAMA_HOST)
        except Exception as e:
            await update.message.reply_text(f"❌ Could not fetch models: {e}")
            return

    if not models:
        await update.message.reply_text("No models found on Ollama. Try `ollama pull <model>`.")
        return

    await update.message.reply_text(
        text=f"Select a model (current: `{OLLAMA_MODEL}`):",
        reply_markup=build_models_keyboard(models),
        parse_mode=ParseMode.MARKDOWN,
    )

async def models_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return
    if not is_authorized(update):
        await query.answer()
        return

    data = query.data or ""
    global OLLAMA_MODEL

    if data.startswith("setmodel:"):
        new_model = data.split("setmodel:", 1)[1].strip()
        if not new_model:
            await query.answer("Invalid model", show_alert=True)
            return
        OLLAMA_MODEL = new_model
        try:
            await query.edit_message_text(
                text=f"✅ Model set to: `{OLLAMA_MODEL}`",
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            await query.edit_message_text(text=f"✅ Model set to: {OLLAMA_MODEL}")
        await query.answer("Model updated")
        return

    if data == "models:refresh":
        async with aiohttp.ClientSession() as session:
            try:
                models = await get_ollama_models(session, OLLAMA_HOST)
            except Exception as e:
                await query.answer()
                await query.edit_message_text(f"❌ Could not fetch models: {e}")
                return
        if not models:
            await query.answer()
            await query.edit_message_text("No models found on Ollama. Try `ollama pull <model>`.")
            return
        await query.answer("Refreshed")
        try:
            await query.edit_message_text(
                text=f"Select a model (current: `{OLLAMA_MODEL}`):",
                reply_markup=build_models_keyboard(models),
                parse_mode=ParseMode.MARKDOWN,
            )
        except Exception:
            await query.edit_message_text(
                text=f"Select a model (current: {OLLAMA_MODEL}):",
                reply_markup=build_models_keyboard(models),
            )
        return

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # Normal text message: obey CURRENT_MODE
    if not is_authorized(update):
        return
    user = update.effective_user
    prompt = (update.message.text or "").strip()
    if not prompt:
        await update.message.reply_text("Send text only.")
        return

    _log_query(user.id, prompt)

    system_prompt = SYSTEM_PROMPT_BRIEF if CURRENT_MODE == "brief" else SYSTEM_PROMPT_FULL

    async with aiohttp.ClientSession() as session:
        text_stream = stream_ollama_chat(
            session=session,
            prompt=prompt,
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT_SECS,
            system=system_prompt,
        )
        await send_or_edit_streamed(update, context, text_stream)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if isinstance(update, Update) and update and update.effective_chat and (
            isinstance(update, Update) and (update.effective_user and update.effective_user.id in ALLOWED_USER_IDS)
        ):
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"❌ Internal error: {context.error}",
            )
    except Exception:
        pass

# -------------------------
# Entrypoint
# -------------------------
def main() -> None:
    app: Application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("whoami", whoami_cmd))      # unauthenticated
    app.add_handler(CommandHandler("ask", ask_cmd))            # brief, one-off
    app.add_handler(CommandHandler("full", full_cmd))          # detailed or toggle default
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CallbackQueryHandler(models_callback, pattern="^(setmodel:|models:refresh)"))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_error_handler(error_handler)

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
