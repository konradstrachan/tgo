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
ALLOWED_USER_ID = int(os.environ.get("ALLOWED_USER_ID", "0"))  # your numeric Telegram user ID
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3")  # default model name
OLLAMA_TIMEOUT_SECS = int(os.environ.get("OLLAMA_TIMEOUT_SECS", "300"))  # per request timeout
STREAM_EDIT_THROTTLE_SECS = float(os.environ.get("STREAM_EDIT_THROTTLE_SECS", "0.6"))  # edit rate

TELEGRAM_MAX_LEN = 4096  # Telegram hard limit per message

if not TELEGRAM_BOT_TOKEN or not ALLOWED_USER_ID:
    raise SystemExit(
        "Please set TELEGRAM_BOT_TOKEN and ALLOWED_USER_ID environment variables."
    )

# -------------------------
# Helpers
# -------------------------
def is_authorized(update: Update) -> bool:
    user = update.effective_user
    return bool(user and user.id == ALLOWED_USER_ID)

async def get_ollama_models(session: aiohttp.ClientSession, host: str) -> List[str]:
    """
    Query Ollama for available models via /api/tags and return a list of model names.
    """
    url = f"{host.rstrip('/')}/api/tags"
    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
        resp.raise_for_status()
        data = await resp.json()
        # Response shape: { "models": [ { "name": "llama3:8b", ... }, ... ] }
        models = [m.get("name") for m in data.get("models", []) if m.get("name")]
        # Deduplicate & stable sort (alpha)
        return sorted(dict.fromkeys(models))

def chunk_text(text: str, limit: int = TELEGRAM_MAX_LEN) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + limit, len(text))
        chunks.append(text[start:end])
        start = end
    return chunks

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

            # When the accumulated text exceeds 4096, send fixed chunks and keep the tail editable.
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

            # Throttle edits to avoid hitting rate limits
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

        # Final flush: send any remaining text cleanly in chunks
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
    # Add refresh button
    buttons.append([InlineKeyboardButton(text="🔄 Refresh", callback_data="models:refresh")])
    return InlineKeyboardMarkup(buttons)

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
        "• /model <name> — set model by name",
        parse_mode=ParseMode.MARKDOWN,
    )

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
    if not is_authorized(update):
        return
    prompt = (update.message.text or "").strip()
    if not prompt:
        await update.message.reply_text("Send text only.")
        return

    async with aiohttp.ClientSession() as session:
        text_stream = stream_ollama_chat(
            session=session,
            prompt=prompt,
            model=OLLAMA_MODEL,
            host=OLLAMA_HOST,
            timeout=OLLAMA_TIMEOUT_SECS,
        )
        await send_or_edit_streamed(update, context, text_stream)

async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        if isinstance(update, Update) and is_authorized(update) and update.effective_chat:
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
    app.add_handler(CommandHandler("model", model_cmd))
    app.add_handler(CommandHandler("models", models_cmd))
    app.add_handler(CallbackQueryHandler(models_callback, pattern="^(setmodel:|models:refresh)"))
    app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message))
    app.add_error_handler(error_handler)

    app.run_polling(close_loop=False)

if __name__ == "__main__":
    main()
