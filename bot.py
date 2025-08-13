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
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:1b")  # preferred model name
OLLAMA_TIMEOUT_SECS = int(os.environ.get("OLLAMA_TIMEOUT_SECS", "300"))  # per request timeout
STREAM_EDIT_THROTTLE_SECS = float(os.environ.get("STREAM_EDIT_THROTTLE_SECS", "0.6"))  # edit rate

TELEGRAM_MAX_LEN = 4096  # Telegram hard limit per message

QUESTION_PLACEHOLDER = "⏳"
# Thinking placeholder shown during <think>...</think>
THINKING_PLACEHOLDER = "🧠"

# -------------------------
# System prompts / modes
# -------------------------
# Improve prompt so the bot only mentions Konrad/Anna when relevant.
BASE_INSTRUCTIONS = (
    "You will be provided with a message from a user. Your job is to reply to this message"
    "You are Anna, a helpful AI assistant. "
    "Answer accurately and directly based on the user's request. "
    "Do NOT mention Konrad Strachan or your own name unless the user explicitly asks about "
    "Konrad, the author, the bot's identity, or Anna. "
    "If the user asks about Konrad: he is a software developer and the author who named you Anna. "
    "Otherwise, avoid bringing this up. "
)

BRIEF_STYLE = "Keep the answer brief (1–3 sentences), unless the user asks for more detail."
FULL_STYLE = "Provide a clear, comprehensive, and accurate answer."

SYSTEM_PROMPT_BRIEF = f"{BASE_INSTRUCTIONS} {BRIEF_STYLE}"
SYSTEM_PROMPT_FULL = f"{BASE_INSTRUCTIONS} {FULL_STYLE}"
CURRENT_MODE = "brief"  # default: brief answers for normal messages

# Special markers for thinking control in stream
THINK_START = "\x00THINK_START\x00"
THINK_END = "\x00THINK_END\x00"

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

async def select_startup_model(preferred: str, host: str) -> str:
    """
    Ensure we start with a model that actually exists in Ollama.
    Falls back to the first available model if preferred is missing.
    """
    async with aiohttp.ClientSession() as session:
        models = await get_ollama_models(session, host)
    if not models:
        raise SystemExit(
            "No models found in Ollama. Use `ollama pull <model>` to install one."
        )
    if preferred in models:
        print(f"[startup] Using preferred model: {preferred}")
        return preferred
    fallback = models[0]
    print(f"[startup] Preferred model '{preferred}' not found. Using available model: {fallback}")
    return fallback

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
    Hides content inside <think>...</think>: shows a temporary THINKING_PLACEHOLDER,
    then removes it and continues with the real content after </think>.
    Yields only deltas (new text), not the entire accumulated text.
    Special control markers THINK_START/THINK_END are yielded to manage the placeholder.
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

        thinking = False
        sent_thinking = False
        visible_text = ""      # What the user should ultimately see
        last_yield_len = 0     # Index in visible_text we have already yielded
        buffer = ""            # Raw buffer of the current streamed chunk

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
                    buffer += chunk

                    # Process any think tags in the buffer
                    while True:
                        if not thinking and "<think>" in buffer:
                            # Emit any visible text before entering <think>
                            before, after = buffer.split("<think>", 1)
                            if before:
                                visible_text += before
                                delta = visible_text[last_yield_len:]
                                if delta:
                                    yield delta
                                    last_yield_len = len(visible_text)
                            buffer = after
                            thinking = True
                            if not sent_thinking:
                                yield THINK_START
                                sent_thinking = True
                            continue

                        if thinking and "</think>" in buffer:
                            # Consume everything up to and including </think>, produce no visible text
                            after = buffer.split("</think>", 1)[1]
                            buffer = after
                            thinking = False
                            yield THINK_END
                            continue

                        break  # No more tags to process right now

                    # If not in thinking mode, whatever remains in buffer is user-visible
                    if not thinking and buffer:
                        visible_text += buffer
                        delta = visible_text[last_yield_len:]
                        if delta:
                            yield delta
                            last_yield_len = len(visible_text)
                        buffer = ""

            if data.get("done"):
                # Flush any remaining visible text (should already be handled)
                if not thinking and buffer:
                    visible_text += buffer
                    delta = visible_text[last_yield_len:]
                    if delta:
                        yield delta
                        last_yield_len = len(visible_text)
                break

async def send_or_edit_streamed(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    text_stream: AsyncGenerator[str, None],
) -> None:
    """
    Sends a placeholder message and live-edits it as chunks arrive.
    Handles THINK_START/THINK_END to show THINKING_PLACEHOLDER temporarily and replace it afterward.
    Skips Telegram edits when the content hasn't changed to avoid 'Message is not modified' errors.
    """
    chat_id = update.effective_chat.id
    sent = await context.bot.send_message(chat_id=chat_id, text=QUESTION_PLACEHOLDER)
    displayed = ""            # What we currently show in Telegram
    has_thinking = False
    last_edit = 0.0
    last_rendered: Optional[str] = None  # Track last rendered text to avoid identical edits

    def render_text(s: str) -> str:
        # Telegram edit helper: keep within limit; show the last part if too long.
        if len(s) <= TELEGRAM_MAX_LEN:
            return s
        tail = s[-(TELEGRAM_MAX_LEN - 1):]
        return "…" + tail

    async def safe_edit(new_text: str, use_markdown: bool = True) -> None:
        nonlocal last_rendered, last_edit, sent
        rendered = render_text(new_text)
        if rendered == (last_rendered or ""):
            return  # Skip identical edits to avoid 'Message is not modified'
        try:
            if use_markdown:
                await sent.edit_text(rendered, parse_mode=ParseMode.MARKDOWN, disable_web_page_preview=True)
            else:
                await sent.edit_text(rendered)
            last_rendered = rendered
            last_edit = time.time()
        except Exception:
            # Retry once without markdown (e.g., bad markdown)
            try:
                await sent.edit_text(rendered)
                last_rendered = rendered
                last_edit = time.time()
            except Exception:
                # Still failing; swallow to keep streaming going
                pass

    try:
        async for piece in text_stream:
            now = time.time()

            if piece == THINK_START:
                if not has_thinking:
                    has_thinking = True
                    displayed = displayed.rstrip()
                    displayed = (displayed + ("\n" if displayed else "") + THINKING_PLACEHOLDER).strip()
                if now - last_edit >= STREAM_EDIT_THROTTLE_SECS:
                    await safe_edit(displayed, use_markdown=False)
                continue

            if piece == THINK_END:
                if has_thinking:
                    has_thinking = False
                    # Remove placeholder (with or without preceding newline)
                    if displayed.endswith(THINKING_PLACEHOLDER):
                        displayed = displayed[:-len(THINKING_PLACEHOLDER)].rstrip()
                    elif displayed.endswith("\n" + THINKING_PLACEHOLDER):
                        displayed = displayed[: -len("\n" + THINKING_PLACEHOLDER)]
                if now - last_edit >= STREAM_EDIT_THROTTLE_SECS:
                    await safe_edit(displayed, use_markdown=False)
                continue

            # Normal text delta
            if piece:
                displayed += piece
                if now - last_edit >= STREAM_EDIT_THROTTLE_SECS:
                    await safe_edit(displayed, use_markdown=True)

        # Final flush
        if not displayed.strip():
            await safe_edit("✅ Done (no output).", use_markdown=False)
        else:
            await safe_edit(displayed, use_markdown=True)

    except Exception as e:
        # If anything unexpected happens, report once
        try:
            await context.bot.send_message(chat_id=chat_id, text=f"❌ Error: {e}")
        except Exception:
            pass

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
    global OLLAMA_MODEL
    # Pick a valid Ollama model before starting the bot
    OLLAMA_MODEL = asyncio.run(select_startup_model(OLLAMA_MODEL, OLLAMA_HOST))

    # Ensure there's an active event loop for PTB (Python 3.13 compatibility)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

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

    app.run_polling()

if __name__ == "__main__":
    main()
