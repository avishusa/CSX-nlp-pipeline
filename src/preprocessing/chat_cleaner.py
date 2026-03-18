"""
src/preprocessing/chat_cleaner.py
====================================
CSX NLP Pipeline — Step 11: Chat Cleaner

Cleans raw chatbot log text before intent classification.
Extracts customer messages, strips bot responses, session
metadata, emoji, and chat-specific noise.

Usage:
    from src.preprocessing.chat_cleaner import clean_chat

    text = clean_chat(raw_chat_log_string)

    # To keep both sides (e.g. for context):
    text = clean_chat(raw_chat_log_string, customer_only=False)
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Session / channel metadata lines
_METADATA_LINE = re.compile(
    r"^(?:session\s*(?:id|start|end|duration)|channel\s*:|"
    r"chat\s*(?:id|start|end)|user\s*id\s*:|bot\s*id\s*:|"
    r"platform\s*:|source\s*:|conversation\s*id\s*:|"
    r"timestamp\s*:|queue\s*:|agent\s*transfer\s*:).*$",
    re.IGNORECASE | re.MULTILINE,
)

# Timestamps inside log lines: [10:32:45], (2025-03-10 09:15:00), 10:32 —
_TIMESTAMP = re.compile(
    r"\[?\(?\d{4}-\d{2}-\d{2}\s*\d{2}:\d{2}(?::\d{2})?\)?\]?\s*[-—]?\s*"
    r"|\[?\(?\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?\)?\]?\s*[-—]?\s*",
    re.IGNORECASE,
)

# Speaker labels
# Bot side: "Bot:", "CSX Bot:", "Agent:", "System:", "Assistant:", "Virtual Agent:"
_BOT_LINE = re.compile(
    r"^\s*(?:\[|\()?(?:bot|csx\s*bot|system|assistant|virtual\s*agent|"
    r"automated\s*response|chatbot|agent|rep(?:resentative)?)(?:\]|\))?\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Customer side labels: "User:", "Customer:", "You:", "Client:"
_CUSTOMER_LINE = re.compile(
    r"^\s*(?:\[|\()?(?:user|customer|you|client|caller)(?:\]|\))?\s*:\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)

# All speaker labels (for customer_only=False mode)
_SPEAKER_LABEL = re.compile(
    r"^\s*(?:\[|\()?(?:bot|csx\s*bot|system|assistant|virtual\s*agent|"
    r"automated\s*response|chatbot|agent|rep(?:resentative)?|"
    r"user|customer|you|client|caller)(?:\]|\))?\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Quick-reply button echoes — exact matches of common CSX chatbot menu options
# These are tapped buttons recorded as user messages, not free-text intent
_QUICK_REPLY = re.compile(
    r"^(?:track\s+(?:my\s+)?shipment|get\s+(?:a\s+)?rate\s+quote|"
    r"report\s+(?:an?\s+)?(?:issue|damage|loss)|schedule\s+(?:a\s+)?pickup|"
    r"cancel\s+(?:my\s+)?shipment|speak\s+(?:to\s+)?(?:an?\s+)?agent|"
    r"main\s+menu|go\s+back|yes|no|help|start\s+over|other)$",
    re.IGNORECASE | re.MULTILINE,
)

# Emoji — Unicode ranges covering emoticons, symbols, pictographs
_EMOJI = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002600-\U000027BF"  # misc symbols
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002500-\U00002BEF"  # box drawing / misc technical
    "]+",
    flags=re.UNICODE,
)

# System event messages logged inline
_SYSTEM_EVENT = re.compile(
    r"\[(?:typing|read|delivered|seen|connected|disconnected|"
    r"transferred|escalated|session\s*(?:started|ended)|"
    r"file\s*(?:sent|received)|image\s*(?:sent|received))[^\]]*\]",
    re.IGNORECASE,
)

# URLs
_URL = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

# Repeated punctuation artifacts from chat ("???", "!!!", "...")
_REPEAT_PUNCT = re.compile(r"([!?.]){2,}")

# Whitespace
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_CRLF          = re.compile(r"\r\n")
_ZERO_WIDTH    = re.compile(r"[\u200b\u200c\u200d\ufeff]")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def clean_chat(text: str, customer_only: bool = True) -> str:
    """
    Clean a raw chatbot log string.

    Args:
        text:          Raw chat log string.
        customer_only: If True (default), extract only customer messages
                       — best for intent classification.
                       If False, keep all turns but strip labels/noise.

    Returns:
        Cleaned text string. Empty string if input is invalid.

    Steps (in order):
      1.  Guard against non-string / empty input
      2.  Normalize line endings and unicode
      3.  Strip session metadata lines
      4.  Strip timestamps
      5.  Strip system event markers
      6.  Strip emoji
      7a. customer_only=True  → extract customer messages only
      7b. customer_only=False → strip all speaker labels, keep all text
      8.  Strip quick-reply button echoes
      9.  Normalize URLs
      10. Clean repeated punctuation
      11. Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Line endings + unicode
    text = _CRLF.sub("\n", text)
    text = _ZERO_WIDTH.sub("", text)
    text = unicodedata.normalize("NFC", text)

    # 2. Metadata lines
    text = _METADATA_LINE.sub("", text)

    # 3. Timestamps
    text = _TIMESTAMP.sub("", text)

    # 4. System event markers
    text = _SYSTEM_EVENT.sub("", text)

    # 5. Emoji
    text = _EMOJI.sub("", text)

    # 6. Speaker extraction
    if customer_only:
        text = _extract_customer_messages(text)
    else:
        text = _SPEAKER_LABEL.sub("", text)

    # 7. Quick-reply echoes — remove lines that are pure button taps
    text = _QUICK_REPLY.sub("", text)

    # 8. URLs
    text = _URL.sub("[URL]", text)

    # 9. Repeated punctuation
    text = _REPEAT_PUNCT.sub(r"\1", text)

    # 10. Whitespace
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_customer_messages(text: str) -> str:
    """
    Return only text from customer-side chat turns.

    Strategy:
      - If explicit speaker labels exist, extract only customer lines.
      - If no labels detected (raw message dump), strip bot lines we
        can identify and return the rest.
    """
    customer_matches = _CUSTOMER_LINE.findall(text)

    if customer_matches:
        return "\n".join(line.strip() for line in customer_matches if line.strip())

    # Unlabeled log — strip bot lines, return remainder
    text = _BOT_LINE.sub("", text)
    text = _SPEAKER_LABEL.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/preprocessing/chat_cleaner.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """\
Session ID: CHT-20250310-4492
Channel: web
Platform: csx.com/chat

[09:15:02] Bot: Hi there! Welcome to CSX Customer Support. How can I help you today?
[09:15:08] User: hi i need to check on my shipment its been delayed 😤
[09:15:09] [typing]
[09:15:11] Bot: I can help with that! Please provide your car number or waybill number.
[09:15:24] User: car number is CSXT 778833, waybill 9912345
[09:15:25] Bot: Thank you! Let me look that up for you. [typing]
[09:15:30] Bot: I can see car CSXT 778833 is currently at the Memphis yard. Expected arrival Atlanta is March 12th.
[09:15:38] User: march 12?? thats 4 days late, we needed it by march 8 for our plant shutdown schedule
[09:15:41] Bot: I understand your frustration. Would you like me to escalate this to a live agent?
[09:15:45] User: yes please, this is urgent
[09:15:46] [transferred]
[09:15:50] Bot: Connecting you now. Reference number for your case is REF-20250310-9981.
[09:15:55] User: Track my shipment
"""

    print("RAW INPUT:")
    print("=" * 60)
    print(SAMPLE)

    print("\nCLEANED — CUSTOMER ONLY (default, for intent classification):")
    print("=" * 60)
    print(clean_chat(SAMPLE, customer_only=True))

    print("\nCLEANED — ALL TURNS (for context / NER):")
    print("=" * 60)
    print(clean_chat(SAMPLE, customer_only=False))