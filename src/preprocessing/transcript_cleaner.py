"""
src/preprocessing/transcript_cleaner.py
=========================================
CSX NLP Pipeline — Step 10: Transcript Cleaner

Cleans raw call transcript text before intent classification.
Extracts customer-side utterances, strips speaker labels,
timestamps, filler words, and system annotations.

Usage:
    from src.preprocessing.transcript_cleaner import clean_transcript

    text = clean_transcript(raw_transcript_string)

    # To keep both sides (e.g. for context):
    text = clean_transcript(raw_transcript_string, customer_only=False)
"""

import re
import unicodedata


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Timestamps: [00:01:32], (0:45), 00:01:32 —, 1:23:45
_TIMESTAMP = re.compile(
    r"\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?\s*[-—]?\s*"
)

# Speaker labels — broad coverage of real-world transcript formats:
#   "Agent:", "Customer:", "Rep:", "Caller:", "CSX Agent:"
#   "[Agent]", "(Customer)", "AGENT:", "CUSTOMER 1:"
_SPEAKER_LABEL = re.compile(
    r"^\s*(?:\[|\()?(?:agent|customer|caller|rep(?:resentative)?|"
    r"csx\s*(?:agent|rep)?|operator|dispatcher|clerk|staff|"
    r"client|user|speaker\s*\d*)\s*\d*(?:\]|\))?\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# Agent-only lines — full line starts with an agent-side label
# Used when customer_only=True to drop entire agent turns
_AGENT_LINE = re.compile(
    r"^\s*(?:\[|\()?(?:agent|rep(?:resentative)?|csx\s*(?:agent|rep)?|"
    r"operator|dispatcher|clerk|staff|speaker\s*[02-9]\d*)(?:\]|\))?\s*:\s*.*$",
    re.IGNORECASE | re.MULTILINE,
)

# Customer-only lines — used to identify lines to keep
_CUSTOMER_LINE = re.compile(
    r"^\s*(?:\[|\()?(?:customer|caller|client|user|speaker\s*1)(?:\]|\))?\s*:\s*(.*?)$",
    re.IGNORECASE | re.MULTILINE,
)

# System / transcription annotations
_SYSTEM_ANNOTATION = re.compile(
    r"\[(?:crosstalk|inaudible|overtalk|pause|silence|laughter|"
    r"background\s*noise|music|hold\s*music|beep|noise|ph|sic|"
    r"unintelligible|unclear)[^\]]*\]",
    re.IGNORECASE,
)

# Call metadata headers (lines at the top of transcript files)
_METADATA_LINE = re.compile(
    r"^(?:call\s*(?:id|date|time|duration|recording|transcript)|"
    r"date\s*:|time\s*:|duration\s*:|agent\s*id\s*:|queue\s*:|"
    r"session\s*id\s*:|dnis\s*:|ani\s*:).*$",
    re.IGNORECASE | re.MULTILINE,
)

# Filler words — whole-word match only, not substrings
_FILLER = re.compile(
    r"\b(?:uh+|um+|hmm+|mhm|uh-huh|mm-?hmm|ah+|er+|like|"
    r"you\s+know|i\s+mean|basically|literally|actually|"
    r"so\s+yeah|right\s+right|okay\s+so)\b",
    re.IGNORECASE,
)

# Repeated punctuation from transcription artifacts: "... ... ..." → "..."
_REPEAT_ELLIPSIS = re.compile(r"(?:\.{2,}\s*){2,}")
_REPEAT_DASH     = re.compile(r"-{2,}")

# Whitespace
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_MULTI_SPACE   = re.compile(r"[ \t]{2,}")
_CRLF          = re.compile(r"\r\n")
_ZERO_WIDTH    = re.compile(r"[\u200b\u200c\u200d\ufeff]")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def clean_transcript(text: str, customer_only: bool = True) -> str:
    """
    Clean a raw call transcript string.

    Args:
        text:          Raw transcript string.
        customer_only: If True (default), extract only customer-side
                       utterances — best for intent classification.
                       If False, keep all turns but strip labels/noise.

    Returns:
        Cleaned text string. Empty string if input is invalid.

    Steps (in order):
      1.  Guard against non-string / empty input
      2.  Normalize line endings and unicode
      3.  Strip call metadata header lines
      4.  Strip timestamps
      5.  Strip system annotations ([inaudible], [crosstalk], etc.)
      6a. customer_only=True  → extract customer turn text only
      6b. customer_only=False → strip all speaker labels, keep all text
      7.  Strip filler words
      8.  Clean up transcription artifacts (ellipsis, dashes)
      9.  Collapse whitespace
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1. Line endings + unicode
    text = _CRLF.sub("\n", text)
    text = _ZERO_WIDTH.sub("", text)
    text = unicodedata.normalize("NFC", text)

    # 2. Metadata header lines
    text = _METADATA_LINE.sub("", text)

    # 3. Timestamps
    text = _TIMESTAMP.sub("", text)

    # 4. System annotations
    text = _SYSTEM_ANNOTATION.sub("", text)

    # 5. Speaker extraction
    if customer_only:
        text = _extract_customer_turns(text)
    else:
        text = _SPEAKER_LABEL.sub("", text)

    # 6. Filler words
    text = _FILLER.sub("", text)

    # 7. Transcription artifacts
    text = _REPEAT_ELLIPSIS.sub("... ", text)
    text = _REPEAT_DASH.sub("-", text)

    # 8. Whitespace
    text = _MULTI_NEWLINE.sub("\n\n", text)
    text = _MULTI_SPACE.sub(" ", text)
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _extract_customer_turns(text: str) -> str:
    """
    Return only the text from customer-side speaker turns.

    Strategy:
      - If the transcript has explicit speaker labels, extract only
        customer/caller/client lines using _CUSTOMER_LINE.
      - If no speaker labels are detected (unlabeled transcript),
        fall back to returning the full text so we don't lose data.
    """
    customer_matches = _CUSTOMER_LINE.findall(text)

    if customer_matches:
        # Labeled transcript — keep only customer utterances
        return "\n".join(line.strip() for line in customer_matches if line.strip())

    # Unlabeled transcript — strip whatever agent lines we can detect,
    # return the rest
    text = _AGENT_LINE.sub("", text)
    text = _SPEAKER_LABEL.sub("", text)
    return text


# ---------------------------------------------------------------------------
# Smoke test — run with: python src/preprocessing/transcript_cleaner.py
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SAMPLE = """\
Call ID: TXN-20250310-8821
Date: 2025-03-10
Duration: 00:04:32
Agent ID: A-1042

[00:00:03] Agent: Thank you for calling CSX customer support, my name is Sarah, how can I help you today?
[00:00:09] Customer: Uh, hi yeah, so I have a shipment, car CSXT 445521, that was supposed to arrive at the Atlanta yard by March 8th and it still hasn't shown up, you know?
[00:00:21] Agent: I'm sorry to hear that, let me pull up that car number for you. Can you give me the waybill number as well?
[00:00:28] Customer: Um, yeah it's 8834521. I mean, we've been waiting for like three days now and our facility is basically at a standstill.
[00:00:41] Agent: I completely understand, that's very frustrating. I'm looking that up right now. [pause]
[00:00:52] Agent: Okay so I can see that car CSXT 445521 is currently showing delayed at Birmingham. There was a [inaudible] issue overnight.
[00:01:02] Customer: Okay so when is it actually going to get to Atlanta? We need an ETA so we can reschedule our crew.
[00:01:12] Agent: Based on what I'm seeing it should arrive by end of day tomorrow, March 11th.
[00:01:18] Customer: Alright, can you send that confirmation to our logistics email?
"""

    print("RAW INPUT:")
    print("=" * 60)
    print(SAMPLE)

    print("\nCLEANED — CUSTOMER ONLY (default, for intent classification):")
    print("=" * 60)
    print(clean_transcript(SAMPLE, customer_only=True))

    print("\nCLEANED — ALL TURNS (for context / NER):")
    print("=" * 60)
    print(clean_transcript(SAMPLE, customer_only=False))