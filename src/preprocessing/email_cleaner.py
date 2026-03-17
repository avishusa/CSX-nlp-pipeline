"""
src/preprocessing/email_cleaner.py
===================================
CSX NLP Pipeline — Step 9: Email Cleaner

Cleans raw freight-rail customer service emails before tokenization and
intent classification. Handles:
  - Forwarded / reply chain stripping
  - Email headers (From, To, Subject, Date, CC)
  - Disclaimer / legal boilerplate blocks
  - Signature blocks
  - HTML tags and entities
  - Inline image / attachment placeholders
  - Excess whitespace and encoding artifacts
  - CSX-specific noise (waybill numbers, equipment IDs, PRO numbers, etc.)
  - Normalization hooks for downstream NER (optional, toggleable)

Usage:
    from src.preprocessing.email_cleaner import EmailCleaner

    cleaner = EmailCleaner()
    clean_text = cleaner.clean("... raw email body ...")

    # Or batch-process a DataFrame column:
    df["clean_text"] = cleaner.clean_batch(df["raw_email"])
"""

from __future__ import annotations

import html
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config dataclass — lets callers toggle behaviour without subclassing
# ---------------------------------------------------------------------------

@dataclass
class CleanerConfig:
    # ── Chain / header stripping ────────────────────────────────────────────
    strip_forwarded_headers: bool = True   # "---------- Forwarded message ---------"
    strip_reply_headers: bool = True       # "On Mon, Jan 1 2024, John wrote:"
    strip_email_headers: bool = True       # From: / To: / Subject: / Date: / CC:
    max_chain_depth: int = 3               # how many nested reply levels to keep (0 = strip all)

    # ── Content removal ─────────────────────────────────────────────────────
    strip_signatures: bool = True
    strip_disclaimers: bool = True
    strip_html: bool = True
    strip_urls: bool = True
    strip_emails: bool = False             # keep email addresses (useful for NER)
    strip_phone_numbers: bool = False      # keep phone numbers
    strip_csx_ids: bool = False            # keep waybills / PRO#s / equipment IDs (useful for NER)

    # ── Normalization ───────────────────────────────────────────────────────
    normalize_whitespace: bool = True
    normalize_unicode: bool = True         # decompose → NFC
    lowercase: bool = False               # intentionally off; DistilBERT is cased-friendly
    max_length: Optional[int] = 512        # truncate tokens; None = no limit

    # ── Extra custom patterns to remove (compiled regexes or raw strings) ───
    extra_removal_patterns: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Compiled pattern library
# ---------------------------------------------------------------------------

class _Patterns:
    """All compiled regex patterns, built once at import time."""

    # ── HTML ────────────────────────────────────────────────────────────────
    HTML_TAG        = re.compile(r"<[^>]+>", re.DOTALL)
    HTML_ENTITY     = re.compile(r"&[a-zA-Z]{2,6};|&#?\w+;")

    # ── Email structural noise ───────────────────────────────────────────────
    # "---------- Forwarded message ---------" and variants
    FORWARDED_BLOCK = re.compile(
        r"-{3,}\s*(?:Forwarded(?:\s+message)?|Original\s+Message|Begin\s+forwarded"
        r"|FWD|Fwd)\s*-{3,}.*?(?=\n{2,}|-{3,}|\Z)",
        re.IGNORECASE | re.DOTALL,
    )

    # "On Mon, Jan 1 2024 at 9:00 AM, John Doe <john@csx.com> wrote:"
    REPLY_HEADER = re.compile(
        r"^On\s.{5,100}(?:wrote|said)\s*:.*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Standard email headers at the start of a block
    EMAIL_HEADER_LINE = re.compile(
        r"^(?:From|To|Cc|Bcc|Subject|Date|Sent|Reply-To|Message-ID)\s*:.*$",
        re.IGNORECASE | re.MULTILINE,
    )

    # Quoted reply lines ("> some text")
    QUOTED_LINE = re.compile(r"^\s*>+.*$", re.MULTILINE)

    # ── Signature detection ──────────────────────────────────────────────────
    # Heuristic: "-- \n" or "Regards," or "Best," etc. followed by a name block
    SIG_DELIMITER = re.compile(
        r"(?:^--\s*$|^_{3,}$)",
        re.MULTILINE,
    )
    SIG_CLOSING = re.compile(
        r"\n(?:Best\s+(?:regards|wishes)|Kind\s+regards|Regards|Sincerely|"
        r"Thanks(?:\s+and\s+regards)?|Thank\s+you|Cheers|Warm\s+regards|"
        r"Respectfully|With\s+appreciation)[,.]?\s*\n",
        re.IGNORECASE,
    )

    # ── Disclaimer / legal boilerplate ───────────────────────────────────────
    # Common patterns in corporate emails
    DISCLAIMER = re.compile(
        r"(?:CONFIDENTIALITY\s+NOTICE|DISCLAIMER|LEGAL\s+NOTICE|"
        r"This\s+(?:e-?mail|message)\s+(?:and\s+any\s+attachments?\s+)?(?:is|are|may\s+be)\s+"
        r"(?:confidential|privileged|intended)|"
        r"If\s+you\s+(?:have\s+received|are\s+not\s+the\s+intended)|"
        r"The\s+information\s+(?:contained|transmitted)\s+in\s+this).*",
        re.IGNORECASE | re.DOTALL,
    )

    # ── URLs / emails / phones ───────────────────────────────────────────────
    URL = re.compile(
        r"https?://\S+|www\.\S+",
        re.IGNORECASE,
    )
    EMAIL_ADDR = re.compile(
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b"
    )
    PHONE = re.compile(
        r"(?<!\d)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)?\d{3}[\s.\-]?\d{4}(?!\d)"
    )

    # ── CSX freight-rail specific IDs ────────────────────────────────────────
    # Waybill / PRO numbers: 7–10 digit standalone numbers
    WAYBILL = re.compile(r"\b\d{7,10}\b")
    # Equipment IDs: e.g. CSXT 123456, TTX 987654
    EQUIPMENT_ID = re.compile(r"\b[A-Z]{2,4}\s?\d{6}\b")
    # Shipment / car order numbers (alphanumeric: e.g. SO-2024-00123)
    ORDER_NUMBER = re.compile(r"\b(?:SO|PO|WB|BO)-\d{4}-\d{4,6}\b", re.IGNORECASE)

    # ── Whitespace / encoding ────────────────────────────────────────────────
    MULTI_NEWLINE  = re.compile(r"\n{3,}")
    MULTI_SPACE    = re.compile(r"[ \t]{2,}")
    ZERO_WIDTH     = re.compile(r"[\u200b\u200c\u200d\ufeff]")
    WINDOWS_CRLF   = re.compile(r"\r\n")

    # ── Attachment / inline image placeholders ───────────────────────────────
    ATTACHMENT_PLACEHOLDER = re.compile(
        r"\[(?:image|cid|attachment|inline\s+image)\s*[^\]]*\]",
        re.IGNORECASE,
    )

    # ── Greeting / auto-reply boilerplate ────────────────────────────────────
    AUTO_REPLY = re.compile(
        r"^(?:This\s+is\s+an?\s+automated?|Do\s+not\s+reply\s+to\s+this|"
        r"You\s+(?:have\s+)?(?:received|are\s+receiving)\s+this).*$",
        re.IGNORECASE | re.MULTILINE,
    )


P = _Patterns()  # singleton


# ---------------------------------------------------------------------------
# EmailCleaner
# ---------------------------------------------------------------------------

class EmailCleaner:
    """
    Stateless cleaner (all state lives in `config`).
    Thread-safe: a single instance can clean emails in parallel.
    """

    def __init__(self, config: Optional[CleanerConfig] = None):
        self.config = config or CleanerConfig()
        self._extra_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.extra_removal_patterns
        ]

    # ── Public API ───────────────────────────────────────────────────────────

    def clean(self, text: str) -> str:
        """Clean a single raw email string. Returns cleaned text."""
        if not isinstance(text, str) or not text.strip():
            return ""

        cfg = self.config

        # 1. Decode HTML entities, strip tags
        if cfg.strip_html:
            text = html.unescape(text)
            text = P.HTML_TAG.sub(" ", text)
            text = P.HTML_ENTITY.sub(" ", text)

        # 2. Normalise line endings
        text = P.WINDOWS_CRLF.sub("\n", text)

        # 3. Remove zero-width / BOM characters
        if cfg.normalize_unicode:
            text = P.ZERO_WIDTH.sub("", text)
            text = unicodedata.normalize("NFC", text)

        # 4. Strip attachment placeholders
        text = P.ATTACHMENT_PLACEHOLDER.sub("", text)

        # 5. Strip auto-reply boilerplate
        text = P.AUTO_REPLY.sub("", text)

        # 6. Strip forwarded message blocks
        if cfg.strip_forwarded_headers:
            text = self._strip_forwarded(text)

        # 7. Strip reply headers ("On Jan 1 … wrote:")
        if cfg.strip_reply_headers:
            text = P.REPLY_HEADER.sub("", text)

        # 8. Strip email header lines (From: / To: / Subject: etc.)
        if cfg.strip_email_headers:
            text = P.EMAIL_HEADER_LINE.sub("", text)

        # 9. Strip quoted lines ("> …")
        text = P.QUOTED_LINE.sub("", text)

        # 10. Strip disclaimer blocks
        if cfg.strip_disclaimers:
            text = self._strip_disclaimer(text)

        # 11. Strip signature blocks
        if cfg.strip_signatures:
            text = self._strip_signature(text)

        # 12. Strip / normalise URLs
        if cfg.strip_urls:
            text = P.URL.sub("[URL]", text)

        # 13. Strip / normalise email addresses
        if cfg.strip_emails:
            text = P.EMAIL_ADDR.sub("[EMAIL]", text)

        # 14. Strip / normalise phone numbers
        if cfg.strip_phone_numbers:
            text = P.PHONE.sub("[PHONE]", text)

        # 15. Strip / normalise CSX freight IDs
        if cfg.strip_csx_ids:
            text = P.EQUIPMENT_ID.sub("[EQUIPMENT_ID]", text)
            text = P.ORDER_NUMBER.sub("[ORDER_NUM]", text)
            text = P.WAYBILL.sub("[WAYBILL]", text)

        # 16. Custom extra patterns
        for pat in self._extra_patterns:
            text = pat.sub("", text)

        # 17. Whitespace normalisation
        if cfg.normalize_whitespace:
            text = P.MULTI_NEWLINE.sub("\n\n", text)
            text = P.MULTI_SPACE.sub(" ", text)
            text = text.strip()

        # 18. Optional lowercase
        if cfg.lowercase:
            text = text.lower()

        # 19. Truncate
        if cfg.max_length:
            text = self._truncate_words(text, cfg.max_length)

        return text

    def clean_batch(
        self,
        series: pd.Series,
        show_progress: bool = True,
    ) -> pd.Series:
        """
        Vectorised (apply-based) batch cleaner.

        Args:
            series: pd.Series of raw email strings.
            show_progress: if True, logs progress every 1000 rows.

        Returns:
            pd.Series of cleaned strings (same index as input).
        """
        total = len(series)
        results = []

        for i, text in enumerate(series):
            if show_progress and i % 1000 == 0:
                logger.info("EmailCleaner: %d / %d rows processed", i, total)
            results.append(self.clean(text))

        return pd.Series(results, index=series.index, name=series.name)

    # ── Private helpers ──────────────────────────────────────────────────────

    def _strip_forwarded(self, text: str) -> str:
        """Remove forwarded-message blocks up to config.max_chain_depth."""
        depth = 0
        while depth < self.config.max_chain_depth or self.config.max_chain_depth == 0:
            new_text = P.FORWARDED_BLOCK.sub("", text, count=1)
            if new_text == text:
                break
            text = new_text
            depth += 1
        return text

    def _strip_signature(self, text: str) -> str:
        """
        Heuristic signature stripper.
        Looks for common closing phrases and strips everything after them.
        Also honours the RFC 3676 "-- \\n" delimiter.
        """
        # RFC 3676 sig delimiter
        match = P.SIG_DELIMITER.search(text)
        if match:
            text = text[: match.start()].rstrip()
            return text

        # Closing phrase heuristic
        match = P.SIG_CLOSING.search(text)
        if match:
            text = text[: match.start()].rstrip()

        return text

    def _strip_disclaimer(self, text: str) -> str:
        """Remove trailing disclaimer / legal notice blocks."""
        match = P.DISCLAIMER.search(text)
        if match:
            text = text[: match.start()].rstrip()
        return text

    @staticmethod
    def _truncate_words(text: str, max_words: int) -> str:
        """Truncate to `max_words` whitespace-delimited tokens."""
        words = text.split()
        if len(words) > max_words:
            return " ".join(words[:max_words])
        return text


# ---------------------------------------------------------------------------
# Convenience factory functions
# ---------------------------------------------------------------------------

def get_default_cleaner() -> EmailCleaner:
    """Return a cleaner with all defaults (safe for DistilBERT input)."""
    return EmailCleaner()


def get_ner_friendly_cleaner() -> EmailCleaner:
    """
    Cleaner that preserves freight-specific IDs, emails, and phone numbers
    so downstream spaCy NER can extract them.
    """
    return EmailCleaner(
        CleanerConfig(
            strip_emails=False,
            strip_phone_numbers=False,
            strip_csx_ids=False,   # keep waybills / equipment IDs for NER
            strip_urls=True,
            lowercase=False,
        )
    )


def get_strict_cleaner() -> EmailCleaner:
    """
    Aggressive cleaner: strips everything — good for intent classification
    where ID tokens add noise rather than signal.
    """
    return EmailCleaner(
        CleanerConfig(
            strip_emails=True,
            strip_phone_numbers=True,
            strip_csx_ids=True,
            strip_urls=True,
            lowercase=False,
            max_length=256,
        )
    )


# ---------------------------------------------------------------------------
# CLI entry point for smoke-testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import textwrap

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    SAMPLE = textwrap.dedent("""
        From: shipper@acme.com
        To: support@csx.com
        Subject: Shipment delay — Car CSXT 123456
        Date: Mon, 10 Mar 2025 09:15:00 -0500

        Hi CSX team,

        Our car CSXT 123456 (Waybill 9876543) was supposed to arrive at
        Nashville yard on 3/8 but we still haven't received a delivery
        notification. The PO number is PO-2025-00812. Can you please check
        the status?

        Thanks,
        Mike Johnson
        Logistics Manager, Acme Corp
        mike.johnson@acme.com | (615) 555-0123

        --
        CONFIDENTIALITY NOTICE: This e-mail and any attachments are
        confidential and intended solely for the use of the individual
        named above. If you are not the intended recipient, please notify
        the sender immediately.

        ---------- Forwarded message ---------
        From: dispatcher@csx.com
        Date: Fri, 7 Mar 2025 at 16:00
        Subject: Re: Car CSXT 123456

        On Fri, Mar 7, 2025 at 3:45 PM Mike Johnson <mike.johnson@acme.com> wrote:
        > Can you confirm dispatch?
    """).strip()

    print("=" * 60)
    print("RAW INPUT:")
    print("=" * 60)
    print(SAMPLE)

    for label, cleaner in [
        ("DEFAULT", get_default_cleaner()),
        ("NER-FRIENDLY", get_ner_friendly_cleaner()),
        ("STRICT", get_strict_cleaner()),
    ]:
        print(f"\n{'=' * 60}")
        print(f"CLEANED ({label}):")
        print("=" * 60)
        result = cleaner.clean(SAMPLE)
        print(result if result else "<empty>")