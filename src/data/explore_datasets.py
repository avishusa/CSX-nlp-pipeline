"""
Step 5: Explore the downloaded datasets
- Look at what intents exist in Bitext and CLINC150
- Check class distribution
- Preview sample rows
- Identify which intents map to our CSX taxonomy
"""

import pandas as pd

# ── Load both CSVs ──────────────────────────────────────────
bitext = pd.read_csv("data/raw/bitext_raw.csv")
clinc  = pd.read_csv("data/raw/clinc150_raw.csv")


# ── 1. BITEXT — Intent list + row counts ───────────────────
print("=" * 60)
print("BITEXT — Intent distribution")
print("=" * 60)
bitext_counts = bitext["intent"].value_counts().sort_index()
for intent, count in bitext_counts.items():
    print(f"  {intent:<40} {count} rows")

print(f"\nTotal: {len(bitext)} rows across {bitext['intent'].nunique()} intents\n")


# ── 2. BITEXT — Sample rows for shipping-relevant intents ──
shipping_intents = [
    "track_order",
    "delivery_options",
    "delivery_period",
    "cancel_order",
    "complaint",
    "check_invoice",
    "get_invoice",
    "contact_human_agent",
]

print("=" * 60)
print("BITEXT — Sample rows for shipping-relevant intents")
print("=" * 60)
for intent in shipping_intents:
    subset = bitext[bitext["intent"] == intent]
    if len(subset) > 0:
        print(f"\n[{intent}] — {len(subset)} rows")
        for row in subset["text"].head(3):
            print(f"  → {row}")


# ── 3. CLINC150 — Domain breakdown ─────────────────────────
print("\n" + "=" * 60)
print("CLINC150 — All unique intents (sorted)")
print("=" * 60)
for intent in sorted(clinc["intent"].unique()):
    count = len(clinc[clinc["intent"] == intent])
    print(f"  {intent:<45} {count} rows")

print(f"\nTotal: {len(clinc)} rows across {clinc['intent'].nunique()} intents\n")


# ── 4. Text length analysis ─────────────────────────────────
print("=" * 60)
print("TEXT LENGTH ANALYSIS")
print("=" * 60)
bitext["text_len"] = bitext["text"].str.split().str.len()
clinc["text_len"]  = clinc["text"].str.split().str.len()

print(f"Bitext  — avg words per message: {bitext['text_len'].mean():.1f}")
print(f"Bitext  — min: {bitext['text_len'].min()}  max: {bitext['text_len'].max()}")
print(f"CLINC150 — avg words per message: {clinc['text_len'].mean():.1f}")
print(f"CLINC150 — min: {clinc['text_len'].min()}  max: {clinc['text_len'].max()}")


# ── 5. CSX Mapping Preview ──────────────────────────────────
print("\n" + "=" * 60)
print("CSX INTENT MAPPING — What we can use from Bitext")
print("=" * 60)

mapping = {
    "track_shipment"         : "track_order",
    "get_eta"                : "delivery_period",
    "cancel_shipment"        : "cancel_order",
    "file_claim / complaint" : "complaint",
    "request_invoice"        : ["check_invoice", "get_invoice"],
    "update_delivery_address": "change_shipping_address",
    "escalate_to_agent"      : "contact_human_agent",
    "get_rate_quote"         : "⚠️  NO MATCH — need synthetic data",
    "report_damage"          : "⚠️  NO MATCH — need synthetic data",
    "report_lost_shipment"   : "⚠️  NO MATCH — need synthetic data",
    "schedule_pickup"        : "⚠️  NO MATCH — need synthetic data",
    "check_freight_class"    : "⚠️  NO MATCH — need synthetic data",
    "general_inquiry"        : "⚠️  NO MATCH — use CLINC150 OOS",
}

for csx_intent, source in mapping.items():
    if isinstance(source, list):
        source = " + ".join(source)
    print(f"  {csx_intent:<30} ← {source}")