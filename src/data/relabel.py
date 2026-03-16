"""
Step 6: Relabel and filter open source datasets to match CSX intent taxonomy.
- Bitext intents get renamed to CSX intent names
- Irrelevant intents get dropped
- CLINC150 out-of-scope examples kept for 'general_inquiry'
- Output saved to data/processed/
"""

import os
import pandas as pd

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"

# ── Bitext → CSX intent mapping ─────────────────────────────
# Left side  = Bitext original label
# Right side = Our CSX label
BITEXT_MAPPING = {
    "track_order"             : "track_shipment",
    "delivery_period"         : "get_eta",
    "cancel_order"            : "cancel_shipment",
    "complaint"               : "file_claim",
    "check_invoice"           : "request_invoice",
    "get_invoice"             : "request_invoice",
    "change_shipping_address" : "update_delivery_address",
    "contact_human_agent"     : "escalate_to_agent",
}


def relabel_bitext():
    """
    Load Bitext, keep only relevant intents, rename to CSX labels.
    """
    print("Processing Bitext dataset...")

    df = pd.read_csv(os.path.join(RAW_DIR, "bitext_raw.csv"))
    print(f"  Original: {len(df)} rows, {df['intent'].nunique()} intents")

    # Keep only rows whose intent is in our mapping
    df = df[df["intent"].isin(BITEXT_MAPPING.keys())].copy()
    print(f"  After filtering: {len(df)} rows")

    # Rename intents to CSX labels
    df["intent"] = df["intent"].map(BITEXT_MAPPING)

    # Add a column to track where this data came from
    df["source"] = "bitext"

    print(f"  Final intents: {sorted(df['intent'].unique())}")
    return df


def extract_clinc_oos():
    """
    From CLINC150, extract out-of-scope (OOS) examples.
    These are messages that don't fit any supported intent —
    we use them to represent 'general_inquiry' in our taxonomy.
    In CLINC150, the OOS intent has label value 42.
    """
    print("\nProcessing CLINC150 dataset...")

    df = pd.read_csv(os.path.join(RAW_DIR, "clinc150_raw.csv"))
    print(f"  Original: {len(df)} rows")

    # OOS intent is labeled as 42 in CLINC150
    oos_df = df[df["intent"] == 42].copy()
    print(f"  OOS examples found: {len(oos_df)} rows")

    # Rename to our CSX label
    oos_df["intent"] = "general_inquiry"
    oos_df["source"] = "clinc150"

    return oos_df


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Process both sources
    bitext_df = relabel_bitext()
    clinc_df  = extract_clinc_oos()

    # Combine into one dataframe
    combined = pd.concat([bitext_df, clinc_df], ignore_index=True)

    # Keep only the columns we need
    combined = combined[["text", "intent", "source"]]

    # Shuffle the rows so intents are mixed
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save
    output_path = os.path.join(PROCESSED_DIR, "relabeled_data.csv")
    combined.to_csv(output_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("RELABELING COMPLETE")
    print("=" * 60)
    print(f"Total rows: {len(combined)}")
    print(f"Saved to: {output_path}")
    print("\nClass distribution:")
    counts = combined["intent"].value_counts().sort_index()
    for intent, count in counts.items():
        bar = "█" * (count // 100)
        print(f"  {intent:<30} {count:>5} rows  {bar}")

    print(f"\n⚠️  Intents still missing (need synthetic data):")
    all_intents = [
        "get_rate_quote", "report_damage", "report_lost_shipment",
        "schedule_pickup", "check_freight_class"
    ]
    for intent in all_intents:
        print(f"  → {intent}")


if __name__ == "__main__":
    main()