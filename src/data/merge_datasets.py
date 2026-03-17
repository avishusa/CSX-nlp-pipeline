"""
Step 8: Merge relabeled open source data and synthetic data
into one final training CSV ready for model training.

Inputs:
  - data/processed/relabeled_data.csv   (9,313 rows)
  - data/synthetic/synthetic_data.csv   (901 rows)

Output:
  - data/final/training_data.csv
"""

import os
import pandas as pd

PROCESSED_DIR = "data/processed"
SYNTHETIC_DIR = "data/synthetic"
FINAL_DIR     = "data/final"

# All 13 CSX intents — every one must be present in final data
ALL_INTENTS = [
    "get_rate_quote",
    "track_shipment",
    "get_eta",
    "report_damage",
    "report_lost_shipment",
    "schedule_pickup",
    "cancel_shipment",
    "update_delivery_address",
    "request_invoice",
    "check_freight_class",
    "file_claim",
    "escalate_to_agent",
    "general_inquiry",
]


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)

    # ── Load both sources ────────────────────────────────────
    relabeled  = pd.read_csv(os.path.join(PROCESSED_DIR, "relabeled_data.csv"))
    synthetic  = pd.read_csv(os.path.join(SYNTHETIC_DIR, "synthetic_data.csv"))

    print(f"Relabeled data:  {len(relabeled):>6} rows")
    print(f"Synthetic data:  {len(synthetic):>6} rows")

    # ── Combine ──────────────────────────────────────────────
    combined = pd.concat([relabeled, synthetic], ignore_index=True)
    print(f"Combined total:  {len(combined):>6} rows")

    # ── Clean up ─────────────────────────────────────────────
    # Drop any rows with missing text or intent
    combined = combined.dropna(subset=["text", "intent"])

    # Strip whitespace from text
    combined["text"] = combined["text"].str.strip()

    # Drop any empty texts
    combined = combined[combined["text"].str.len() > 0]

    # Drop duplicates
    before_dedup = len(combined)
    combined = combined.drop_duplicates(subset=["text"])
    after_dedup = len(combined)
    print(f"Duplicates removed: {before_dedup - after_dedup}")

    # ── Shuffle ──────────────────────────────────────────────
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── Verify all 13 intents are present ────────────────────
    print("\n" + "=" * 60)
    print("FINAL CLASS DISTRIBUTION")
    print("=" * 60)

    missing_intents = []
    for intent in ALL_INTENTS:
        count = len(combined[combined["intent"] == intent])
        bar   = "█" * (count // 50)
        print(f"  {intent:<30} {count:>5} rows  {bar}")
        if count == 0:
            missing_intents.append(intent)

    if missing_intents:
        print(f"\n❌ WARNING — These intents have 0 rows: {missing_intents}")
    else:
        print(f"\n✅ All 13 intents present")

    # ── Save ─────────────────────────────────────────────────
    output_path = os.path.join(FINAL_DIR, "training_data.csv")
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f"Total rows in final dataset : {len(combined)}")
    print(f"Total intents               : {combined['intent'].nunique()}")
    print(f"Sources present             : {sorted(combined['source'].unique())}")
    print(f"Saved to                    : {output_path}")
    print("=" * 60)
    print("\n✅ Step 8 complete — data is ready for model training")


if __name__ == "__main__":
    main()