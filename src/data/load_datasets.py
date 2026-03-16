"""
Load open source datasets from HuggingFace
Datasets used:
  - Bitext customer support (shipping intents)
  - CLINC150 (broad intent classification + out-of-scope examples)
"""

import os
import pandas as pd
from datasets import load_dataset

# Where we save the raw data
RAW_DATA_DIR = "data/raw"


def load_bitext():
    """
    Loads the Bitext customer support dataset from HuggingFace.
    This dataset has real human phrasing for intents like:
    track_order, delivery_options, cancel_order, complaint etc.
    We will later rename these to match our CSX intent taxonomy.
    """
    print("Downloading Bitext customer support dataset...")

    dataset = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
    )

    # It comes as a DatasetDict - we want the train split as a dataframe
    df = dataset["train"].to_pandas()

    # We only need the text (instruction) and the intent label
    df = df[["instruction", "intent"]].copy()
    df.columns = ["text", "intent"]

    # Save to CSV
    output_path = os.path.join(RAW_DATA_DIR, "bitext_raw.csv")
    df.to_csv(output_path, index=False)

    print(f"Bitext saved: {len(df)} rows → {output_path}")
    print(f"Intents found: {sorted(df['intent'].unique())}\n")

    return df


def load_clinc150():
    """
    Loads the CLINC150 dataset from HuggingFace.
    This dataset has 150 intent classes across 10 domains.
    Most importantly it has an 'oos' (out of scope) class —
    examples of messages that don't fit any intent.
    We use these to teach our model what 'unknown' looks like.
    """
    print("Downloading CLINC150 dataset...")

    dataset = load_dataset("clinc/clinc_oos", "plus")

    # Combine train, validation and test splits into one dataframe
    splits = []
    for split_name in ["train", "validation", "test"]:
        split_df = dataset[split_name].to_pandas()
        splits.append(split_df)

    df = pd.concat(splits, ignore_index=True)

    # Rename columns to match our format
    df = df[["text", "intent"]].copy()

    # Save to CSV
    output_path = os.path.join(RAW_DATA_DIR, "clinc150_raw.csv")
    df.to_csv(output_path, index=False)

    print(f"CLINC150 saved: {len(df)} rows → {output_path}")
    print(f"Number of unique intents: {df['intent'].nunique()}\n")

    return df


def main():
    # Create raw data directory if it doesn't exist
    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    # Load both datasets
    bitext_df = load_bitext()
    clinc_df = load_clinc150()

    print("=" * 50)
    print("All datasets downloaded successfully.")
    print(f"Bitext:   {len(bitext_df)} rows")
    print(f"CLINC150: {len(clinc_df)} rows")
    print("=" * 50)


if __name__ == "__main__":
    main()