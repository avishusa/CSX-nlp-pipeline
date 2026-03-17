"""
Step 7: Generate synthetic training data for the 5 CSX intents
that have no open source equivalent.

Intents to generate:
  - get_rate_quote
  - report_damage
  - report_lost_shipment
  - schedule_pickup
  - check_freight_class

We use Claude API ONCE here offline to generate examples.
After this, no LLM is used at runtime — ever.
"""

import os
import json
import time
import pandas as pd
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

SYNTHETIC_DIR = "data/synthetic"
client = Anthropic()

# ── Intent definitions with context for the LLM ─────────────
INTENTS_TO_GENERATE = {
    "get_rate_quote": {
        "description": "Customer wants to know the cost of shipping freight",
        "entities": "origin city, destination city, weight, cargo type",
        "examples": [
            "How much would it cost to ship 5000 lbs from Chicago to Miami?",
            "Can I get a quote for moving steel coils from Houston to Detroit?",
            "What's the rate for shipping automotive parts from Atlanta to Buffalo?"
        ]
    },
    "report_damage": {
        "description": "Customer is reporting that their shipment arrived damaged",
        "entities": "tracking number, cargo type, damage description",
        "examples": [
            "My shipment arrived and several boxes are crushed",
            "The cargo we received has visible damage to the packaging",
            "Our delivery came in today and the goods are damaged"
        ]
    },
    "report_lost_shipment": {
        "description": "Customer is reporting that their shipment never arrived or is missing",
        "entities": "tracking number, expected delivery date, origin, destination",
        "examples": [
            "My shipment was supposed to arrive 3 days ago and I still haven't received it",
            "We can't locate our freight, it's been over a week",
            "Our cargo seems to be missing, tracking shows no updates"
        ]
    },
    "schedule_pickup": {
        "description": "Customer wants to arrange a pickup of freight from their location",
        "entities": "pickup address, date, cargo type, weight",
        "examples": [
            "I need to schedule a pickup from our warehouse in Dallas on Monday",
            "Can you arrange to pick up a pallet of goods from our facility?",
            "We need a freight pickup from our loading dock this Friday"
        ]
    },
    "check_freight_class": {
        "description": "Customer wants to know the freight classification for their cargo",
        "entities": "cargo type, weight, dimensions",
        "examples": [
            "What freight class would automotive parts fall under?",
            "Can you tell me the classification for shipping machinery equipment?",
            "What class should I use for pallets of consumer electronics?"
        ]
    }
}

ROWS_PER_INTENT = 200


def generate_examples_for_intent(intent_name, intent_info):
    """
    Calls Claude API once to generate synthetic training examples
    for a single intent in 3 different styles:
    - formal email
    - casual chat
    - spoken transcript style
    """
    print(f"\nGenerating examples for: {intent_name}")

    prompt = f"""You are generating training data for an NLP intent classifier 
for CSX, a freight railroad company.

Generate exactly {ROWS_PER_INTENT} diverse customer messages that express 
the intent: "{intent_name}"

Intent description: {intent_info['description']}
Key entities often mentioned: {intent_info['entities']}

Seed examples to guide style variety:
{chr(10).join(f'- {e}' for e in intent_info['examples'])}

Rules:
1. Mix 3 styles equally (~67 each):
   - Formal email style: professional, complete sentences
   - Casual chat style: short, informal, typos ok, abbreviations ok
   - Spoken transcript style: conversational, filler words ok (uh, um), incomplete sentences ok
2. Vary the specific details (different cities, weights, cargo types, dates)
3. Use realistic freight/rail terminology: PRO number, BOL, intermodal, 
   demurrage, freight class, LTL, FTL, consignee, shipper
4. Each message should be on its own line
5. Output ONLY the messages, one per line, no numbering, no labels

Generate {ROWS_PER_INTENT} messages now:"""

    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )

    # Parse the response — one message per line
    raw_text = message.content[0].text
    lines = [line.strip() for line in raw_text.strip().split("\n")
             if line.strip() and len(line.strip()) > 10]

    print(f"  Generated {len(lines)} examples")

    # Small delay to avoid rate limiting
    time.sleep(2)

    return lines


def main():
    os.makedirs(SYNTHETIC_DIR, exist_ok=True)

    all_rows = []

    for intent_name, intent_info in INTENTS_TO_GENERATE.items():
        examples = generate_examples_for_intent(intent_name, intent_info)

        for text in examples:
            all_rows.append({
                "text"  : text,
                "intent": intent_name,
                "source": "synthetic"
            })

    # Save to CSV
    df = pd.DataFrame(all_rows)
    output_path = os.path.join(SYNTHETIC_DIR, "synthetic_data.csv")
    df.to_csv(output_path, index=False)

    # Summary
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA GENERATION COMPLETE")
    print("=" * 60)
    print(f"Total rows generated: {len(df)}")
    print(f"Saved to: {output_path}")
    print("\nBreakdown:")
    for intent, count in df["intent"].value_counts().sort_index().items():
        print(f"  {intent:<30} {count} rows")


if __name__ == "__main__":
    main()