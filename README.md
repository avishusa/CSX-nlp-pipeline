# Intelligent Logistics NLP Pipeline

Extracts **intent** and **entities** from unstructured customer interaction 
data — call transcripts, emails, and chatbot logs — using a cost-efficient 
hybrid NLP architecture.

## Problem
CSX captures thousands of customer interactions daily across multiple channels.
This data currently exists as unstructured text — impossible to analyze at scale.
This pipeline converts raw conversations into structured, BI-ready JSON records.

## Solution
A tiered NLP pipeline:
- **Intent Classifier** (DistilBERT) — classifies every message into 1 of 13 intents
- **Entity Extractor** (spaCy NER) — pulls structured fields (origin, destination, weight, tracking number etc.)
- **LLM Fallback** (GPT-4o-mini via Azure OpenAI) — only fires when classifier confidence < 85%

## Supported Intents
| Intent | Example |
|---|---|
| `get_rate_quote` | "How much to ship 2,000 lbs from Chicago to Atlanta?" |
| `track_shipment` | "Where is my shipment right now?" |
| `get_eta` | "When will my delivery arrive?" |
| `report_damage` | "My cargo arrived damaged" |
| `report_lost_shipment` | "My package never showed up" |
| `schedule_pickup` | "I need to schedule a pickup for Monday" |
| `cancel_shipment` | "I want to cancel my order" |
| `update_delivery_address` | "Can I change the drop-off location?" |
| `request_invoice` | "I need the invoice for BOL-29847" |
| `check_freight_class` | "What freight class are automotive parts?" |
| `file_claim` | "I want to file a formal claim" |
| `escalate_to_agent` | "Let me talk to a real person" |
| `general_inquiry` | "What are your operating hours?" |

## Tech Stack
- Python 3.13
- HuggingFace Transformers (DistilBERT)
- spaCy (custom NER)
- Azure CLU / Azure OpenAI
- Streamlit (demo UI)

## Project Structure
```
csx-nlp-pipeline/
├── data/               # Raw, processed, synthetic, and final datasets
├── notebooks/          # EDA and training notebooks
├── src/
│   ├── preprocessing/  # Source-specific text cleaners
│   ├── data/           # Dataset loading and relabeling
│   ├── models/         # Intent classifier and NER model
│   ├── pipeline/       # End-to-end inference pipeline
│   └── utils/          # Shared utilities
├── tests/              # Unit and integration tests
└── app/                # Streamlit demo
```

## Status
🚧 In active development
