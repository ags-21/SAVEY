# Savey 💾 — Expense Tracking Chat Assistant

Savey is a conversational expense tracker built with LangGraph and served through a Chainlit chat UI. It can parse expenses from natural language, convert foreign currencies to GBP, count how many days your spending spans, and identify your most-bought items.

## Prerequisites

- Python 3.10+
- An OpenAI API key (or OpenRouter key with an OpenAI-compatible base URL)

## Setup

1. Clone the repo and install dependencies:

```bash
pip install -r requirements.txt
```

2. Copy the example env file and fill in your keys:

```bash
cp .env-example .env
```

At minimum you need `OPENAI_API_KEY` set. If you're using OpenRouter, also set `OPENAI_API_BASE`. The LangSmith keys are optional (for tracing).

3. Run the app:

```bash
chainlit run app.py
```

This starts a local server (default `http://localhost:8000`) with the Savey chat interface.

## What it does

- Tracks expenses from plain English messages (e.g. "I spent £4 on coffee and £8 on lunch")
- Converts foreign currencies (USD, EUR, JPY, etc.) to GBP using live exchange rates
- Counts how many days your expenses span
- Identifies your most frequently purchased item
- Handles complex multi-step queries by planning and executing steps in order
