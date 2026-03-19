# SAVEY - AI Financial Assistant

A LangGraph-powered chatbot that helps users track expenses, manage budgets, and get financial insights with persistent memory.

## Features

- **Expense Tracking**: Parse natural language expense messages and calculate totals
- **Multi-Currency Support**: Real-time currency conversion via external API
- **Memory System**: Long-term and short-term memory stored in Firestore
- **Sub-Agents**: Specialized agents for duration parsing and currency conversion
- **Task Management**: Built-in TODO system for complex multi-step operations

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```
OPENAI_API_KEY=your_openai_key
GOOGLE_APPLICATION_CREDENTIALS=path/to/firestore-credentials.json
```

3. Configure Firestore:
   - Create a Firebase project named `savey-490713`
   - Create a Firestore database named `savey`
   - Download service account credentials

## Usage

Run the agent:
```bash
python saveyAgent.py
```

Example interaction:
```python
"I spent £10 on lunch and £5 on coffee today, and £20 on a dog food yesterday?"
```

## Architecture

- **State Management**: `state.py` - Defines graph state structure
- **Tools**: `tools.py` - Expense parsing, currency conversion, TODO management
- **Database**: `database.py` - Firestore operations for profiles and memories
- **Agent**: `saveyAgent.py` - Main LangGraph workflow with nodes for memory loading, reasoning, and summarization

## Graph Flow

```
START → load_memory → agent → [tools/summarize] → update_state → long_term_sync → END
```
