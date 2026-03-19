# Python file to be imported in the Savey Challenge to provide agent tools

#
from locale import currency
import re
import json
from datetime import date
from statistics import mode
from typing import TypedDict

# LangGraph — new in v3
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent

# 
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage

# update for the live converter api:
import requests

@tool
def retrieve_total_expenses(text: str) -> str:
    """
    Parse a natural language expense message and return the total amount spent.
    Only handles GBP (£) and USD ($) amounts. Use convert_to_gbp first for other currencies.
    Example input: 'Today I bought a £4 coffee and a £8 sandwich.'
    """
    matches = re.findall(r'[£$]{1}(\d+)', text)
    return sum(float(x) for x in matches)

@tool
def retrieve_purchased_item(text: str) -> str:
    """
    Parse a natural language expense message and return the most commonly purchased item.
    Example input: 'Today I bought a £4 coffee and a £8 sandwich. Yesterday I only got a £4 coffee.'
    """
    matches = re.findall(r'[£$]{1}\d+\s?(\w+)[\s,.!?]?', text)
    return mode(matches)


@tool
def get_today_date() -> str:
    """Returns today's date so relative references like 'yesterday' or 'last Monday' can be resolved accurately."""
    return date.today().strftime("%A, %d %B %Y")

DURATION_SYSTEM_PROMPT = """You are a specialist in reading expense descriptions and identifying 
how many distinct days they span.
You have access to a tool that returns today's date — use it to resolve relative 
references like 'last Monday' or 'three days ago'.
Return ONLY a single integer — the number of days. Do not explain your answer."""

duration_model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)

duration_agent = create_react_agent(
    duration_model,
    tools=[get_today_date],
    prompt=DURATION_SYSTEM_PROMPT
)

@tool
def ask_duration_agent(text: str) -> str:
    """
    Delegate to the duration sub-agent to determine how many distinct days
    a natural language expense description spans.
    Use this whenever the user asks about duration, number of days, or time period.
    Example input:  "Today I bought a £4 coffee. Yesterday I got a £8 sandwich."
    Example output: 2
    """
    result = duration_agent.invoke(
        {"messages": [{"role": "user", "content": text}]}
    )
    return result["messages"][-1].content


BASE_URL = "https://api.frankfurter.app"

def _fetch_rate(currency: str, on_date: date = None) -> float:
    """
    Fetch exchange rate to GBP from Frankfurter API.
    Pass on_date for historic rates, or leave None for today's rate.
    """
    endpoint = on_date.strftime("%Y-%m-%d") if on_date else "latest"
    response = requests.get(
        f"{BASE_URL}/{endpoint}",
        params={"from": currency.upper(), "to": "GBP"}
    )
    if response.status_code == 404:
        raise ValueError(f"No data for {currency} on {on_date}")
    response.raise_for_status()
    data = response.json()
    return data["rates"]["GBP"]

@tool
def convert_to_gbp(amount: float, currency: str, on_date: str = None) -> str:
    """
    Convert a foreign currency amount to GBP.
    Use this whenever an expense is not already in GBP (£).
    Supported currencies: USD, EUR, JPY, CAD, AUD, CHF.

    on_date: optional date string in YYYY-MM-DD format for historic rates.
             Leave blank for today's rate.
             Example: convert_to_gbp(20.0, 'USD', on_date='2025-03-10')
    """
    try:
        parsed_date = date.fromisoformat(on_date) if on_date else None
        rate = _fetch_rate(currency, on_date=parsed_date)
        converted = round(amount * rate, 2)
        date_label = f" on {on_date}" if on_date else " (today's rate)"
        return f"{amount} {currency.upper()} = £{converted} GBP{date_label}"
    except ValueError as e:
        return f"Could not convert: {e}"
    

# Implement TODO List
# Module-level TODO state — persists for the lifetime of a single agent run
_todo_state = {"steps": [], "status": []}

def _reset_todo():
    """Helper to clear state between runs."""
    _todo_state["steps"] = []
    _todo_state["status"] = []

@tool
def manage_todo(action: str, steps: list[str] = None, step_index: int = None, new_status: str = None) -> str:
    """
    Manage Savey's internal TODO list for complex, multi-step tasks.

    Use this tool when a user request requires multiple tool calls or careful sequencing.
    For simple single-step requests, do NOT use this tool — just answer directly.

    Actions:
      - "create": Initialise a new TODO list.
                  Requires: steps (list of strings describing each task).
      - "read":   Return the current TODO list with statuses.
      - "update": Change the status of a specific step.
                  Requires: step_index (0-based int), new_status (one of: "pending", "in_progress", "done").

    Workflow for complex tasks:
      1. Call manage_todo(action="create", steps=[...]) to write your plan.
      2. Call manage_todo(action="read") to see what's next.
      3. Mark the next step in_progress, execute it, then mark it done.
      4. Repeat until all steps are complete.
    """
    status_icons = {"pending": "⬜", "in_progress": "🔄", "done": "✅"}

    if action == "create":
        if not steps:
            return "Error: 'create' action requires a non-empty list of steps."
        _todo_state["steps"] = list(steps)
        _todo_state["status"] = ["pending"] * len(steps)
        lines = [f"📋 TODO list created with {len(steps)} steps:"]
        for i, step in enumerate(steps):
            lines.append(f"  {i}. {status_icons['pending']} {step}")
        return "\n".join(lines)

    elif action == "read":
        if not _todo_state["steps"]:
            return "No TODO list exists yet. Use action='create' to start one."
        lines = ["📋 Current TODO list:"]
        for i, (step, status) in enumerate(zip(_todo_state["steps"], _todo_state["status"])):
            lines.append(f"  {i}. {status_icons[status]} {step}")
        pending = [i for i, s in enumerate(_todo_state["status"]) if s == "pending"]
        if pending:
            lines.append(f"\n➡️  Next up: step {pending[0]} — '{_todo_state['steps'][pending[0]]}'")
        else:
            lines.append("\n🎉 All steps complete!")
        return "\n".join(lines)

    elif action == "update":
        if step_index is None or new_status is None:
            return "Error: 'update' action requires both step_index and new_status."
        if step_index < 0 or step_index >= len(_todo_state["steps"]):
            return f"Error: step_index {step_index} is out of range."
        if new_status not in status_icons:
            return f"Error: new_status must be one of {list(status_icons.keys())}."
        _todo_state["status"][step_index] = new_status
        icon = status_icons[new_status]
        return f"Updated step {step_index} ('{_todo_state['steps'][step_index]}') → {icon} {new_status}."

    else:
        return f"Error: unknown action '{action}'. Use 'create', 'read', or 'update'."
    


CONVERTER_SYSTEM_PROMPT = """You are a specialist in converting foreign currency expenses to GBP.

You will receive a natural language description of expenses that may span multiple dates
and contain multiple currencies.

Your job:
1. Use get_today_date to resolve any relative date references (e.g. "yesterday", "last Monday") to real YYYY-MM-DD dates
2. For each foreign currency expense, call convert_to_gbp with the correct amount, currency, and date
3. Return a JSON array summarising each conversion, in this exact format:
   [
     {"original": "20.0 USD", "gbp": 15.80, "date": "2025-03-18"},
     {"original": "50.0 EUR", "gbp": 42.50, "date": "2025-03-17"}
   ]

Return ONLY the JSON array. No explanation, no markdown, no code fences."""

converter_model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)

converter_agent = create_react_agent(
    converter_model,
    tools=[get_today_date, convert_to_gbp],
    prompt=CONVERTER_SYSTEM_PROMPT
)

@tool
def ask_converter_agent(text: str) -> str:
    """
    Delegate to the converter sub-agent to handle foreign currency expenses.
    Use this whenever the user mentions expenses in non-GBP currencies.
    The agent will resolve dates and apply the correct historic exchange rate for each expense.
    Example input:  "Yesterday I spent $20 on dinner. Last Monday I spent €50 on a hotel."
    Example output: [{"original": "20.0 USD", "gbp": 15.80, "date": "2025-03-18"}, ...]
    """
    result = converter_agent.invoke(
        {"messages": [{"role": "user", "content": text}]}
    )
    return result["messages"][-1].content