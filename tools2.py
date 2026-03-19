# savey_tools.py
import re
from datetime import date
from statistics import mode
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent
import requests
import json
import os
from dotenv import load_dotenv
from utils import format_messages


load_dotenv()


# Tool 1: Total Expenses
@tool
def retrieve_total_expenses(text: str) -> str:
    """
    Parse a natural language expense message and return the total amount spent.
    Only handles GBP (£) and USD ($) amounts.
    """
    matches = re.findall(r'[£$]{1}(\d+)', text)
    return sum(float(x) for x in matches)

# Tool 2: Most Purchased Item
@tool
def retrieve_purchased_item(text: str) -> str:
    """
    Parse a natural language expense message and return the most commonly purchased item.
    """
    matches = re.findall(r'[£$]{1}\d+\s?(\w+)[\s,.!?]?', text)
    return mode(matches) if matches else "Unknown"

# Tool 3: Get Today's Date
@tool
def get_today_date() -> str:
    """Returns today's date so relative references like 'yesterday' or 'last Monday' can be resolved accurately."""
    return date.today().strftime("%A, %d %B %Y")

# Duration Sub-Agent Setup
DURATION_SYSTEM_PROMPT = """You are a specialist in reading expense descriptions and identifying 
how many distinct days they span.
You have access to a tool that returns today's date — use it to resolve relative 
references like 'last Monday' or 'three days ago'.
Return ONLY a single integer — the number of days. Do not explain your answer."""

duration_model = init_chat_model(model="gpt-4o-mini", temperature=0.0)

duration_agent = create_react_agent(
    duration_model,
    tools=[get_today_date],
    prompt=DURATION_SYSTEM_PROMPT
)

# Tool 4: Duration Agent
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

@tool
def get_exchange_rate(currency: str) -> str:
    """
    Fetch the current exchange rate for a given currency to GBP.
    Example: get_exchange_rate("USD") returns the USD to GBP conversion rate.
    """
    try:
        url = f"https://currency-api-978796294307.europe-west1.run.app/convert/{currency.upper()}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        # Assuming API returns something like {"currency": "USD", "rate": 0.79}
        return json.dumps(data)
    except Exception as e:
        return f"Error fetching exchange rate: {str(e)}"

@tool
def calculate_conversion(amount: float, rate: float) -> str:
    """
    Calculate the converted amount given an amount and exchange rate.
    Example: calculate_conversion(100, 0.79) returns 79.0 (100 USD = 79 GBP)
    """
    return str(amount * rate)

# Currency Sub-Agent Setup
CURRENCY_SYSTEM_PROMPT = """You are a currency conversion specialist.
You have access to tools to:
1. Fetch live exchange rates for any currency to GBP
2. Calculate conversions

When given an amount and currency:
1. First get the exchange rate using get_exchange_rate
2. Then calculate the conversion using calculate_conversion
3. Return ONLY the final GBP amount as a number (e.g., "79.0")

Do not explain your reasoning, just return the converted amount."""

currency_model = init_chat_model(model="gpt-4o-mini", temperature=0.0)

currency_agent = create_react_agent(
    currency_model,
    tools=[get_exchange_rate, calculate_conversion],
    prompt=CURRENCY_SYSTEM_PROMPT
)

@tool
def ask_currency_agent(amount: float, currency: str) -> str:
    """
    Convert an amount from any currency to GBP using live exchange rates.
    Use this whenever the user mentions non-GBP/USD currencies (EUR, JPY, etc.)
    Example: ask_currency_agent(100, "EUR") returns the GBP equivalent
    """
    prompt = f"Convert {amount} {currency.upper()} to GBP"
    result = currency_agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]}
    )
    return result["messages"][-1].content

# Initialize the TODO state dictionary at module level
_todo_state = {"steps": [], "status": []}

@tool
def manage_todo(action: str, steps: list[str] = None, step_index: int = None, new_status: str = None) -> str:
    """
    Manage Savey's internal TODO list for complex, multi-step tasks.
    Use this tool when a user request requires multiple tool calls or careful sequencing.
    For simple single-step requests, do NOT use this tool — just answer directly.
    
    Actions:
      - "create": Initialise a new TODO list.
                  Requires: steps (list of strings describing each task).
                  Example: manage_todo(action="create", steps=["Calculate total expenses", "Find most purchased item", "Count days"])
      - "read":   Return the current TODO list with statuses.
                  No extra arguments needed.
                  Returns a formatted string showing each step and its status.
      - "update": Change the status of a specific step.
                  Requires: step_index (0-based int), new_status (one of: "pending", "in_progress", "done").
                  Example: manage_todo(action="update", step_index=0, new_status="done")
    
    Workflow for complex tasks:
      1. Call manage_todo(action="create", steps=[...]) to write your plan.
      2. Call manage_todo(action="read") to see what's next.
      3. Execute the next pending step using the appropriate tool.
      4. Call manage_todo(action="update", step_index=N, new_status="done") to tick it off.
      5. Repeat from step 2 until all steps are done.
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
            return f"Error: step_index {step_index} is out of range. Valid range: 0–{len(_todo_state['steps']) - 1}."
        if new_status not in status_icons:
            return f"Error: new_status must be one of {list(status_icons.keys())}."
        _todo_state["status"][step_index] = new_status
        icon = status_icons[new_status]
        return f"Updated step {step_index} ('{_todo_state['steps'][step_index]}') → {icon} {new_status}."
    
    else:
        return f"Error: unknown action '{action}'. Use 'create', 'read', or 'update'."

TOOLS = [
    retrieve_total_expenses, 
    retrieve_purchased_item, 
    get_today_date, 
    manage_todo,           # Make sure this tool is defined in the file!
    ask_currency_agent     # And this one if you are using the sub-agent
]