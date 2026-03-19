import os
import re
import requests
from datetime import date
from statistics import mode
from typing import TypedDict, Annotated

from dotenv import load_dotenv
from utils import format_messages

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode, create_react_agent

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

from IPython.display import Image, display
import ipywidgets as widgets

from langsmith import traceable

load_dotenv()


# =============================================================================
# Tools
# =============================================================================

@tool
def retrieve_total_expenses(text: str) -> str:
    """
    Parse a natural language expense message and return the total amount spent.
    Only handles GBP (£) and USD ($) amounts. Use convert_to_gbp first for other currencies.
    Example input: 'Today I bought a £4 coffee and a £8 sandwich.'
    """
    matches = re.findall(r"[£$]{1}(\d+(?:\.\d+)?)", text)
    return str(sum(float(x) for x in matches))


@tool
def retrieve_purchased_item(text: str) -> str:
    """
    Parse a natural language expense message and return the most commonly purchased item.
    Example input: 'Today I bought a £4 coffee and a £8 sandwich. Yesterday I only got a £4 coffee.'
    """
    matches = re.findall(r"[£$]{1}\d+(?:\.\d+)?\s?(\w+)[\s,.!?]?", text)
    return mode(matches) if matches else "unknown"


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
    Example input: "Today I bought a £4 coffee. Yesterday I got a £8 sandwich."
    Example output: 2
    """
    result = duration_agent.invoke(
        {"messages": [{"role": "user", "content": text}]}
    )
    return result["messages"][-1].content


@tool
def check_db() -> str:
    """Get the conversation history from the database."""
    return "No external database connected."


def _fetch_rate(currency: str) -> float:
    """Fetch a conversion rate. Raises ValueError for unrecognised currency codes."""
    url = "https://currency-api-978796294307.europe-west1.run.app/"
    response = requests.get(url)
    available_currency = response.json()["available_currencies"]

    exchange_rates = {}
    for curr in available_currency:
        exchange_rates[curr] = requests.get(f"{url}convert/{curr}").json()["rate"]

    if currency.upper() not in exchange_rates:
        raise ValueError(f"Unrecognised currency code: {currency}")

    return exchange_rates[currency.upper()]


@tool
def convert_to_gbp(amount: float, currency: str) -> str:
    """
    Convert a foreign currency amount to GBP.
    Use this whenever an expense is not already in GBP (£).
    Supported currencies: USD, EUR, JPY, CAD, AUD, CHF.
    Example: convert_to_gbp(20.0, 'USD') → '20.0 USD = £15.8 GBP'
    """
    try:
        rate = _fetch_rate(currency)
        converted = round(amount * rate, 2)
        return f"{amount} {currency.upper()} = £{converted} GBP"
    except ValueError as e:
        return f"Could not convert: {e}. Please use one of: {', '.join(MOCK_EXCHANGE_RATES.keys())}"


@tool
def set_should_summarize(value: bool) -> str:
    """
    Set whether Savey should generate a spending summary/advice after state is updated.
    Call this with:
    - True if the user is asking for a summary, spending advice, or money-saving tips
    - False otherwise
    """
    return str(value)


# Simple global store for the TODO tool
_todo_state_store = {"steps": [], "status": []}


@tool
def manage_todo(action: str, steps: list[str] = None, step_index: int = None, new_status: str = None) -> str:
    """
    Manage Savey's internal TODO list for complex, multi-step tasks.

    Use this tool when a user request requires multiple tool calls or careful sequencing.
    For simple single-step requests, do NOT use this tool — just answer directly.
    """
    status_icons = {"pending": "⬜", "in_progress": "🔄", "done": "✅"}

    if action == "create":
        if not steps:
            return "Error: 'create' action requires a non-empty list of steps."
        _todo_state_store["steps"] = list(steps)
        _todo_state_store["status"] = ["pending"] * len(steps)
        lines = [f"📋 TODO list created with {len(steps)} steps:"]
        for i, step in enumerate(steps):
            lines.append(f"  {i}. {status_icons['pending']} {step}")
        return "\n".join(lines)

    elif action == "read":
        if not _todo_state_store["steps"]:
            return "No TODO list exists yet. Use action='create' to start one."
        lines = ["📋 Current TODO list:"]
        for i, (step, status) in enumerate(zip(_todo_state_store["steps"], _todo_state_store["status"])):
            lines.append(f"  {i}. {status_icons[status]} {step}")
        pending = [i for i, s in enumerate(_todo_state_store["status"]) if s == "pending"]
        if pending:
            lines.append(f"\n➡️  Next up: step {pending[0]} — '{_todo_state_store['steps'][pending[0]]}'")
        else:
            lines.append("\n🎉 All steps complete!")
        return "\n".join(lines)

    elif action == "update":
        if step_index is None or new_status is None:
            return "Error: 'update' action requires both step_index and new_status."
        if step_index < 0 or step_index >= len(_todo_state_store["steps"]):
            return f"Error: step_index {step_index} is out of range. Valid range: 0–{len(_todo_state_store['steps']) - 1}."
        if new_status not in status_icons:
            return f"Error: new_status must be one of {list(status_icons.keys())}."
        _todo_state_store["status"][step_index] = new_status
        icon = status_icons[new_status]
        return f"Updated step {step_index} ('{_todo_state_store['steps'][step_index]}') → {icon} {new_status}."

    else:
        return f"Error: unknown action '{action}'. Use 'create', 'read', or 'update'."


# =============================================================================
# State
# =============================================================================

class SaveyState(TypedDict):
    messages: Annotated[list, add_messages]
    expense_log: list
    total_spent: float
    days_tracked: int
    _todo_state: list[str]
    should_summarize: bool


# =============================================================================
# Main agent
# =============================================================================

savey_tools = [
    retrieve_total_expenses,
    retrieve_purchased_item,
    ask_duration_agent,
    convert_to_gbp,
    manage_todo,
    set_should_summarize,
]

SYSTEM_PROMPT = """You are Savey 💾 — an expert at tracking and advising on day-to-day expenses.

You have access to the following tools:
- retrieve_total_expenses: for summing up GBP/USD amounts from text
- retrieve_purchased_item: for identifying the most frequently bought item
- ask_duration_agent: for determining how many days a description spans — ALWAYS use this for duration questions
- convert_to_gbp: for converting foreign currency amounts — use this BEFORE retrieve_total_expenses if the currency is not GBP
- manage_todo: for complex multi-step tasks
- set_should_summarize: use this every turn to say whether advice/summary should be generated after state update

---------------

- To use the day counter tool, you only need to pass the original query.
- To use the retrieve total expenses and retrieve purchased item tools, you should turn the user prompt into a
list of prices followed by items.

Example 1:
User prompt: I bought a £15 steak, a bottle of wine for £9, another £15 steak, and some 50p cheese.
Input for tools: £15 steak, £9 wine, £15 steak, £0.50 cheese.

Example 2:
User prompt: Today I've bought three packets of ham for £15 each and a packet of crisps for £1.
Input for tools: £15 ham, £15 ham, £15 ham, £1 crisps

-------------

Always convert foreign currencies to GBP before calculating totals.
For duration questions, always delegate to ask_duration_agent.

Every turn, call set_should_summarize exactly once:
- True if the user asks for a summary, overall spending summary, advice, or money-saving tips
- False otherwise
"""

model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)
model_with_tools = model.bind_tools(savey_tools)


@traceable
def agent_node(state: SaveyState) -> dict:
    system_message = SystemMessage(content=f"""{SYSTEM_PROMPT}

Current state:
- Total spent so far: £{state.get('total_spent', 0.0)}
- Days tracked: {state.get('days_tracked', 0)}
- Expense log: {state.get('expense_log', [])}
- Current To-Do List: {state.get('_todo_state', [])}

If the user asks for a summary or total, read directly from the state above — do not call any tools to recalculate.
""")

    response = model_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(savey_tools)


@traceable
def update_state_node(state: SaveyState) -> dict:
    expense_log = list(state.get("expense_log", []))
    days_tracked = state.get("days_tracked", 0)
    todo_state = list(state.get("_todo_state", []))
    should_summarize = state.get("should_summarize", False)

    messages = state["messages"]
    last_ai_idx = next(
        (
            i for i in reversed(range(len(messages)))
            if isinstance(messages[i], AIMessage) and messages[i].tool_calls
        ),
        None
    )

    if last_ai_idx is None:
        return {}

    current_tool_messages = [
        m for m in messages[last_ai_idx + 1:]
        if isinstance(m, ToolMessage)
    ]

    converted_this_round = any(m.name == "convert_to_gbp" for m in current_tool_messages)

    for msg in current_tool_messages:
        content = str(msg.content).strip()

        if msg.name == "set_should_summarize":
            should_summarize = content.lower() == "true"

        elif msg.name == "manage_todo":
            todo_state = list(_todo_state_store["steps"])

        elif msg.name == "convert_to_gbp":
            fx_match = re.search(r"= £([\d.]+) GBP", content)
            if fx_match:
                expense_log.append({
                    "item": "(foreign currency expense)",
                    "amount_gbp": float(fx_match.group(1))
                })

        elif msg.name == "retrieve_total_expenses" and not converted_this_round:
            amount_gbp = float(content)
            if amount_gbp > 0:
                expense_log.append({
                    "item": "(expenses from message)",
                    "amount_gbp": amount_gbp
                })

        elif msg.name == "ask_duration_agent":
            try:
                days_tracked += int(content)
            except Exception:
                print("Could not retrieve duration")

    total_spent = round(sum(e["amount_gbp"] for e in expense_log), 2)

    return {
        "expense_log": expense_log,
        "total_spent": total_spent,
        "days_tracked": days_tracked,
        "_todo_state": todo_state,
        "should_summarize": should_summarize,
    }


@traceable
def should_continue(state: SaveyState) -> str:
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# =============================================================================
# Advisor
# =============================================================================

def get_last_n_human_ai_messages(messages, n=6) -> str:
    filtered = [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    recent = filtered[-n:]
    lines = []

    for m in recent:
        if isinstance(m, HumanMessage):
            lines.append(f"User: {m.content}")
        elif isinstance(m, AIMessage) and getattr(m, "content", None):
            lines.append(f"Assistant: {m.content}")

    return "\n".join(lines)


ADVISOR_SYSTEM_PROMPT = """You are Savey's savings advisor.

You will be given:
1. The user's current spending state
2. The recent conversation history

Your job:
- Give a short summary of the user's spending so far
- Give practical, specific advice on how they could save money
- Base your advice on actual spending patterns when possible
- Keep it concise and useful
"""

advisor_model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)


@traceable
def advisor_node(state: SaveyState) -> dict:
    recent_context = get_last_n_human_ai_messages(state["messages"], n=6)

    advisor_input = f"""
Current spending state:
- Total spent: £{state.get('total_spent', 0.0)}
- Days tracked: {state.get('days_tracked', 0)}
- Expense log: {state.get('expense_log', [])}

Recent conversation:
{recent_context}
"""

    response = advisor_model.invoke([
        SystemMessage(content=ADVISOR_SYSTEM_PROMPT),
        HumanMessage(content=advisor_input)
    ])

    return {
        "messages": [response],
        "should_summarize": False,
    }


@traceable
def should_summarize(state: SaveyState) -> str:
    if state.get("should_summarize", False):
        return "advisor_agent"
    return END


# =============================================================================
# Graph
# =============================================================================

graph_builder = StateGraph(SaveyState)

graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", tool_node)
graph_builder.add_node("update_state", update_state_node)
graph_builder.add_node("advisor_agent", advisor_node)

graph_builder.set_entry_point("agent")
graph_builder.add_conditional_edges("agent", should_continue)
graph_builder.add_edge("tools", "update_state")
graph_builder.add_conditional_edges("update_state", should_summarize)
graph_builder.add_edge("advisor_agent", END)

checkpointer = MemorySaver()
savey = graph_builder.compile(checkpointer=checkpointer)


# =============================================================================
# Graph display
# =============================================================================

savey_graph_out = widgets.Output()
duration_graph_out = widgets.Output()

with savey_graph_out:
    print("🧠 Savey (parent StateGraph)")
    display(Image(savey.get_graph().draw_mermaid_png()))

with duration_graph_out:
    print("🔍 Duration Sub-Agent")
    display(Image(duration_agent.get_graph().draw_mermaid_png()))

display(widgets.HBox([savey_graph_out, duration_graph_out]))


# =============================================================================
# Demo
# =============================================================================

config = {"configurable": {"thread_id": "savey-demo-001"}}

result_1 = savey.invoke(
    {
        "messages": [{"role": "user", "content": "Monday I bought a £4 coffee and a £8 sandwich."}],
        "expense_log": [],
        "total_spent": 0.0,
        "days_tracked": 0,
        "_todo_state": [],
        "should_summarize": False,
    },
    config=config
)

format_messages(result_1["messages"])
print("\nState after turn 1:")
print(f"   expense_log : {result_1['expense_log']}")
print(f"   total_spent : £{result_1['total_spent']}")
print(f"   days_tracked: {result_1['days_tracked']}")
print(f"   should_summarize: {result_1['should_summarize']}")

result_2 = savey.invoke(
    {"messages": [{"role": "user", "content": "Yesterday I spent $20 on dinner."}]},
    config=config
)

format_messages(result_2["messages"])
print("\nState after turn 2:")
print(f"   expense_log : {result_2['expense_log']}")
print(f"   total_spent : £{result_2['total_spent']}")
print(f"   days_tracked: {result_2['days_tracked']}")
print(f"   should_summarize: {result_2['should_summarize']}")

result_3 = savey.invoke(
    {"messages": [{"role": "user", "content": "Can you give me a summary of my spending so far and some advice on how I can save money?"}]},
    config=config
)

format_messages(result_3["messages"])
print("\nState after turn 3:")
print(f"   expense_log : {result_3['expense_log']}")
print(f"   total_spent : £{result_3['total_spent']}")
print(f"   days_tracked: {result_3['days_tracked']}")
print(f"   should_summarize: {result_3['should_summarize']}")