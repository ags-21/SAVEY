# Run the practice questions here
# provides updates to savey

import os
from dotenv import load_dotenv
from utils import format_messages

import re
import json

# LangGraph — for building the main graph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# LangChain — for the model and messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages

# State definition
from typing import TypedDict, Annotated
from datetime import date

load_dotenv()
import saveytools as st  # must come after load_dotenv() so the API key is available to the tools

from langsmith import traceable

# ── State ─────────────────────────────────────────────────────────────────────

class SaveyState(TypedDict):
    messages: Annotated[list, add_messages]
    expense_log: list
    total_spent: float
    days_tracked: int
    todo_list: list       # each item: {"step": str, "status": "pending"|"in_progress"|"done"}
    complexity: str       # "simple" | "complex" — set by router, read by conditional edge

# ── Tools ─────────────────────────────────────────────────────────────────────

savey_tools = [
    st.retrieve_total_expenses,
    st.retrieve_purchased_item,
    st.ask_duration_agent,
    st.ask_converter_agent, # replaces convert_to_gbp
]

# ── Prompts ───────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are a task classifier for Savey, an expense-tracking assistant.

Your only job is to read the user's latest message and decide whether it is:

SIMPLE  — requires only ONE tool call or a single direct answer.
          Examples: "What did I spend total?" / "Convert $20 to GBP"

COMPLEX — requires TWO OR MORE tool calls, multiple questions answered together,
          or careful step-by-step sequencing.
          Examples: "What did I spend, what's my most bought item, and how many days?"

Reply with EXACTLY one word: either "simple" or "complex". Nothing else."""

PLANNER_PROMPT = """You are the planning step for Savey, an expense-tracking assistant.

The user has a complex, multi-step request. Your job is to break it down into a clear 
ordered list of steps that Savey should execute — one step per tool call needed.

Available tools:
- retrieve_total_expenses: sum GBP/USD amounts from text
- retrieve_purchased_item: identify the most frequently bought item
- ask_duration_agent: count how many distinct days expenses span
- convert_to_gbp: convert a foreign currency amount to GBP (use before retrieve_total_expenses if needed)

Return ONLY a JSON array of step description strings, in execution order.
No explanation, no markdown, no code fences. Just the raw JSON array.

Example output:
["Convert $20 USD to GBP", "Calculate total expenses in GBP", "Count number of days"]"""

AGENT_PROMPT = """You are Savey 💾 — an expert at tracking and advising on day-to-day expenses.

You have access to the following tools:
- retrieve_total_expenses: for summing up GBP amounts from text
- retrieve_purchased_item: for identifying the most frequently bought item
- ask_duration_agent: for determining how many days a description spans
- ask_converter_agent: for converting foreign currency expenses to GBP — use this whenever
  the user mentions non-GBP currencies. It handles multiple currencies and historic rates automatically.

Always use ask_converter_agent before retrieve_total_expenses if any foreign currencies are present."""

# ── Models ────────────────────────────────────────────────────────────────────

model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)
router_model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)
planner_model = init_chat_model(model="openai:gpt-4.1-mini", temperature=0.0)
model_with_tools = model.bind_tools(savey_tools, parallel_tool_calls=False)

# ── Node 1: router ────────────────────────────────────────────────────────────
# Classifies the latest user message as "simple" or "complex".
# Sets state["complexity"] which the conditional edge reads to branch the graph.

@traceable
def router_node(state: SaveyState) -> dict:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )
    if last_human is None:
        return {"complexity": "simple"}

    response = router_model.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=last_human.content)
    ])

    complexity = response.content.strip().lower()
    if complexity not in ("simple", "complex"):
        complexity = "simple"  # safe default

    print(f"🔀 Router decision: {complexity}")
    return {"complexity": complexity}


# ── Node 2: planner ───────────────────────────────────────────────────────────
# Only reached on complex tasks. Asks the LLM to produce an ordered step list
# and writes it to state["todo_list"] as pending items.

@traceable
def planner_node(state: SaveyState) -> dict:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None
    )

    response = planner_model.invoke([
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=last_human.content)
    ])

    try:
        steps = json.loads(response.content.strip())
        if not isinstance(steps, list):
            raise ValueError("Not a list")
    except (json.JSONDecodeError, ValueError):
        steps = ["Complete the user's request"]  # graceful fallback

    todo_list = [{"step": s, "status": "pending"} for s in steps]

    print(f"📋 TODO created ({len(todo_list)} steps):")
    for i, item in enumerate(todo_list):
        print(f"   {i}. ⬜ {item['step']}")

    return {"todo_list": todo_list}


# ── Node 3: agent ─────────────────────────────────────────────────────────────
# The LLM reasoning step. Reads full state (including TODO) and decides next tool.

@traceable
def agent_node(state: SaveyState) -> dict:
    todo_list = state.get("todo_list", [])

    # Build a readable TODO summary to inject into the system prompt
    status_icons = {"pending": "⬜", "in_progress": "🔄", "done": "✅"}
    if todo_list:
        todo_lines = ["", "TODO list:"]
        for i, item in enumerate(todo_list):
            icon = status_icons.get(item["status"], "⬜")
            todo_lines.append(f"  {i}. {icon} {item['step']}")
        pending = [i for i, t in enumerate(todo_list) if t["status"] == "pending"]
        if pending:
            todo_lines.append(f"\n➡️  Next: step {pending[0]} — '{todo_list[pending[0]]['step']}'")
        else:
            todo_lines.append("\n🎉 All steps complete — compile your final answer.")
        todo_summary = "\n".join(todo_lines)
    else:
        todo_summary = ""

    system_message = SystemMessage(content=f"""{AGENT_PROMPT}

Current state:
- Total spent so far: £{state.get('total_spent', 0.0)}
- Days tracked: {state.get('days_tracked', 0)}
- Expense log: {state.get('expense_log', [])}
{todo_summary}
If the user asks for a summary or total, read directly from the state above — do not call any tools to recalculate.""")

    response = model_with_tools.invoke([system_message] + state["messages"])
    return {"messages": [response]}


# ── Node 4: tools ─────────────────────────────────────────────────────────────
# Executes whichever tool the agent called. ToolNode handles this automatically.

tool_node = ToolNode(savey_tools)

# ── Node 5: update_state ──────────────────────────────────────────────────────
# Pure Python. Reads ToolMessages and updates expense/duration/TODO state fields.

@traceable
def update_state_node(state: SaveyState) -> dict:
    expense_log = list(state.get("expense_log", []))
    days_tracked = state.get("days_tracked", 0)
    todo_list = [dict(t) for t in state.get("todo_list") or []]

    messages = state["messages"]
    last_ai_idx = next(
        (i for i in reversed(range(len(messages)))
         if isinstance(messages[i], AIMessage) and messages[i].tool_calls),
        None
    )

    if last_ai_idx is None:
        return {}

    # The tool calls from the most recent AI message — used to advance the TODO
    last_ai_tool_names = [tc["name"] for tc in messages[last_ai_idx].tool_calls]

    current_tool_messages = [
        m for m in messages[last_ai_idx + 1:]
        if isinstance(m, ToolMessage)
    ]

    converted_this_round = any(m.name == "ask_converter_agent" for m in current_tool_messages)

    for msg in current_tool_messages:
        content = msg.content.strip()

        if msg.name == "ask_converter_agent":
            try:
                conversions = json.loads(content)
                for conversion in conversions:
                    expense_log.append({
                        "item": "(foreign currency expense)",
                        "amount_gbp": float(conversion["gbp"]),
                        "date": conversion["date"],
                        "original": conversion["original"]
                    })
                # Tick off one TODO step per conversion returned
                for _ in conversions:
                    for i, item in enumerate(todo_list):
                        if item["status"] == "pending":
                            todo_list[i]["status"] = "done"
                            print(f"   ✅ TODO step {i} done: '{todo_list[i]['step']}'")
                            break
            except (json.JSONDecodeError, KeyError, ValueError):
                print("Could not parse converter agent response")

        elif msg.name == "retrieve_total_expenses" and not converted_this_round:
            try:
                amount_gbp = float(content)
                if amount_gbp > 0:
                    expense_log.append({
                        "item": "(expenses from message)",
                        "amount_gbp": amount_gbp
                    })
            except ValueError:
                pass

        elif msg.name == "ask_duration_agent":
            try:
                days_tracked += int(content)
            except ValueError:
                print("Could not retrieve duration")

    # ── Advance the TODO list ──────────────────────────────────────────────────
    # Find the first pending step whose description loosely matches the tool just called,
    # or simply advance the first pending step (since the agent works sequentially).
    if todo_list:
        tool_to_keyword = {
            "retrieve_total_expenses": ["total", "expense", "spent", "spend", "sum"],
            "retrieve_purchased_item": ["item", "purchased", "bought", "frequent"],
            "ask_duration_agent":      ["day", "duration", "period", "how long"],
            "convert_to_gbp":          ["convert", "gbp", "currency", "usd", "eur"],
        }

        for tool_name in last_ai_tool_names:
            keywords = tool_to_keyword.get(tool_name, [])
            # Try to match by keyword first, fall back to first pending
            matched_idx = None
            for i, item in enumerate(todo_list):
                if item["status"] == "pending":
                    if any(kw in item["step"].lower() for kw in keywords):
                        matched_idx = i
                        break
            if matched_idx is None:
                # Fall back to first pending step
                for i, item in enumerate(todo_list):
                    if item["status"] == "pending":
                        matched_idx = i
                        break
            if matched_idx is not None:
                todo_list[matched_idx]["status"] = "done"
                print(f"   ✅ TODO step {matched_idx} done: '{todo_list[matched_idx]['step']}'")

    total_spent = round(sum(e["amount_gbp"] for e in expense_log), 2)

    return {
        "expense_log": expense_log,
        "total_spent": total_spent,
        "days_tracked": days_tracked,
        "todo_list": todo_list,
    }

# ── Routing functions ─────────────────────────────────────────────────────────

def route_by_complexity(state: SaveyState) -> str:
    """After router_node: branch to planner (complex) or agent (simple)."""
    return "planner" if state.get("complexity") == "complex" else "agent"

@traceable
def should_continue(state: SaveyState) -> str:
    """After agent_node: go to tools if there's a tool call, otherwise END."""
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return END


# ── Assemble the graph ────────────────────────────────────────────────────────
#
#  START → router → (complex?) → planner → agent → should_continue? → tools → update_state → agent (loop)
#                      ↓ (simple)                        ↓ (no tool call)
#                    agent                              END

graph_builder = StateGraph(SaveyState)

graph_builder.add_node("router",       router_node)
graph_builder.add_node("planner",      planner_node)
graph_builder.add_node("agent",        agent_node)
graph_builder.add_node("tools",        tool_node)
graph_builder.add_node("update_state", update_state_node)

graph_builder.set_entry_point("router")
graph_builder.add_conditional_edges("router", route_by_complexity, {"planner": "planner", "agent": "agent"})
graph_builder.add_edge("planner", "agent")
graph_builder.add_conditional_edges("agent", should_continue)
graph_builder.add_edge("tools", "update_state")
graph_builder.add_edge("update_state", "agent")

checkpointer = MemorySaver()

#expose the compiled graph as a callable function that takes the state dict and returns the final messages
savey = graph_builder.compile(checkpointer=checkpointer)

# ── Run ─────────────────────────────────────────────────────────────────────
