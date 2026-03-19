import os
from datetime import datetime
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.messages import RemoveMessage
from langchain_core.runnables import RunnableConfig
from google.cloud import firestore
from langgraph_checkpoint_firestore import FirestoreSaver
# Import from your existing files
from state import SaveyState
from tools import TOOLS
from database import db, fetch_user_profile  # Ensure db is exported in database.py

load_dotenv()
# 1. Initialize Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(TOOLS)

def load_memory_node(state: SaveyState, config: RunnableConfig):
    """Bridge between Firestore and the graph: Loads Profile + Recent Summaries"""
    user_id = config["configurable"].get("user_id", "anon_user")
    
    # Get Profile (your existing function)
    profile_data = fetch_user_profile(user_id)
    
    # Get 3 most recent summaries (Teammate's logic)
    summaries_ref = db.collection("users").document(user_id).collection("short_summaries")
    query = summaries_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(3)
    docs = query.stream()
    recent_summaries = [doc.to_dict()["content"] for doc in docs][::-1]

    return {
        "user_profile": profile_data,
        "short_summaries": recent_summaries,
        "long_memory": profile_data.get("long_memory", "")
    }

def generate_short_summary_node(state: SaveyState, config: RunnableConfig):
    """Teammate's Code: Summarizes current session and cleans up messages"""
    user_id = config["configurable"].get("user_id", "anon_user")
    messages = state["messages"]
    existing_summaries = "\n".join(state.get("short_summaries", []))
    long_memory = state.get("long_memory", "")

    # 1. Generate the summary
    summary_prompt = f"""
    Distill the following conversation into a concise summary (2-3 sentences).
    CONTEXT (LONG-TERM): {long_memory}
    PREVIOUS SESSION SUMMARIES: {existing_summaries}
    NEW MESSAGES: {messages}
    """
    response = model.invoke(summary_prompt)
    new_short_summary = response.content

    # 2. Save to Firestore
    db.collection("users").document(user_id).collection("short_summaries").add({
        "content": new_short_summary,
        "created_at": firestore.SERVER_TIMESTAMP,
    })

    # 3. Clean up the message history (Deletes messages from state to save space)
    delete_messages = [RemoveMessage(id=m.id) for m in messages[:-1]]

    return {
        "short_summaries": state.get("short_summaries", []) + [new_short_summary],
        "messages": delete_messages,
    }

def update_long_memory_node(state: SaveyState, config: RunnableConfig):
    """Teammate's Code: Merges short summaries into long-term memory every 3 sessions"""
    if len(state.get("short_summaries", [])) < 3:
        return {}

    user_id = config["configurable"].get("user_id", "anon_user")
    current_long = state.get("long_memory", "No existing long-term memory.")
    new_data = "\n".join(state["short_summaries"])

    prompt = f"Update the 'Long-Term Memory' based on 'Recent Short-Term Summaries'.\nOLD: {current_long}\nNEW: {new_data}"
    response = model.invoke(prompt)
    updated_long = response.content

    # Update Firestore
    db.collection("users").document(user_id).set({"long_memory": updated_long}, merge=True)

    return {"long_memory": updated_long, "short_summaries": []}

def agent_node(state: SaveyState):
    """The 'Brain': Decides what to do based on LTM + current messages."""
    profile = state.get("user_profile", {})
    persona = profile.get("financial_persona", "a new user")
    goals = profile.get("goals", "no specific goals yet")
    history = state.get("long_memory", "No previous history")

    system_msg = {
        "role": "system", 
        "content": f"You are Savey. User Profile: {persona}. Goals: {goals}. "
                    f"Historical Context: {history}."
                   "Use this context to tailor your advice. If the user's current "
                   "spending contradicts their goals, gently point it out."
    }
    response = model.invoke([system_msg] + state["messages"])
    return {"messages": [response]}

def update_state_node(state: SaveyState):
    """The 'Accountant': Updates numerical state based on tool outputs."""
    # .get(key, default) prevents the KeyError if the state is uninitialized
    new_total = state.get("total_spent", 0.0)
    new_log = list(state.get("expense_log", []))
    new_todo = list(state.get("todo", []))
    
    # We look at the LAST message (which should be the ToolMessage from the 'tools' node)
    last_message = state["messages"][-1]
    
    # Logic: If the agent just used 'retrieve_total_expenses', add that to our running total
    if hasattr(last_message, "content"):
        # This is a simple logic check - in a production app, you'd match 
        # the specific ToolMessage ID to the tool output.
        try:
            # If the tool returned a number, add it
            val = float(last_message.content)
            new_total += val
        except ValueError:
            pass # Not a number, skip updating total
            
    return {
        "total_spent": new_total, 
        "expense_log": new_log,
        "todo": new_todo
    }

# 4. BUILD THE UPDATED GRAPH
# ==========================================
builder = StateGraph(SaveyState)

builder.add_node("load_memory", load_memory_node)
builder.add_node("agent", agent_node) 
builder.add_node("tools", ToolNode(TOOLS))
builder.add_node("update_state", update_state_node)
builder.add_node("summarize", generate_short_summary_node)
builder.add_node("long_term_sync", update_long_memory_node)

# Flow logic
builder.add_edge(START, "load_memory")
builder.add_edge("load_memory", "agent")

def should_continue(state: SaveyState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "summarize" # Instead of END, go to summarize

builder.add_conditional_edges("agent", should_continue)
builder.add_edge("tools", "update_state")
builder.add_edge("update_state", "agent")

# Memory logic flow
builder.add_edge("summarize", "long_term_sync")
builder.add_edge("long_term_sync", END)

# Checkpointer
from langgraph_checkpoint_firestore import FirestoreSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from database import db

class SaveyFirestoreSaver(FirestoreSaver):
    def __init__(self, client, collection_name="checkpoints"):
        # 1. Set the client and collection names
        self.client = client
        self.collection_name = collection_name
        self.writes_collection_name = "checkpoint_writes" # The missing piece
        
        # 2. Define the internal collection objects
        self.checkpoints_collection = self.client.collection(self.collection_name)
        self.writes_collection = self.client.collection(self.writes_collection_name) # Fixes the current error
        
        # 3. Setup the Serializer
        self.firestore_serde = JsonPlusSerializer()
        
        # 4. Standard LangGraph flags
        self.json = True 

# --- INITIALIZE ---
checkpointer = SaveyFirestoreSaver(db)
savey = builder.compile(checkpointer=checkpointer)
# Example Usage
if __name__ == "__main__":
    # We provide the full initial state to prevent any KeyErrors in the first node
    initial_input = {
        "messages": [("user", "I spent £10 on lunch and £5 on coffee. What is my total?")],
        "total_spent": 0.0,
        "expense_log": [],
        "todo": []
    }
    
    config = {
        "configurable": {
            "thread_id": "session_001",
            "user_id": "user001"
            }}
    
    print("--- Starting Savey Session ---")
    
    for event in savey.stream(initial_input, config):
    # event is a dictionary where keys are node names
        for node_name, value in event.items():
            print(f"\n--- Node: {node_name} ---")
            
            # 1. Check for messages to print
            if isinstance(value, dict) and "messages" in value:
                last_msg = value["messages"][-1]
                if hasattr(last_msg, 'content') and last_msg.content:
                    print(f"Savey: {last_msg.content}")
            
            # 2. Check for state updates
            if isinstance(value, dict) and "total_spent" in value:
                print(f"Total Spent: £{value['total_spent']}")