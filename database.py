import json
import os
from google.cloud import firestore

db = firestore.Client(project="savey-490713", database = "savey")

def fetch_user_profile(user_id: str):
    """Combines profile, long_memory, and recent short_summaries."""
    # 1. Get Profile & Long Memory
    doc = db.collection("users").document(user_id).get()
    data = doc.to_dict() if doc.exists else {}
    
    long_memory = data.get("long_memory", "No history yet.")
    user_profile = data # includes habits/persona
    
    # 2. Get 3 most recent short summaries (Teammate's code)
    summaries_ref = db.collection("users").document(user_id).collection("short_summaries")
    query = summaries_ref.order_by("created_at", direction=firestore.Query.DESCENDING).limit(3)
    
    docs = query.stream()
    recent_summaries = [doc.to_dict()["content"] for doc in docs]
    
    return {
        "long_memory": long_memory,
        "short_summaries": recent_summaries[::-1], # Chronological order
        "user_profile": user_profile
    }

def update_ltm_profile(user_id: str, updates: dict):
    """
    Task: Write-back function for the 'save_memory' node.
    Updates the user's permanent profile with new AI insights.
    """
   
    doc_ref = db.collection("users").document(user_id)
    doc_ref.set(updates, merge=True)
    print(f"Cloud DB updated for {user_id}")

    print(f"MOCK DB UPDATE: {user_id} received {updates}")