import chainlit as cl
from savey import savey  # import compiled graph

import time

@cl.on_chat_start
async def on_chat_start():
    # Create a fresh thread per user session
    thread_id = f"savey-{int(time.time())}"
    cl.user_session.set("config", {"configurable": {"thread_id": thread_id}})
    cl.user_session.set("state", {
        "expense_log": [],
        "total_spent": 0.0,
        "days_tracked": 0,
        "todo_list": [],
        "complexity": ""
    })
    await cl.Message(content="Hi! I'm Savey 💾 — tell me about your expenses.").send()

@cl.on_message
async def on_message(message: cl.Message):
    config = cl.user_session.get("config")
    state = cl.user_session.get("state")

    result = savey.invoke(
        {**state, "messages": [{"role": "user", "content": message.content}]},
        config=config
    )

    # Persist updated state for next turn
    cl.user_session.set("state", {
        "expense_log": result["expense_log"],
        "total_spent": result["total_spent"],
        "days_tracked": result["days_tracked"],
        "todo_list": result["todo_list"],
        "complexity": result["complexity"],
    })

    # Get Savey's final response
    last_ai = next(
        (m for m in reversed(result["messages"]) if hasattr(m, "type") and m.type == "ai"),
        None
    )
    if last_ai:
        await cl.Message(content=last_ai.content).send()