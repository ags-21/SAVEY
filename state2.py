from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class SaveyState(TypedDict):
    messages: Annotated[list, add_messages]
    short_summaries: list[str]
    long_memory: str
    expense_log: list
    total_spent: float
    days_tracked: int
    todo: list[str]
    user_profile: dict