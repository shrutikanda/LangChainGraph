from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict,Optional,Literal,List, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from operator import add 
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pprint import PrettyPrinter, pprint
from pydantic import BaseModel, Field
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

class ApprovalState(TypedDict):
    action_details: str
    status: Optional[Literal["pending", "approved", "rejected"]]

def approval_node(state: ApprovalState) -> Command[Literal["proceed", "cancel"]]:
    # Expose details so the caller can render them in a UI
    decision = interrupt({
        "question": "Approve this action?",
        "details": state["action_details"],
    })

    # Route to the appropriate node after resume
    return Command(goto="proceed" if decision else "cancel")

def proceed_node(state: ApprovalState):
    return {"status": "approved"}


def cancel_node(state: ApprovalState):
    return {"status": "rejected"}


builder = StateGraph(ApprovalState)
builder.add_node("approval", approval_node)
builder.add_node("proceed", proceed_node)
builder.add_node("cancel", cancel_node)
builder.add_edge(START, "approval")
builder.add_edge("proceed", END)
builder.add_edge("cancel", END)

# Use a more durable checkpointer in production
checkpointer = MemorySaver()
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "approval-456"}}
initial = graph.invoke(
    {"action_details": "Send Daily Summary", "status": "pending"},
    config=config,
)
print(initial["__interrupt__"])  # -> [Interrupt(value={'question': ..., 'details': ...})]