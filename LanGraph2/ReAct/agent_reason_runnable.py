from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import TypedDict, List, Annotated
import operator
import datetime

llm = ChatOpenAI(model="gpt-4o")

# ─────────────────────────────────────────
# 1. DEFINE CUSTOM STATE
# ─────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List, operator.add]  # messages accumulate
    iteration: int                           # custom field - track loops
    final_answer: str                        # custom field - store final answer

# ─────────────────────────────────────────
# 2. DEFINE TOOLS
# ─────────────────────────────────────────
@tool
def get_time() -> str:
    """Returns current time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def multiply(a: float, b: float) -> float:
    """Multiplies two numbers"""
    return a * b

tools = [get_time, multiply]
tools_map = {t.name: t for t in tools}
llm_with_tools = llm.bind_tools(tools)

# ─────────────────────────────────────────
# 3. DEFINE NODES (just Python functions)
# ─────────────────────────────────────────
def reason_node(state: AgentState) -> AgentState:
    """Agent thinks and decides what to do"""
    print(f"--- REASON NODE (iteration {state['iteration']}) ---")
    response = llm_with_tools.invoke(state["messages"])
    return {
        "messages": [response],
        "iteration": state["iteration"] + 1,
        "final_answer": ""
    }

def act_node(state: AgentState) -> AgentState:
    """Execute tool calls"""
    print("--- ACT NODE ---")
    last_message = state["messages"][-1]
    tool_messages = []
    for tool_call in last_message.tool_calls:
        tool_fn = tools_map[tool_call["name"]]
        result = tool_fn.invoke(tool_call["args"])
        tool_messages.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )
    return {"messages": tool_messages, "iteration": state["iteration"], "final_answer": ""}

def final_node(state: AgentState) -> AgentState:
    """Extract and store final answer"""
    print("--- FINAL NODE ---")
    final = state["messages"][-1].content
    return {"messages": [], "iteration": state["iteration"], "final_answer": final}

# ─────────────────────────────────────────
# 4. ROUTING LOGIC
# ─────────────────────────────────────────
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if state["iteration"] > 5:
        return "final"                       # safety — max iterations
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "act"                         # has tool calls → execute them
    return "final"                           # no tool calls → done

# ─────────────────────────────────────────
# 5. BUILD THE GRAPH
# ─────────────────────────────────────────
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("reason", reason_node)
graph.add_node("act", act_node)
graph.add_node("final", final_node)

# Add edges
graph.add_edge(START, "reason")                      # start → reason
graph.add_conditional_edges("reason", should_continue, {
    "act": "act",                                    # has tool calls → act
    "final": "final"                                 # no tool calls → final
})
graph.add_edge("act", "reason")                      # act → reason (loop back)
graph.add_edge("final", END)                         # final → end

# Compile
app = graph.compile()

# ─────────────────────────────────────────
# 6. RUN IT
# ─────────────────────────────────────────
result = app.invoke({
    "messages": [HumanMessage(content="What is the current time and what is 6 multiplied by 7?")],
    "iteration": 0,
    "final_answer": ""
})

print("\n=== FINAL ANSWER ===")
print(result["final_answer"])
# ```

# ---

# ### The graph looks like this:
# ```
# START
#   ↓
# [reason_node]  ← ─────────────┐
#   ↓                            │
# should_continue?               │
#   ├── "act"  → [act_node] ─────┘  (loop back)
#   └── "final" → [final_node]
#                     ↓
#                    END