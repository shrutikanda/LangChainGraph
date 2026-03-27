from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_tavily import TavilySearch
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv, find_dotenv
import sqlite3

# Load environment variables (for API keys, etc.)
load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=1, verbose=True)

# Setup SQLite connection & checkpointer
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Define State
class BasicChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Chatbot function
def chatbot(state: BasicChatState):
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

# Build the graph
graph = StateGraph(BasicChatState)

graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)
graph.set_entry_point("chatbot")

# Compile graph with checkpointing
app = graph.compile(checkpointer=memory)

# Configuration for checkpointing
config = {"configurable": {"thread_id": "1"}}  # Thread ID must be a string

# Interactive loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "end"]:
        break

    # Invoke the graph with user input
    result = app.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )

    # Print AI response
    ai_message: AIMessage = result["messages"][-1]
    print("AI: " + ai_message.content)