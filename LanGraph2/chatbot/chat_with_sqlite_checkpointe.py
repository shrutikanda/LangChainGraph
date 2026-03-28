from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv, find_dotenv
import sqlite3

# Load environment variables
load_dotenv(find_dotenv())

# Initialize LLM
llm = ChatOpenAI(model="gpt-5-mini", temperature=1, verbose=True)

# Setup SQLite connection & checkpointer
sqlite_conn = sqlite3.connect("checkpoint.sqlite", check_same_thread=False)
memory = SqliteSaver(sqlite_conn)

# Define State
class BasicChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Function to trim chat history (keep last N messages)
def get_trimmed_context(messages: List[BaseMessage], max_messages: int = 6) -> List[BaseMessage]:
    return messages[-max_messages:]

# Chatbot function with trimmed context
def chatbot(state: BasicChatState):
    # Only send last few messages to the model
    trimmed_messages = get_trimmed_context(state["messages"])
    response = llm.invoke(trimmed_messages)
    return {"messages": [response]}

# Build the graph
graph = StateGraph(BasicChatState)
graph.add_node("chatbot", chatbot)
graph.add_edge("chatbot", END)
graph.set_entry_point("chatbot")

# Compile graph with checkpointing
app = graph.compile(checkpointer=memory)

# Configuration for checkpointing
config = {"configurable": {"thread_id": "1"}}

# Interactive loop
chat_history: List[BaseMessage] = []

while True:
    user_input = input("User: ")
    if user_input.lower() in ["exit", "end"]:
        break

    # Append user message
    chat_history.append(HumanMessage(content=user_input))

    # Invoke chatbot with full history (for memory), but model sees trimmed
    result = app.invoke({"messages": chat_history}, config=config)

    # Get AI response
    ai_message: AIMessage = result["messages"][-1]
    print("AI: " + ai_message.content)

    # Store AI response in history
    chat_history.append(ai_message)