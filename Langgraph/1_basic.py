from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv
import tiktoken
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, List
from IPython.display import Image, display

# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

class graph_scehma(TypedDict):
    name: str
    message: str


def weclome(state:graph_scehma) -> graph_scehma:
    curr_name = state["name"]
    curr_message = state["message"]

    response = llm.invoke(f"My name is {curr_name}.{curr_message}").content

    state["message"] = response

    return state

def main():
  
    graph = StateGraph(graph_scehma)

    graph.add_node("welcome", weclome)
    graph.add_edge(START, "welcome")
    graph.add_edge("welcome", END)

    first_graph = graph.compile()
    # png_bytes = first_graph.get_graph().draw_mermaid_png()

    # with open("1_basic.png", "wb") as f:
    #     f.write(png_bytes)

    # print("Graph saved as 1_basic.png")

    response = first_graph.invoke({"name": "Alice", "message": "How are you?"})

    print(response)

    

if __name__ == "__main__":
    main()
