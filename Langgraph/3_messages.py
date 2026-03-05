from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict,List, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from operator import add 

# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

class graph_scehma(TypedDict):
    messages_manual: List
    messages_auto: Annotated[List,add]


def create_post(state:graph_scehma) -> graph_scehma:

    messages_manual = state["messages_manual"]
    response_maunal = llm.invoke(messages_manual).content
    response_manual_ai = AIMessage(content=response_maunal)
    state["messages_manual"] = state["messages_manual"] + [response_manual_ai]


    return state

def curate_post(state:graph_scehma) -> graph_scehma:

    messages_manual = state["messages_manual"]
    response_maunal = llm.invoke(messages_manual).content
    response_manual_ai = AIMessage(content=response_maunal)
    state["messages_manual"] = state["messages_manual"] + [response_manual_ai]

    return state


def main():
  
    graph = StateGraph(graph_scehma)

    graph.add_node("create_post", create_post)
    graph.add_node("curate_post", curate_post)
    
    graph.add_edge(START, "create_post")
    graph.add_edge("create_post", "curate_post")
    graph.add_edge("curate_post", END)

    first_graph = graph.compile()
    # png_bytes = first_graph.get_graph().draw_mermaid_png()

    # with open("1_pydentic.png", "wb") as f:
    #     f.write(png_bytes)   

    response = first_graph.invoke(
                      {"messages_manual": [HumanMessage(content="Generate a linkedin post about .Net jobs in US now a days")],
                       "messages_auto": []}
        )

    print(response)

    

if __name__ == "__main__":
    main()
