from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display

# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

class graph_scehma(BaseModel):
    topic: str = Field(description="The topic to generate a social media post about")
    post : str = Field(description="The generated linked post")
    curated_post: str = Field(description="The curated Loinked post ")
    


def create_post(state:graph_scehma) -> graph_scehma:

    state = state.model_dump()
    topic = state["topic"]    

    post = llm.invoke(f"write linkedin post about {topic}").content

    state["post"] = post

    return state

def curate_post(state:graph_scehma) -> graph_scehma:

    state = state.model_dump()

    post = state["post"]    

    curated_post = llm.invoke(f"curate linkedin post: {post}").content

    state["curated_post"] = curated_post

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
                      {"topic": "Generate a linkedin post about .Net jobs in US now a days",            
                      "post": "",
                      "curated_post": ""}
        )

    print(response)

    

if __name__ == "__main__":
    main()
