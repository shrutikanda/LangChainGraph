from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict,List, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from operator import add 
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pprint import PrettyPrinter, pprint
from pydantic import BaseModel, Field

# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

class llm_schema(BaseModel):
    tasks: List[str] = Field(..., description="A list of tasks to be performed by the worker.")

llm_with_schema = llm.with_structured_output(llm_schema)

class graph_schema(TypedDict):

    tasks: List[str]
    query: str
    results: List[str]
    summary: str

def orchestrator_node(state: graph_schema) -> graph_schema:

    # Fetching the user query from the state
    user_query = state['query']

    # Create the prompt for the LLM
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an orchestrator that breaks down a user query into tasks for the worker."),
            ("user", f"User query: {user_query}. Please generate one prompt per task for the worker to complete. Return the tasks in a list format."),
        ]
    )

    # Create the chain
    chain = prompt | llm_with_schema

    # Run the chain with the user query as input
    response = chain.invoke({"query": user_query})


    # Update the state with the generated tasks
    state['tasks'] = response.tasks

    return state

# Execute Function

def execute(query:str) :

    response = llm.invoke(f"Please execute this task {query}")
    return response.content

from concurrent.futures import ThreadPoolExecutor

def worker_node(state: graph_schema) -> graph_schema:

    tasks = state['tasks']
    results = []

    with ThreadPoolExecutor(max_workers=len(tasks)) as executor:

        results_futures = executor.map(execute, tasks)
        for result in results_futures:
            results.append(result)
    
    state['results'] = results
    
    return state
def collector_node(state: graph_schema):

    results = state['results']

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a collector that summarizes the results from the worker."),
            ("user", f"Here are the results from the worker: {results}. Please summarize these results in a concise manner."),
        ]
    )

    chain = prompt | llm

    summary = chain.invoke({"results": results})

    state['summary'] = summary.content

    return state

graph = StateGraph(graph_schema)

graph.add_node("orchestrator_node",orchestrator_node)
graph.add_node("worker_node",worker_node)
graph.add_node("collector_node",collector_node)

graph.add_edge(START, "orchestrator_node")
graph.add_edge("orchestrator_node", "worker_node")
graph.add_edge("worker_node", "collector_node")
graph.add_edge("collector_node", END)

complex_graph = graph.compile()

complex_graph.invoke(
    {
        "query": "What is the capital of France and what is the population of Paris? & What is the capital of Germany and what is the population of Berlin?",
        "tasks": [],
        "results": [],
        "summary": "",
    }
)