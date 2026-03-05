from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage, ToolMessage
from pathlib import Path
from dotenv import load_dotenv
from typing import TypedDict,List, Annotated
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from operator import add 
from typing import TypedDict, List 
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from pprint import PrettyPrinter, pprint
# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

llm = ChatOpenAI(model="gpt-5-mini", temperature=1)

@tool
def personal_info(name: str):

    """Use this tool to get personal information about Alice, Bob, or Charlie. 
    """

    info = {
        "Alice": "Alice is a software engineer with 5 years of experience in AI.",
        "Bob": "Bob is a data scientist who loves working with large datasets.",
        "Charlie": "Charlie is a product manager with a background in tech startups."
    }
    return info.get(name, "No information available for this person.")

@tool
def user_bank_info(name: str):

    """Use this tool to get bank information about Alice, Bob, or Charlie. 
    """

    info = {
        "Alice": "Alice having $5000.",
        "Bob": "Bob having $3000.",
        "Charlie": "Charlie having $7000."
    }
    return info.get(name, "No information available for this person.")


tools = [personal_info, user_bank_info]

llm_with_tools = llm.bind_tools(tools)

class graph_scehma(TypedDict):
    messages: List

def llm_node(state: graph_scehma) -> graph_scehma:

    messages = state["messages"]

    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. which can use tools to get information about people and their bank info."),
        ("human", "{input}")
    ])

    chain = prompt | llm_with_tools

    response = chain.invoke({"input": messages})
        
    state["messages"] = state["messages"] + [response]

    return state

def tool_call(state: graph_scehma) -> graph_scehma:

    messages = state["messages"]

    tools_by_name = {tool.name: tool for tool in tools}
    print("Tools by name:", tools_by_name)

    tool_results = []

    for tool_call in messages[-1].tool_calls:
         tool = tools_by_name.get(tool_call["name"])
         observation = tool.invoke(tool_call["args"])
         tool_results.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
 
    # Update the state with the tool results
    state['messages'] = messages + tool_results

    return state

def if_tools_call(state: graph_scehma) -> str:
     
     last_message = state["messages"][-1]

     if last_message.tool_calls:
         return "tool_call"
     else:
          return "end"

graph = StateGraph(graph_scehma)

graph.add_node("llm_node", llm_node)
graph.add_node("tool_call", tool_call)

graph.add_edge(START, "llm_node")   
graph.add_conditional_edges("llm_node",if_tools_call, {"tool_call": "tool_call", "end": END})
graph.add_edge("tool_call", "llm_node")
graph.add_edge("llm_node", END)
  

react_graph = graph.compile()


png_bytes = react_graph.get_graph().draw_mermaid_png()

with open("6_ReAct.png", "wb") as f:
            f.write(png_bytes) 

response = react_graph.invoke({"messages": ["What is Alice's bank info?"]})
pp = PrettyPrinter(indent=2, width=80)
pp.pprint(response)

# for chunk in react_graph.stream(
#     {"messages": [HumanMessage(content="What is the latest news on AI?")]},
#     stream_mode="updates"
# ):
    #print(chunk)