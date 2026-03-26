from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from typing import TypedDict, List

# Load env
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

# ---- STATE ----
class GraphSchema(TypedDict):
    messages: List[BaseMessage]

# ---- PROMPTS ----
generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            " Generate the best twitter post possible for the user's request."
            " If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a viral twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            " Always provide detailed recommendations including virality, style, length, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# ---- LLM ----
llm = ChatOpenAI(model="gpt-5-mini", temperature=1,  verbose=True,)

generation_chain = generation_prompt | llm
reflection_chain = reflection_prompt | llm

# ---- GRAPH ----
GENERATE = "generate"
REFLECT = "reflect"

graph = StateGraph(GraphSchema)

# ---- NODES ----
def generate_node(state: GraphSchema):
    response = generation_chain.invoke({
        "messages": state["messages"]
    })

    return {
        "messages": state["messages"] + [response]
    }


def reflect_node(state: GraphSchema):
    response = reflection_chain.invoke({
        "messages": state["messages"]
    })

    return {
        "messages": state["messages"] + [HumanMessage(content=response.content)]
    }

# ---- EDGES ----
def should_continue(state: GraphSchema):
    if len(state["messages"]) > 3:
        return END
    return REFLECT

# ---- BUILD GRAPH ----
graph.add_node(GENERATE, generate_node)
graph.add_node(REFLECT, reflect_node)

graph.add_edge(START, GENERATE)
graph.add_conditional_edges(GENERATE, should_continue)
graph.add_edge(REFLECT, GENERATE)

app = graph.compile()

# ---- VISUALIZE ----
print(app.get_graph().draw_mermaid())
app.get_graph().print_ascii()

# ---- RUN ----
response = app.invoke({
    "messages": [HumanMessage(content="AI Agents taking over content creation")]
})

print(response)