from  langchain_openai import ChatOpenAI
import os 
from dotenv import load_dotenv
from pathlib import Path
# This function will load all the variables from the .env file and will 
# make them available in the os.environ dictionary (env variables)
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

if os.environ.get("OPENAI_API_KEY"):
    print("Bro API KEY Variable exists")
else:
    raise ValueError("OPENAI_API_KEY not found")

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from  langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser

llm_openai = ChatOpenAI(model="gpt-5-mini",temperature=0)

from langchain_community.utilities.sql_database import SQLDatabase

sql_db = SQLDatabase.from_uri("sqlite:///SalesDB/sales.db")

from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=sql_db, llm=llm_openai)
toolkit.get_tools()

from langchain.agents import create_agent

agent = create_agent(llm_openai, toolkit.get_tools())
agent

example_query = "How much total sales we made for Tablet"

events = agent.stream(
    {"messages": [("user", example_query)]},
    stream_mode="values",
)
for event in events:
    event["messages"][-1].pretty_print()