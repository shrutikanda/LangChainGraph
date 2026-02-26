from langchain_openai import ChatOpenAI
from langchain_core.messages  import HumanMessage, SystemMessage, AIMessage
from pathlib import Path
from dotenv import load_dotenv
import tiktoken


# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def main():
    llm = ChatOpenAI(model="gpt-5-mini", temperature=1)
    request = "What is the capital of France?"
    
    my_messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=request),        
    ]

    response = llm.invoke(my_messages).content
    print(response)

    ### Promiot with tempalte ###
    from langchain_core.prompts import PromptTemplate

    user_template = input("Enter a topic for a fun fact: ")

    dynamic_prompt = PromptTemplate.from_template("Write a fun fact about {topic}")

    ready_prompt = dynamic_prompt.invoke({"topic": user_template})

    response = llm.invoke(ready_prompt).content
    print(response)


    # print(tiktoken.list_encoding_names())
    # enc = tiktoken.get_encoding("o200k_base")   

    # # To get the tokeniser corresponding to a specific model in the OpenAI API:
    # enc = tiktoken.encoding_for_model("gpt-5-mini")
    # print(enc.encode(request))
    # print(enc.encode(response))

if __name__ == "__main__":
    main()
