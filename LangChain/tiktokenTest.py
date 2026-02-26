from langchain_openai import ChatOpenAI
from pathlib import Path
from dotenv import load_dotenv
import os
import tiktoken


# Go up one level from CH-1 to project root
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def main():
    # llm = ChatOpenAI(model="gpt-5-mini", temperature=1)
    # response = llm.invoke("What is the capital of France?")
    # print(response)

    print(tiktoken.list_encoding_names())
    enc = tiktoken.get_encoding("o200k_base")
    assert enc.decode(enc.encode("hello world")) == "hello world"

    # To get the tokeniser corresponding to a specific model in the OpenAI API:
    enc = tiktoken.encoding_for_model("gpt-4o")
    print(enc.encode("hello world"))

if __name__ == "__main__":
    main()
