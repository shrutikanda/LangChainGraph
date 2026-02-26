from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def dictionary_maker(text: str) -> dict:
    return {"text": text}

dictionary_maker_runnable = RunnableLambda(dictionary_maker)

def main():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant"),
        ("human", "{input}")
    ])

    prompt_post = ChatPromptTemplate.from_messages([
        ("system", "You're a social media post generator."),
        ("human", "Create a post for the following text for LinkedIn: {text}")
    ])

    llm = ChatOpenAI(model="gpt-5-mini", temperature=1, timeout=60, max_retries=3)
    parser = StrOutputParser()

    full_chain = prompt_template | llm | parser | dictionary_maker_runnable | prompt_post | llm | parser

    result = full_chain.invoke({"input": "What is the capital of France?"})
    print(result)

if __name__ == "__main__":
    main()