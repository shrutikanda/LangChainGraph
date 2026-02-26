from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Literal

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


class llm_schema(BaseModel):
    userQuery: str
    movie_summary_flag: Literal["positive", "negative"]

def pydantic_json(text: llm_schema) -> dict:
    return {
        "text": text.userQuery,
        "sentiment": text.movie_summary_flag,
    }

def insta_chain(text: dict):  

    text  = text["text"]                   
    insta_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a instagram social media post generator."),
        ("human", "Create a post for the following text for Instagram: {text}")
    ])


    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=1,
        timeout=20,        # lower timeout so it doesn't hang forever
        max_retries=2,
    )

    parser = StrOutputParser()

    chain_insta  = insta_prompt | llm | parser

    result = chain_insta.invoke({"text": text})

    return result


def linked_chain(text: dict):  

    text  = text["text"]  
    linked_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a linkedin social media post generator."),
        ("human", "Create a post for the following text for LinkedIn: {text}")
    ])


    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=1,
        timeout=20,        # lower timeout so it doesn't hang forever
        max_retries=2,
    )

    parser = StrOutputParser()

    chain_linked  = linked_prompt | llm | parser

    result = chain_linked.invoke({"text": text})

    return result

def main():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie review evaluator."),
        ("human", "Please categorize the review as positive, negative, or neutral: {text}")
    ]) 

    llm = ChatOpenAI(
        model="gpt-5-mini",
        temperature=1,
        timeout=20,        # lower timeout so it doesn't hang forever
        max_retries=2,
    ) 
    llm_structured_output =llm.with_structured_output(llm_schema)

    pydantic_json_lambda  = RunnableLambda(pydantic_json)     
    
    insta_chain_runnable = RunnableLambda(insta_chain)
    linked_chain_runnable = RunnableLambda(linked_chain)   

    chain_linkedin = linked_chain_runnable
    chain_instagram = insta_chain_runnable

    #Conditional chain

    conditional_chain = RunnableBranch(
        (lambda x: x.get("sentiment") == "positive", chain_linkedin),
        chain_instagram
    )
    
    final_orchestrator = prompt_template | llm_structured_output | pydantic_json_lambda | conditional_chain

    ans = final_orchestrator.invoke({"text": "I loved this KGF movie"})

    print(ans)

  

if __name__ == "__main__":
    main()