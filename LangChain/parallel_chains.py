from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

def dictionary_maker(text: str) -> dict:
    return {"text": text}


def insta_chain(text: dict):  

    text  = text["text"
                 ]  
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

def main():
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "You are a movie summarizer"),
        ("human", "Please summarize the movie in brief : {input}")
    ])

    llm = ChatOpenAI(model="gpt-5-mini", temperature=1, timeout=60, max_retries=3)

    parser = StrOutputParser()

    #Parallel chain

    linkedin_prompt = ChatPromptTemplate.from_messages([
        ("system", "You're a linked social media post generator."),
        ("human", "Create a post for the following text for LinkedIn: {text}")
    ])
    
    dictionary_maker_runnable = RunnableLambda(dictionary_maker)

    insta_chain_runnable = RunnableLambda(insta_chain)

    chain_linkedIn  = linkedin_prompt | llm | parser

    final_chain = ( prompt_template | 
                    llm | 
                    parser | 
                    dictionary_maker_runnable 
                    | RunnableParallel(branches={ "linkedin": chain_linkedIn, "instagram": insta_chain_runnable})
                  )
    
    result = final_chain.invoke({"input": "Inception"})

    print(result)
    

if __name__ == "__main__":
    main()