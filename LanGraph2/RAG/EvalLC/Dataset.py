from langsmith import Client
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langsmith import wrappers
from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid
examples = [
    # --- Gym History & Founder ---
    {
        "inputs": {"question": "Who founded Peak Performance Gym?"},
        "outputs": {"answer": "Peak Performance Gym was founded by Marcus Chen, a former Olympic athlete with over 15 years of professional athletics experience."},
        "metadata": {"source": "about.txt", "category": "gym_history"},
    },
    {
        "inputs": {"question": "When was Peak Performance Gym established?"},
        "outputs": {"answer": "Peak Performance Gym was founded in 2015."},
        "metadata": {"source": "about.txt", "category": "gym_history"},
    },
    {
        "inputs": {"question": "How large is Peak Performance Gym?"},
        "outputs": {"answer": "The gym spans 10,000 square feet and features state-of-the-art equipment."},
        "metadata": {"source": "about.txt", "category": "gym_history"},
    },

    # # --- Operating Hours ---
    # {
    #     "inputs": {"question": "What are the gym's weekday hours?"},
    #     "outputs": {"answer": "Peak Performance Gym is open Monday through Friday from 5:00 AM to 11:00 PM."},
    #     "metadata": {"source": "hours.txt", "category": "operating_hours"},
    # },
    # {
    #     "inputs": {"question": "Is the gym open on weekends?"},
    #     "outputs": {"answer": "Yes, the gym is open on weekends from 7:00 AM to 9:00 PM."},
    #     "metadata": {"source": "hours.txt", "category": "operating_hours"},
    # },
    # {
    #     "inputs": {"question": "Can I access the gym on public holidays?"},
    #     "outputs": {"answer": "The gym is closed on major national holidays for regular members, but Premium members can access the gym 24/7 using their key cards, including holidays."},
    #     "metadata": {"source": "hours.txt", "category": "operating_hours"},
    # },

    # # --- Membership Plans ---
    # {
    #     "inputs": {"question": "What membership plans are available at Peak Performance Gym?"},
    #     "outputs": {"answer": "The gym offers three plans: Basic at ₹1,500/month (gym floor and basic equipment), Standard at ₹2,500/month (adds group classes and locker facilities), and Premium at ₹4,000/month (24/7 access, personal training, and spa facilities)."},
    #     "metadata": {"source": "membership.txt", "category": "membership"},
    # },
    # {
    #     "inputs": {"question": "Is there a student discount available?"},
    #     "outputs": {"answer": "Yes, students and senior citizens receive a 15% discount on all membership plans."},
    #     "metadata": {"source": "membership.txt", "category": "membership"},
    # },
    # {
    #     "inputs": {"question": "Does the gym offer corporate memberships?"},
    #     "outputs": {"answer": "Yes, corporate partnerships are available for companies with 10 or more employees joining."},
    #     "metadata": {"source": "membership.txt", "category": "membership"},
    # },

    # # --- Fitness Classes ---
    # {
    #     "inputs": {"question": "What group fitness classes does the gym offer?"},
    #     "outputs": {"answer": "The gym offers Yoga (beginner, intermediate, advanced), HIIT, Zumba, Spin Cycling, CrossFit, and Pilates."},
    #     "metadata": {"source": "classes.txt", "category": "fitness_classes"},
    # },
    # {
    #     "inputs": {"question": "When are beginner fitness classes held?"},
    #     "outputs": {"answer": "Beginner classes are held every Monday and Wednesday at 6:00 PM."},
    #     "metadata": {"source": "classes.txt", "category": "fitness_classes"},
    # },
    # {
    #     "inputs": {"question": "Where can I find the full class schedule?"},
    #     "outputs": {"answer": "The full class schedule is available on the gym's mobile app or at the reception desk."},
    #     "metadata": {"source": "classes.txt", "category": "fitness_classes"},
    # },

    # # --- Personal Trainers ---
    # {
    #     "inputs": {"question": "How much does a personal training session cost?"},
    #     "outputs": {"answer": "Individual sessions cost ₹800 each. Package options are available: 10 sessions for ₹7,000 or 20 sessions for ₹13,000."},
    #     "metadata": {"source": "trainers.txt", "category": "personal_training"},
    # },
    # {
    #     "inputs": {"question": "Do new members get a free training session?"},
    #     "outputs": {"answer": "Yes, every new member receives a complimentary fitness assessment and one free session with a personal trainer."},
    #     "metadata": {"source": "trainers.txt", "category": "personal_training"},
    # },
    # {
    #     "inputs": {"question": "Who is the head trainer at Peak Performance Gym?"},
    #     "outputs": {"answer": "The head trainer is Neha Kapoor, who specializes in rehabilitation fitness and sports-specific training."},
    #     "metadata": {"source": "trainers.txt", "category": "personal_training"},
    # },

    # # --- Facilities & Equipment ---
    # {
    #     "inputs": {"question": "Does the gym have a swimming pool?"},
    #     "outputs": {"answer": "Yes, the gym has a 25-meter swimming pool."},
    #     "metadata": {"source": "facilities.txt", "category": "facilities"},
    # },
    # {
    #     "inputs": {"question": "What recovery facilities are available at the gym?"},
    #     "outputs": {"answer": "The gym has sauna and steam rooms available for members."},
    #     "metadata": {"source": "facilities.txt", "category": "facilities"},
    # },
    # {
    #     "inputs": {"question": "How often is the gym equipment upgraded?"},
    #     "outputs": {"answer": "Equipment is replaced or upgraded every 3 years to ensure members have access to the latest fitness technology."},
    #     "metadata": {"source": "facilities.txt", "category": "facilities"},
    # },

    # # --- Off-topic (negative test cases) ---
    # {
    #     "inputs": {"question": "What does Apple Inc. do?"},
    #     "outputs": {"answer": "I'm sorry! I cannot answer this question!"},
    #     "metadata": {"source": "off_topic", "category": "off_topic"},
    # },
    # {
    #     "inputs": {"question": "What is the capital of France?"},
    #     "outputs": {"answer": "I'm sorry! I cannot answer this question!"},
    #     "metadata": {"source": "off_topic", "category": "off_topic"},
    # },

    # # --- Cannot Answer (gap in knowledge base) ---
    # {
    #     "inputs": {"question": "What is the cancellation policy for memberships?"},
    #     "outputs": {"answer": "I'm sorry, but I cannot find the information you're looking for."},
    #     "metadata": {"source": "cannot_answer", "category": "knowledge_gap"},
    # },
    # {
    #     "inputs": {"question": "Is there parking available at the gym?"},
    #     "outputs": {"answer": "I'm sorry, but I cannot find the information you're looking for."},
    #     "metadata": {"source": "cannot_answer", "category": "knowledge_gap"},
    # },
]


client = Client()
dataset_name = "Gym Questions"

# # Storing inputs in a dataset lets us
# # run chains and LLMs over a shared set of examples.
# dataset = client.create_dataset(
#   dataset_name=dataset_name, description="Questions and answers about Gym.",
# )

# # Prepare inputs, outputs, and metadata for bulk creation
# client.create_examples(
#   dataset_id=dataset.id,
#   examples=examples
# )

import openai
from langsmith import wrappers

openai_client = wrappers.wrap_openai(openai.OpenAI())

eval_instructions = "You are an expert professor specialized in grading users' answers to questions."

def correctness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""
Question:
{inputs['question']}

Ground truth answer:
{reference_outputs['answer']}

Model answer:
{outputs['response']}

Respond ONLY with:
CORRECT or INCORRECT
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": eval_instructions},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content

    return response.strip().upper() == "CORRECT"

# 2. Concision
def concision(outputs: dict, reference_outputs: dict) -> bool:
    """
    Checks if answer is short (<= 20 words)
    """
    return len(outputs["response"].split()) <= 20


default_instructions = "Respond to the users question in a short, concise manner (one short sentence)."

def ls_target(inputs: dict) -> dict:
    """
    Runs your existing LangGraph RAG pipeline
    and extracts the final AI response.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "rag",
        os.path.join(os.path.dirname(__file__), "..", "multi_step_rag.py")
    )
    rag = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rag)

    graph = rag.graph
    
     
    result = graph.invoke(
        input={"question": HumanMessage(content=inputs["question"])},
        config={"configurable": {"thread_id": str(uuid.uuid4())}}
    )
    
     # Print the full AgentState returned by the graph
    print("\n=== FULL RAG RESULT ===")
    print(result)

    ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
    response = ai_messages[-1].content if ai_messages else "No response generated"
    
    # Collect the retrieved docs from the graph state
    retrieved_docs = result.get("state", {}).get("documents", [])
    

    return {
        "response": response,
        "retrieved_docs": retrieved_docs
    }


def groundedness(inputs: dict, outputs: dict, reference_outputs: dict) -> bool:
    user_content = f"""
Reference answer:
{reference_outputs['answer']}

Model answer:
{outputs['response']}

Does the model add ANY new facts not present in reference?

Reply ONLY:
GROUNDED or HALLUCINATED
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You detect hallucinations."},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content

    return response.strip().upper() == "GROUNDED"


def hullicination(inputs, outputs, retrieved_docs) -> bool:
    """
    Check if the response contains ONLY facts in the retrieved documents
    """
    retrieved_text = "\n".join([d.page_content for d in retrieved_docs])

    user_content = f"""
Reference documents (source of truth):
{retrieved_text}

AI's answer:
{outputs['response']}

Does the AI add ANY fact not present in the documents?
Reply ONLY: GROUNDED or HALLUCINATED
"""
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        temperature=0,
        messages=[
            {"role": "system", "content": "You are a hallucination detection expert."},
            {"role": "user", "content": user_content},
        ],
    ).choices[0].message.content.strip()

    return response.upper() == "GROUNDED"

# Add all 3 evaluators
experiment_results = client.evaluate(
    ls_target,
    data=dataset_name,
    evaluators= [concision], #[concision, correctness, hullicination],  # ← groundedness added
    experiment_prefix="multi-step-rag",
)

