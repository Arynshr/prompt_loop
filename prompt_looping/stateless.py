from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from typing import TypedDict
from dotenv import load_dotenv
import os

class LoopState(TypedDict):
    task: str
    draft: str
    critique: str
    iteration: int

load_dotenv()
GROK_API_KEY = os.getenv("GROQ_API_KEY")
if not GROK_API_KEY:
    raise ValueError("Missing GROQ api key in .env")

llm = ChatGroq(
    model = "llama3-8b-8192",
    temperature = 0.3
)

generate_prompt =  ChatPromptTemplate.from_Template(
    "You are a research assistant, generate research grade article for:\n {task}"
)

critique_prompt = ChatPromptTemplate.from_Template(
    "You are a strict evaluator. List weaknesses. \nAnswer: \n {draft}"
)

refine_prompt = ChatPromptTemplate.from_Template(
    """
    Improve the answer using only critique.
    Donot rewrite the whole text from scratch 
    Answer:
    {draft}
    Critique:
    {critique}
    """
)

generate_chain = generate_prompt | llm
critique_chain = critique_prompt | llm
refine_chain = refine_prompt | llm

def stateless_loop(task: str, max_iters = 3):
    state : LoopState = {
        "task" : task,
        "draft" : "",
        "critique" : "",
        "iteration" : 0 
    }
    state["draft"] = generate_chain.invoke({"task":task}).content
    
    while state["iteration"] < max_iters:
        state["iteration"] += 1
        
        prev = state["draft"]
        
        state["critique"] = critique_chain.invoke({
            "draft" : state["draft"]
        }).content
        
        state["draft"] = refine_chain.invoke({
            "draft" : state["draft"],
            "critique": state["critique"]
        }).content
        
        if state["draft"].strip() == prev.strip():
            print("Converged early")
            break
            
        print(f"Iteration {state['Iteration']} complete")
        
    return state["draft"]
