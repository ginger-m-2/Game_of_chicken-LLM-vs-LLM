# File to host LLM StateGraph code

from langgraph.graph import StateGraph
from typing import Dict, Union, List, Any, TypedDict
from langchain.chat_models import init_chat_model

llm_agent_model_string = "" # Fill in desired model we want!
external_evidence_researcher_llm = init_chat_model(model=llm_agent_model_string)

class ChickenAgentState(TypedDict):
    prompt: str
    context: str

def decision_node(state: ChickenAgentState):
    pass