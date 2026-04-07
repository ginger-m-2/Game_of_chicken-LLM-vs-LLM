# File to host LLM StateGraph code
"""
llm.py

Provides a unified interface for language model inference.

This module abstracts backend-specific implementation details
(e.g., Ollama, Hugging Face, vLLM) and exposes a standardized
generation function for use by Agent instances.

It handles:
    - Prompt submission
    - Temperature and token controls
    - Optional reproducibility seeds
    - Timeouts and backend errors

This file centralizes LLM interactions to ensure consistency
across conditioning methods.
"""
from langgraph.graph import StateGraph
from typing import Dict, Union, List, Any, TypedDict, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from dotenv import load_dotenv
import os 

load_dotenv(override=True)

os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

llm_agent_model_string = "gemini-2.5-flash-lite"

# DOES NOT WORK AS OF NOW - NEED TO ACTUALLY INSERT VARIABLES INTO THE GAME PROMPT
class LLMAgent:

    def __init__(self, mbti_prompt, game_prompt):
        self.mbti_prompt = mbti_prompt
        self.game_prompt = game_prompt
        self.mbti_llm = init_chat_model(model=llm_agent_model_string, model_provider="google_genai")

    class GameAgentState(TypedDict):
        mbti_prompt: str
        game_prompt: str
        context: dict
        chosen_action: str

    def _decision_node(self, state: GameAgentState):
        mbti_prompt = state["mbti_prompt"]
        game_prompt = state["game_prompt"]

        formatted_game_prompt = game_prompt.format(
            mbti=mbti_prompt
        )

        # (If your MBTI comes from prompt and not fine tuning) Combine the game prompt and mbti prompt together.

        # Have a third prompt (based on the game prompt) that asks the LLM to take an action 
        # - based on the context it has already received on the game and the MBTI personality of the 
        # - agent. Be VERY strict that the LLM has to return one word, and one word only - the action it chose.
        # (Note: may want to include a rational and then an action, as if to say here are my thoughts on the game,
        # - and thus here is what I want to do)
        agent_action_context_window = [SystemMessage(content=formatted_game_prompt), HumanMessage(content="Follow system instructions and ONLY RETURN THE CHOSEN ACTION AND NOTHING ELSE.")]

        # Get action from the LLM, and return it to the outside world.
        return {
            "chosen_action": self.mbti_llm.invoke(agent_action_context_window)
        } 

    def _generate_input(self) -> GameAgentState:
        return {
            "mbti_prompt":self.mbti_prompt,
            "game_prompt":self.game_prompt,
            "chosen_action":""
        }


    def get_action(self):
        agent_builder = StateGraph(self.GameAgentState)

        agent_builder.add_node("decision_node", self._decision_node)

        agent_builder.set_entry_point("decision_node")
        agent_builder.set_finish_point("decision_node")

        agent_graph = agent_builder.compile()

        input_state = self._generate_input()

        return agent_graph.invoke(input_state)


