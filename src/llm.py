# File to host LLM StateGraph code

from langgraph.graph import StateGraph
from typing import Dict, Union, List, Any, TypedDict, Literal
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

llm_agent_model_string = "" # Fill in desired model we want!

# DOES NOT WORK AS OF NOW - NEED TO ACTUALLY INSERT VARIABLES INTO THE GAME PROMPT
class LLMAgent:

    def __init__(self, mbti_prompt, game_prompt, context, game_type):
        self.mbti_prompt = mbti_prompt
        self.game_prompt = game_prompt
        self.context = context
        self.chicken_action_prompt = "FILL IN WITH PROMPT THAT MAKES AGENT CHOOSE ONE VALID ACTION ONLY"
        self.prisoner_action_prompt = "FILL IN WITH PROMPT THAT MAKES AGENT CHOOSE ONE VALID ACTION ONLY"
        self.game_type: Literal["chicken", "prisoner"] = game_type
        self.mbti_llm = init_chat_model(model=llm_agent_model_string)

    class GameAgentState(TypedDict):
        mbti_prompt: str
        game_prompt: str
        context: dict
        chosen_action: str

    def _decision_node(self, state: GameAgentState):
        mbti_prompt = state["mbti_prompt"]
        game_prompt = state["game_prompt"]

        # Format the game prompt by inserting variables that detail the state of the game.
        #FIXME: REPLACE THIS BY ACTUALLY INSERTING VARIABLES HERE
        formatted_game_prompt = game_prompt.format(
            game_context_var=""
        )

        # (If your MBTI comes from prompt and not fine tuning) Combine the game prompt and mbti prompt together.

        combined_prompt = formatted_game_prompt
        if mbti_prompt is not None:
            combined_prompt =  "\n\n\n\n" + mbti_prompt

        # Have a third prompt (based on the game prompt) that asks the LLM to take an action 
        # - based on the context it has already received on the game and the MBTI personality of the 
        # - agent. Be VERY strict that the LLM has to return one word, and one word only - the action it chose.
        # (Note: may want to include a rational and then an action, as if to say here are my thoughts on the game,
        # - and thus here is what I want to do)
        agent_action_context_window = [SystemMessage(content=combined_prompt)]

        agent_action_prompt = None

        if self.game_type == "chicken":
            agent_action_prompt = self.chicken_action_prompt
        elif self.game_type == "prisoner":
            agent_action_prompt = self.prisoner_action_prompt
        
        agent_action_context_window.append(HumanMessage(content=agent_action_prompt + "\n\nFollow system instructions and return the chosen action."))

        # Get action from the LLM, and return it to the outside world.
        return {
            "chosen_action": self.mbti_llm.invoke(agent_action_context_window)
        } 



    def get_action(self):
        # Compile the agent stage graph here, invoke it, and return the action to the outside world.
        pass

