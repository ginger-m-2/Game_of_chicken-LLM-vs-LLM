from src.llm import LLMAgent

from prompts.mbti_prompts import istj_prompt, isfj_prompt, infj_prompt, intj_prompt, \
  istp_prompt, isfp_prompt, infp_prompt, intp_prompt, estp_prompt, esfp_prompt, enfp_prompt, \
  entp_prompt, estj_prompt, esfj_prompt, enfj_prompt, entj_prompt

from prompts.game_prompts import chicken_game_prompt

estp_agent = LLMAgent(estp_prompt, chicken_game_prompt)
print(estp_agent.get_action()["chosen_action"].content)