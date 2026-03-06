# Game of Chicken: LLM vs LLM

**CS 4701 — Practicum in Artificial Intelligence**
**Team:** Kamie Aran (ka447), Alif Abdullah (aa2298), Ginger McCoy (gmm225)

## Overview

We are building a tournament for the game of Chicken played between LLM agents, where each agent is assigned a different MBTI personality type, in order to determine which traits correlate with victory in a strategic social game.

## Methodology

- LLM-based agents with fixed MBTI personality prompts
- One-shot Game of Chicken interactions
- Fixed payoff matrix
- Randomized single-elimination tournament
- Controlled, multi-run experiments comparing MBTI-conditioned agents against a neutral baseline
- Outcomes logged and analyzed for strategic and payoff-level differences

MBTI is used as a structured personality abstraction rather than a validated psychological model.

## Games

| Game | Players Choose | Tension |
|------|---------------|---------|
| **Chicken** (primary) | Escalate or Yield | Brinksmanship — mutual escalation is the worst outcome, but unilateral escalation wins |

## Experimental Design

- 16 LLM agents, each assigned a unique MBTI type
- Agents are prompted with MBTI persona descriptions via system prompts to guide their decision-making behavior
- One-shot interactions with a fixed payoff matrix
- Multiple tournament runs with randomized pairings for statistical significance
- Neutral (no persona) agents as a baseline control
- All decisions, reasoning traces, and metadata logged in structured format (JSON/SQLite)

## Evaluation

- **Behavioral consistency** — Do agents act in line with their MBTI type's expected tendencies?
- **Reasoning trace analysis** — Keyword/sentiment analysis of chain-of-thought outputs to assess personality alignment
- **Strategic performance** — Win rates, escalation/yielding frequencies, mutual worst-outcome rates
- **Trait-level analysis** — Results analyzed along MBTI dimensions (E/I, S/N, T/F, J/P)
- **Baseline comparison** — MBTI-conditioned agents compared against neutral agents to validate that personality prompts produce meaningful behavioral differences

## Stretch Goals

- **LoRA fine-tuning** — If time allows, we plan to fine-tune separate LoRA/QLoRA adapters on Llama 3 8B for each MBTI type using personality-consistent text data (e.g., Kaggle MBTI dataset) and [Unsloth](https://github.com/unslothai/unsloth) for efficient training on Google Colab. This would enable a comparison between personality injected through prompting vs. personality internalized through fine-tuning.
- **Prisoner's Dilemma** — Implementing a second game to test whether personality-driven behavioral patterns generalize across different strategic environments.

## Repository Structure

```
src/
  agent.py              # LLM agent with observe-think-act loop
  chicken.py            # Game of Chicken engine
  tournament.py         # Tournament orchestration and bracket logic
  evaluation.py         # Role-play fidelity metrics and payoff analysis
  analysis.py           # Statistical analysis and visualization
  main.py               # Entry point

config/
  mbti_profiles.yaml    # MBTI trait definitions and expected behaviors
  payoff_matrix.yaml    # Payoff matrices
  model_config.yaml     # Model parameters (temperature, tokens, etc.)

prompts/
  mbti_prompts/         # 16 persona prompt templates

data/
  results/              # Experiment logs (JSON/SQLite)
  analysis/             # Derived statistics and visualizations
```

## Setup

```bash
# Clone the repository
git clone https://github.com/KAsqech/Game_of_chicken-LLM-vs-LLM.git
cd Game_of_chicken-LLM-vs-LLM

# Install dependencies
pip install -r requirements.txt

# Pull the base model
ollama pull llama3:8b
```
