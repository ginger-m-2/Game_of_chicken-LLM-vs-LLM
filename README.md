# Game of Chicken: MBTI Personality and Strategic Decision-Making in LLM Agents

**CS 4701 — Practicum in Artificial Intelligence**
**Team:** Kamie Aran (ka447), Alif Abdullah (aa2298), Ginger McCoy (gmm225)

## Overview

This project investigates whether fine-tuning LLM agents on MBTI personality-typed text data produces more behaviorally distinct and strategically consistent agents than persona prompting alone. We use Llama 3 8B as a fixed base model and train separate LoRA/QLoRA adapters for each MBTI type, then evaluate agents in classical game-theoretic settings.

### Research Question

**Does fine-tuning produce more consistent and distinct agent behavior than prompting alone, and does that translate to measurably different strategic outcomes?**

We wanted to go beyond the observation that different prompts produce different outputs. Fine-tuned agents internalize personality traits at the weight level rather than relying on instruction-following, which may produce deeper behavioral consistency, or it may not.

## AI Components

- **LoRA/QLoRA fine-tuning** — Training separate lightweight adapters for each MBTI type on personality-consistent text data (e.g., Kaggle MBTI dataset, MBTI subreddit posts). This is the core ML training component of the project.
- **Fine-tuned vs. prompt-only comparison** — Running identical tournaments with fine-tuned agents and prompt-only agents to quantify the difference in behavioral consistency and strategic performance.
- **Behavioral evaluation pipeline** — Quantitative metrics to assess role-play fidelity, including behavioral consistency scoring, reasoning trace analysis, and baseline controls.

## Games

| Game | Players Choose | Tension |
|------|---------------|---------|
| **Chicken** (primary) | Escalate or Yield | Brinksmanship — mutual escalation is the worst outcome, but unilateral escalation wins |
| **Prisoner's Dilemma** (stretch) | Cooperate or Defect | Self-interest — mutual defection is suboptimal but individually rational |

We focus on Chicken as the primary game to keep the fine-tuning workload manageable. Prisoner's Dilemma is a stretch goal for cross-game comparison.

## Methodology

### Fine-Tuning Pipeline
1. **Data collection** — Gather personality-typed text from the Kaggle MBTI dataset (~8,600 users with labeled forum posts) and/or MBTI subreddit posts. Clean and format as instruction-tuning data.
2. **LoRA adapter training** — For each MBTI type (starting with 4 representative types, expanding to 16 if time allows), fine-tune a separate LoRA adapter on Llama 3 8B using Hugging Face PEFT + QLoRA for memory efficiency.
3. **Validation** — Verify that fine-tuned adapters produce personality-consistent text outside of the game context before running tournaments.

### Tournament Design
- 16 agents (one per MBTI type), all sharing the same Llama 3 8B base with different LoRA adapters
- Parallel tournament with prompt-only agents (same base model, MBTI persona in system prompt, no fine-tuning) as a control group
- One-shot interactions with a fixed payoff matrix
- Single-elimination bracket, 15–20 iterations with randomized pairings
- All decisions, reasoning traces, and metadata logged (JSON/SQLite)

MBTI is used as a structured personality abstraction rather than a validated psychological model.

## Evaluation

### Role-Play Fidelity
- **Behavioral consistency scoring** — Do agents act in line with their MBTI type's expected tendencies? (e.g., ENTJ escalates more, ISFP cooperates more)
- **Reasoning trace analysis** — Keyword/sentiment analysis of chain-of-thought outputs to assess personality alignment
- **Fine-tuned vs. prompted comparison** — Are fine-tuned agents more consistent in their personality-typed behavior than prompted agents?
- **Baseline controls** — Neutral (no persona) agents and shuffled persona labels

### Strategic Performance
- Win rates, escalation frequencies, mutual worst-outcome rates
- Analysis along MBTI dimensions (E/I, S/N, T/F, J/P)
- Comparative performance: fine-tuned vs. prompt-only agents

## Repository Structure

```
src/
  agent.py              # LLM agent with observe-think-act loop
  chicken.py            # Game of Chicken engine
  tournament.py         # Tournament orchestration and bracket logic
  evaluation.py         # Role-play fidelity metrics
  analysis.py           # Statistical analysis and visualization
  main.py               # Entry point

fine_tuning/
  prepare_data.py       # Data cleaning and formatting for LoRA training
  train_lora.py         # LoRA/QLoRA fine-tuning script
  validate_adapter.py   # Personality consistency validation for adapters

config/
  mbti_profiles.yaml    # MBTI trait definitions and expected behaviors
  payoff_matrix.yaml    # Payoff matrices
  model_config.yaml     # Base model and LoRA hyperparameters

prompts/
  mbti_prompts/         # 16 persona prompt templates (for prompt-only control group)

data/
  training/             # MBTI-typed text data for fine-tuning
  results/              # Tournament logs (JSON/SQLite)
  analysis/             # Derived statistics and visualizations
```

## Setup

```bash
# Clone the repository
git clone https://github.com/KAsqech/Game_of_chicken-LLM-vs-LLM.git
cd Game_of_chicken-LLM-vs-LLM

# Install dependencies
pip install -r requirements.txt

# Key dependencies:
#   torch, transformers, peft, bitsandbytes (for QLoRA)
#   ollama or vllm (for inference)
#   pandas, numpy, scipy, matplotlib

# Pull the base model
ollama pull llama3:8b
```