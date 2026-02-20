# Game of Chicken: MBTI Personality and Strategic Decision-Making in LLM Agents

**CS 4701 — Practicum in Artificial Intelligence**
**Team:** Kamie Aran (ka447), Alif Abdullah (aa2298), Ginger McCoy (gmm225)

## Overview

This project investigates whether MBTI-based persona prompts produce measurably distinct strategic behaviors in LLM agents across classical game-theoretic settings. We use a single fixed open-source model (Llama 3 8B / Mistral 7B) and vary only the persona prompts, isolating the effect of personality from model-level differences.

### Research Questions

1. Do LLM agents prompted with different MBTI types exhibit distinct strategic behaviors?
2. Are personality-driven behavioral patterns consistent across different games?
3. Can systematic prompt optimization improve both role-play fidelity and strategic performance?

## AI Components

- **Behavioral evaluation** — Quantitative metrics and an LLM-as-judge pipeline to assess how faithfully each agent embodies its assigned MBTI type during gameplay.
- **Prompt optimization** — Iterative refinement of persona prompts, measuring the impact of different prompting strategies (trait emphasis, few-shot examples, chain-of-thought) on both fidelity and performance.
- **Multi-game generalization** — Testing agents across three strategically distinct games to determine whether personality-driven patterns are robust or game-specific.

## Games

| Game | Players Choose | Tension |
|------|---------------|---------|
| **Chicken** | Escalate or Yield | Brinksmanship — mutual escalation is the worst outcome, but unilateral escalation wins |
| **Prisoner's Dilemma** | Cooperate or Defect | Self-interest — mutual defection is suboptimal but individually rational |
| **Stag Hunt** | Stag or Hare | Trust — high-payoff cooperation requires coordination |

## Methodology

- 16 LLM agents, each assigned a unique MBTI type, all running on the same base model
- One-shot interactions with fixed payoff matrices per game
- Single-elimination tournament (Chicken) and round-robin (PD, Stag Hunt) formats
- 15–20 tournament iterations with randomized pairings for statistical power
- Control experiments with neutral (no persona) and shuffled-persona agents
- All outcomes, reasoning traces, and metadata logged in structured format (JSON/SQLite)

MBTI is used as a structured personality abstraction rather than a validated psychological model.

## Evaluation

### Role-Play Fidelity
- **Behavioral consistency scoring** — comparing agent actions against expected MBTI tendencies (e.g., ENTJ should escalate more; ISFP should cooperate more)
- **Reasoning trace analysis** — keyword/sentiment analysis + LLM-as-judge assessment of chain-of-thought outputs
- **Baseline comparisons** — neutral agents and shuffled persona labels as controls

### Strategic Performance
- Win rates, escalation/cooperation frequencies, mutual worst-outcome rates
- Analysis along each MBTI dimension (E/I, S/N, T/F, J/P)
- Cross-game behavioral pattern comparison

### Prompt Optimization
- Performance metrics tracked across prompt iterations
- Measuring the tradeoff between role-play fidelity and strategic success

## Repository Structure

```
src/
  agent.py              # LLM agent with observe-think-act loop
  chicken.py            # Game of Chicken engine
  prisoners_dilemma.py  # Prisoner's Dilemma engine
  stag_hunt.py          # Stag Hunt engine
  tournament.py         # Tournament orchestration (bracket + round-robin)
  evaluation.py         # Role-play fidelity metrics and analysis
  analysis.py           # Statistical analysis and visualization
  main.py               # Entry point

config/
  mbti_profiles.yaml    # MBTI trait definitions and expected behaviors
  payoff_matrix.yaml    # Payoff matrices for all three games
  model_config.yaml     # Base model parameters (temperature, tokens, etc.)

prompts/
  mbti_prompts/         # 16 versioned persona prompt templates

data/
  results/              # Tournament logs (JSON/SQLite)
  analysis/             # Derived statistics and visualizations
```

## Setup

```bash
# Clone the repository
git clone https://github.com/ginger-m-2/Game_of_chicken-LLM-vs-LLM.git
cd Game_of_chicken-LLM-vs-LLM

# Install dependencies
pip install -r requirements.txt

# Install and pull the base model (via Ollama)
ollama pull llama3:8b
```