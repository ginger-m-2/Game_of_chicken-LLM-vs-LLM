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
- Three experimental conditions:
  - **true_persona** — each agent receives its own MBTI prompt
  - **neutral** — every agent receives the same persona-free prompt (baseline control)
  - **shuffled_persona** — each agent receives a different MBTI's prompt via a derangement (control to test whether effects come from persona content vs. prompt structure)
- All decisions, reasoning traces, and metadata logged as JSONL

## Evaluation

- **Behavioral consistency** — Do agents act in line with their MBTI type's expected tendencies?
- **Reasoning trace analysis** — Keyword/sentiment analysis of chain-of-thought outputs to assess personality alignment
- **Strategic performance** — Win rates, escalation/yielding frequencies, mutual worst-outcome rates
- **Trait-level analysis** — Results analyzed along MBTI dimensions (E/I, S/N, T/F, J/P)
- **Baseline comparison** — MBTI-conditioned agents compared against neutral agents to validate that personality prompts produce meaningful behavioral differences

## Stretch Goals

- **Local Llama 3 8B via Ollama** — Port the experiment off the Gemini API onto a locally-hosted Llama 3 8B using Ollama, both to remove API dependencies and to enable the LoRA fine-tuning path below.
- **LoRA fine-tuning** — If time allows, we plan to fine-tune separate LoRA/QLoRA adapters on Llama 3 8B for each MBTI type using personality-consistent text data (e.g., Kaggle MBTI dataset) and [Unsloth](https://github.com/unslothai/unsloth) for efficient training on Google Colab. This would enable a comparison between personality injected through prompting vs. personality internalized through fine-tuning.
- **Prisoner's Dilemma** — Implementing a second game to test whether personality-driven behavioral patterns generalize across different strategic environments.

## Repository Structure

```
src/
  main.py                    # CLI entry point (run-many-tournaments, run-all-conditions)
  agent.py                   # LLM agent with observe-think-act loop
  llm.py                     # LangGraph-based LLM wrapper (Gemini via langchain-google-genai)
  model_adapter.py           # Model backend adapter
  chicken.py                 # Game of Chicken engine and payoff logic
  tournament.py              # Single-elimination tournament orchestration
  mbti_conditions.py         # Condition resolution (true_persona / neutral / shuffled_persona)
  run_experiment.py          # Single-tournament runner
  run_many_tournaments.py    # Multi-tournament driver for statistical runs
  analysis.py                # Statistical analysis helpers
  analyze_results.py         # CLI: summarize a results.jsonl file
  check_results.py           # Sanity checks on logged results
  utils.py                   # Shared utilities

config/
  mbti_profiles.yaml         # MBTI trait definitions and expected behaviors
  payoff_matrix.yaml         # Chicken payoff matrix
  tournament.yaml            # Tournament/model parameters (seed, runs, temperature)

prompts/
  INTJ.txt ... ESFP.txt      # 16 MBTI persona prompt templates
  neutral.txt                # Persona-free baseline prompt
  mbti_prompts.json          # Structured persona prompt bundle (v2.0)
  mbti_prompts.py            # Loader helpers for persona prompts
  game_prompts.py            # Game instruction / decision-request prompts

data/
  results/                   # Experiment logs (JSONL)
  results*.jsonl             # Per-condition run outputs
  results*_meta.json         # Run metadata
```

## Setup

```bash
# Clone the repository
git clone https://github.com/KAsqech/Game_of_chicken-LLM-vs-LLM.git
cd Game_of_chicken-LLM-vs-LLM

# Install dependencies
pip install -r requirements.txt

# Configure the Gemini API key (primary backend: gemini-2.5-flash-lite)
echo "GEMINI_API_KEY=your_key_here" > .env
```

## Running Experiments

```bash
# Validate setup (API key + prompt files) without running
python src/main.py run-many-tournaments --n-tournaments 10 --dry-run

# Run one condition across many tournaments
# (--output is optional; defaults to data/results/<condition>_<timestamp>.jsonl)
python src/main.py run-many-tournaments \
    --n-tournaments 10 \
    --condition true_persona

# Run all three conditions (true_persona, neutral, shuffled_persona)
python src/main.py run-all-conditions --n-tournaments 10

# Summarize a results file
python src/main.py summarize data/results/true_persona_<timestamp>.jsonl
```

The CLI performs preflight checks on startup (missing `GEMINI_API_KEY`, missing
prompt files) and prints per-tournament progress with a ranked champion table
at the end. Set `NO_COLOR=1` to disable ANSI colors.

### Mock mode (no API calls)

For offline smoke tests or when you don't want to spend API credits, force the
deterministic mock backend:

```bash
MOCK_MODEL=1 python src/main.py run-many-tournaments --n-tournaments 2
```

The model adapter prints a clear warning whenever it falls back to the mock
(missing key, auth failure, etc.) so mock results are never silently mistaken
for real ones.
