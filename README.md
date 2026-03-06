# Hallucination Detector

**CSC8208 — Multi-Agent Systems | Newcastle University**

A research framework for studying how hallucinations form, spread, and get corrected across networks of LLM agents.

---

## What This Is

Each version builds on the last, testing a new idea in multi-agent hallucination detection.

| Version | Description |
|---------|-------------|
| `v0.1`  | Single primary agent — baseline factual Q&A with response hashing |
| `v0.3`  | Full multi-agent discussion framework — 5 agents, configurable topologies (mesh/ring/star), 5 rounds, convergence + influence + deviation tracking |
| `v0.4`  | Simplified re-evaluation loop — 3 agents, triangle topology, 100 rounds, peer hallucination flagging, dark-themed visualisations |

---

## How It Works (v0.4)

- 3 agents all run `llama3.2` locally via Ollama
- **Round 1** — each agent answers the question independently
- **Rounds 2–100** — each agent sees its own previous answer and both neighbours' answers, then:
  - Re-evaluates and corrects itself if wrong
  - Flags each neighbour as hallucinating (`YES` or `NO`)
- Results saved as JSON + 3 PNG graphs per run

### Topology

```
agent_0 ── agent_1
   \         /
    agent_2
```

Every agent sees every other agent (triangle / fully connected).

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally with `llama3.2` pulled.

```bash
# pull the model
ollama pull llama3.2

# install dependencies
pip install -r requirements.txt

# run v0.4
cd v0.4
python main.py
```

---

## Output

Each run saves timestamped files to `v0.4/results/`:

| File | Contents |
|------|----------|
| `results_YYYYMMDD_HHMMSS.json` | Full raw data — all rounds, answers, verdicts |
| `hallucination_votes_*.png` | How many agents flagged each agent per round |
| `word_counts_*.png` | Answer length per agent per round |
| `answer_changes_*.png` | How many agents changed their answer per round |

---

## Project Structure

```
hallucination_detector/
├── v0.1/
│   └── primary_agent.py
├── v0.3/
│   ├── main.py
│   ├── discussion.py
│   ├── prompt_builder.py
│   ├── agent_registry.py
│   ├── topology_manager.py
│   ├── ollama_client.py
│   ├── response_parser.py
│   ├── convergence_tracker.py
│   ├── influence_tracker.py
│   ├── deviation_tracker.py
│   ├── logger.py
│   ├── result_analyzer.py
│   ├── visualizer.py
│   └── experiment_runner.py
└── v0.4/
    ├── config.py
    ├── experiment.py
    ├── visualizer.py
    ├── main.py
    └── results/
```

---

## Research Questions

- Do agents naturally converge on the correct answer through peer review?
- Does a hallucination in one agent get identified and corrected by its neighbours?
- How does topology affect the spread or correction of hallucinations?
