# Hallucination Detector

**CSC8208 — Multi-Agent Systems | Newcastle University**

A research framework for studying how hallucinations form, spread, and get corrected across networks of LLM agents.

---

## What This Is

Each version builds on the last, testing a new idea in multi-agent hallucination detection.

| Version | Description |
|---------|-------------|
| `v0.1`  | Single primary agent — baseline factual Q&A with response hashing |
| `v0.2`  | Full 12-stage detection pipeline — NLI contradiction layer (DeBERTa-v3), consensus panel judge, weighted adaptive fusion, blockchain audit log, Byzantine adversarial agent, trajectory tracker |
| `v0.3`  | Multi-agent discussion framework — 5 agents, configurable topologies (mesh/ring/star), 5 rounds, convergence + influence + deviation tracking |
| `v0.4`  | Simplified re-evaluation loop — 3 agents, triangle topology, 100 rounds, pure behavioural detection, 5 detection metrics |
| `v0.5`  | Ground truth validation — 5 factual questions, 10 rounds, dual fact checking (keyword + NLI via DeBERTa), hallucination heatmaps |
| `v0.6.1`| CSV-driven batch experiment — ROT13/Caesar cipher challenges, 5 rounds, manual annotation workflow |

---

## How It Works (v0.6.1)

- 3 agents all run `llama3.2` locally via Ollama
- questions are loaded from `questions.csv` — one experiment per question
- **Round 1** — each agent answers independently (cold start)
- **Rounds 2–5** — each agent sees its own previous answer and both neighbours' answers, then re-evaluates
- after each run, a blank `annotations.json` is generated for manual ground truth labelling

### Topology

```
agent_0 ── agent_1
   \         /
    agent_2
```

Every agent sees every other agent (triangle / fully connected).

### Manual Annotation Workflow

```
python experiment.py          # runs discussion, saves results.json + blank annotations.json
  → open annotations.json     # read each answer, set hallucinating: true or false
  → python plot.py            # generates plots from your annotations
```

---

## How It Works (v0.5)

- 5 factual questions, each with curated ground truth facts
- 10-round discussion (same clean re-evaluate prompt as v0.4)
- after discussion, two fact checkers run independently:
  - keyword matching — fast, rule-based
  - NLI via DeBERTa — semantic, model-based
- agents never see the ground truth — detection is purely post-hoc
- output: JSON results + hallucination heatmap PNG per question

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com) running locally with `llama3.2` pulled.

```bash
# pull the model
ollama pull llama3.2

# start ollama (must be running before any experiment)
ollama serve

# install dependencies
pip install -r requirements.txt
```

---

## Running

```bash
# v0.6.1 — current version, CSV-driven batch
cd v0.6.1 && python experiment.py

# v0.5 — ground truth fact checking across 5 questions
cd v0.5 && python main.py

# v0.4 — 100-round behavioural detection, single question
cd v0.4 && python main.py

# v0.4 — batch all 5 detection ideas across 3 questions
cd v0.4 && python run_all.py
```

---

## Output

### v0.6.1

Results saved to `v0.6.1/results/` per question:

| File | Contents |
|------|----------|
| `q00N_results.json` | full discussion — all rounds, all answers |
| `q00N_annotations.json` | blank template — fill in `hallucinating: true/false` per agent per round |

### v0.5

Results saved to `v0.5/results/` per question:

| File | Contents |
|------|----------|
| `qN_slug_timestamp.json` | discussion results + ground truth evaluation |
| `qN_slug_timestamp.png` | hallucination heatmap — agent × round |

### v0.4

Results saved to `v0.4/results/` (timestamped):

| File | Contents |
|------|----------|
| `results_YYYYMMDD_HHMMSS.json` | full raw data — all rounds, answers |
| `hallucination_votes_*.png` | how many agents flagged each agent per round |
| `word_counts_*.png` | answer length per agent per round |
| `answer_changes_*.png` | how many agents changed their answer per round |

---

## Research Questions

- do agents naturally converge on the correct answer through peer review?
- does a hallucination in one agent get identified and corrected by its neighbours?
- can hallucination be detected from behaviour alone — without ground truth, NLI, or embeddings?
- how does topology affect the spread or correction of hallucinations?
