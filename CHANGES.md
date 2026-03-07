# CHANGES — Research Decision Log

CSC8208 Multi-Agent Hallucination Detection Framework
Newcastle University — MSc Cybersecurity

This file tracks every significant design decision made during development.
Not just what changed, but why — this is the methodology building itself.

---

## [v0.4 — behavioural prompt] 07 Mar 2026

### Decision: removed hallucination prompting from agents entirely

**Before:**
The discussion prompt asked agents to explicitly flag each other:
```
Also, for each other agent, state whether their answer is hallucinating (YES or NO).
```

**After:**
The prompt is now purely:
```
Here is your previous answer: X

Here are other agents' answers:
[agent_0]: ...
[agent_1]: ...

Re-evaluate your answer. If you are wrong, correct it.
```

**Why:**
Asking agents whether others are hallucinating contaminates the behaviour we
are trying to observe. If agents know they are being evaluated for hallucination,
they may respond to the framing rather than the content. The detection must come
entirely from us — not from the agents themselves.

**Impact:**
- `build_discussion_prompt()` simplified — no YES/NO verdict lines
- `parse_response()` simplified — only extracts ANSWER, no verdict parsing
- `ideas.py` unchanged — all 5 detection methods work on answer text alone
- JSON output no longer includes `verdicts` field per agent per round

**Academic basis:**
Behavioural Hallucination Detection — novel contribution. No ground truth,
no NLI, no embeddings, no explicit hallucination prompting. Detection is
purely from observed behavioural patterns across rounds:
  - Stability (who changed first)
  - Flip rate (who kept changing)
  - Convergence (who moved toward majority)
  - Interrogation (who cracked under escalating pressure)
  - Consistency (who gave different answers to the same question phrased differently)

---

## [v0.4 — initial build] 07 Mar 2026

### Decision: rebuild from scratch as v0.4 (simplified architecture)

**Why:**
v0.3 had a complex 3-step JUDGE_PROMPT with scoring rubrics, convergence
trackers, influence trackers, deviation trackers, and a separate logger.
This complexity made it harder to isolate what was actually causing agents
to change their answers.

v0.4 strips everything back to the minimum — 3 agents, triangle topology,
simple re-evaluate prompt, raw JSON output — so behaviour is clean to analyse.

**Files:**
- `config.py` — all settings in one place
- `experiment.py` — round logic, prompt building, answer parsing
- `visualizer.py` — 3 dark-themed plots per run
- `main.py` — entry point
- `ideas.py` — 5 behavioural detection methods (standalone + CLI)
- `run_all.py` — automates all 5 ideas across 3 questions
- `watch_and_run.py` — watches for a run to finish then launches the next

### Decision: triangle topology (3 agents, fully connected)

**Why:**
Each agent sees both other agents. This is the simplest topology where
peer influence is possible. Isolates topology as a variable — all agents
have identical connectivity. Scales up to ring/star/complete in later experiments.

### Decision: 100 rounds

**Why:**
Long enough to observe convergence, divergence, or oscillation.
Short runs (5-10) don't show the full behavioural pattern.
100 rounds gives a clear trajectory for all 5 detection methods.

---

## [v0.3] — previous version

Full multi-agent discussion framework. 5 agents, configurable topologies
(mesh/ring/star), 5 rounds, convergence + influence + deviation tracking,
structured 3-step judge prompt with scoring rubrics.

Key limitation: the JUDGE_PROMPT told agents explicitly what to evaluate
and how to score it — this framed their responses artificially.

---

## [v0.2] — previous version

Full 12-stage detection pipeline: NLI layer (DeBERTa-v3), consensus panel
judge, weighted adaptive fusion (cosine 0.25 / NLI 0.35 / judge 0.40),
blockchain audit log, Byzantine adversarial agent, trajectory tracker,
explanation generator, evaluation suite (HaluEval F1, ROC curve, 5 experiments).

---

## [v0.1] — baseline

Single primary agent + 4 verification agents. Cosine similarity aggregation,
simple voting module, SHA-256 response hashing, JSON audit log.

Known limitation: cosine similarity cannot detect numeric-level hallucinations
(e.g. "Armstrong 1969" vs "Armstrong 1971" scores as highly similar).
This directly motivated the NLI layer in v0.2.
