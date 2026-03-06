# CHANGELOG
## Hallucination Detection Framework
### CSC8208 | Newcastle University | MSc Cybersecurity

---

## [v0.2] — In Development

### Why v0.2 Exists
Version 0.1 established a working baseline but had one
documented limitation: cosine similarity alone cannot detect
numeric-level hallucinations (e.g. correct entity, wrong year).
v0.2 directly addresses this and significantly extends the
system's detection capability, adversarial resilience, and
academic rigour.

---

### New Files

**agent_registry.py**
- Defines all 10 agents with model, temperature, style, role
- Supports provider field for future API extension
  (openai, anthropic, grok, google, mistral, cohere)
- Byzantine adversarial agent defined here
- Justification: 5 real model architectures decorrelate
  outputs more effectively than prompt variation alone

**topology_manager.py**
- Implements star, ring, and complete graph topologies
- Controls information flow between agents
- Used in Byzantine resilience experiment:
  "Does topology affect malicious agent influence spread?"
- Academic basis: network science + distributed systems theory

**nli_layer.py**
- Natural Language Inference using DeBERTa-v3 (HuggingFace)
- Labels each verifier vs primary: ENTAILMENT/NEUTRAL/CONTRADICTION
- Directly addresses v0.1 limitation — catches numeric errors
  cosine similarity missed (Armstrong 1969 vs Armstrong 1971)
- Independent verdict: ACCEPT / FLAG / REGENERATE
- Weight in fusion: 0.35

**judge_layer.py**
- Consensus panel judge system
- Every agent judges every other agent (temp 0.3)
- Structured JSON output — replaces fragile regex extraction
- Scores: factual accuracy, hallucination, completeness,
  reasoning quality, verdict, written justification
- Aggregated across all judges for consensus verdict
- Weight in fusion: 0.40 (main layer — highest weight)
- Academic basis: Zheng et al. (2023) arXiv:2306.05685
- Extension: single judge → consensus panel to eliminate bias

**cross_validation.py**
- Compares all 3 independent layer verdicts
- Determines confidence: HIGH / MODERATE / LOW
- Infers hallucination TYPE from agreement pattern:
    Cosine only     → semantic divergence
    NLI only        → factual contradiction
    Judge only      → reasoning flaw
    Cosine + NLI    → clear factual error
    All 3           → confirmed hallucination
- Novel contribution: no existing paper characterises
  hallucination type via layer cross-validation

**fusion_module.py**
- Replaces voting_module.py
- Weighted adaptive fusion of all 3 layer scores:
    final_risk = 0.25×cosine + 0.35×nli + 0.40×panel
- Confidence weighting applied on top of layer weights
- Verdict fusion: majority of layer verdicts wins

**decision_engine.py**
- Clean separation of decision logic from fusion
- Thresholds: 0.00-0.20 ACCEPT, 0.20-0.45 FLAG,
  0.45-1.00 REGENERATE

**trajectory_tracker.py**
- Records risk score after each detection layer
- Produces confidence trajectory per pipeline run
- Feeds Figure 2 (line graph) in evaluation
- Proves layers are complementary not redundant

**explanation_generator.py**
- Generates plain English explanation for every decision
- Includes: which layers fired, scores, hallucination type,
  regeneration outcome, improvement delta
- Makes system interpretable to end users

**blockchain_logger.py**
- Local Python blockchain — no Ethereum required
- Genesis block on system initialisation
- One block per pipeline run: all scores, hashes,
  decisions, trajectory, regeneration outcome
- SHA-256 chain hash links blocks sequentially
- Chain validation detects any tampering
- Justified over distributed ledger: same cryptographic
  guarantee, no network infrastructure needed for
  single-machine local deployment
- Satisfies CSC8208 integrity + audit requirements

**evaluate.py**
- Runs all 5 evaluation experiments:
    1. HaluEval F1 scoring (50 questions)
    2. v0.1 vs v0.2 comparison
    3. Byzantine fault ratio (0,1,2,3 agents × 3 topologies)
    4. Regeneration study (which models self-correct)
    5. ROC curve across thresholds
- Produces all 7 figures and 4 tables
- Model reliability leaderboard

---

### Modified Files

**verification_agents.py**
- Extended from 4 to 9 verification agents
- Added temperature variation (0.2 to 0.9) across agents
- Added Byzantine adversarial agent (agent_09)
  forces confidently wrong answers via system prompt
- Added 5 real different model architectures
- Reason: decorrelates outputs more effectively,
  stronger divergence signal when hallucination occurs

**cosine_layer.py** (was aggregation_module.py)
- Renamed to reflect its role as Layer 1 only
- Added variance calculation alongside mean
- Added independent verdict output
- High variance = agents wildly inconsistent =
  stronger hallucination signal than mean alone
- Weight in fusion: 0.25

**regeneration_module.py**
- Now receives panel judge scores and justifications
- Correction prompt now includes:
    all 9 agent answers
    judge score per agent
    written justification
    cross-validation pattern
    hallucination type identified
- Re-scores corrected answer through ALL 3 layers
- Measures improvement delta per layer
- Tracks which models self-correct vs refuse
- Directly answers Mujeeb's research question:
  "Do LLMs actually correct themselves when told
   they hallucinated? Which ones? By how much?"

**main.py**
- Updated to orchestrate full v0.2 pipeline
- Includes topology selection via CLI argument
- Saves to blockchain logger instead of simple JSON

---

### Unchanged Files

**primary_agent.py** — no changes required

---

### Key Design Decisions

**Why consensus panel judge over single judge?**
A single judge inherits that model's biases and failure modes.
If the judge itself hallucinates, the verdict is wrong.
A consensus panel where every model judges every other model
eliminates individual bias — the same principle that justifies
peer review in academic publishing. Extends Zheng et al. (2023)
from single-judge to consensus-panel architecture.

**Why local blockchain over Ethereum?**
All agents run on a single local machine. A distributed ledger
solves a distributed trust problem. Our threat model is integrity
of the audit record on a single deployment. SHA-256 chain hashing
provides identical tamper-evidence guarantees without distributed
infrastructure overhead. In a production multi-machine deployment,
this would be replaced with a distributed ledger.

**Why 0.25 / 0.35 / 0.40 fusion weights?**
Judge layer receives highest weight (0.40) because it provides
the most interpretable and semantically rich signal — written
justification and multi-dimensional scoring. NLI receives 0.35
because it is purpose-built for contradiction detection and
catches the numeric errors cosine misses. Cosine receives 0.25
as a fast first-pass filter. These weights are hyperparameters
and their sensitivity is tested in evaluation.

**Why neutral topics only?**
Per supervisor guidance (Mujeeb): ethical compliance requires
avoiding politically sensitive or controversial topics. All
test questions use science, history, geography, and climate.

---

## [v0.1] — Complete (Baseline)

### Files
- primary_agent.py
- verification_agents.py (4 agents)
- aggregation_module.py
- voting_module.py
- regeneration_module.py
- main.py
- test_hallucination.py
- audit_log.json

### Evaluation Results
- True Positive (Buzz Aldrin 1971): FLAG ✅ detected
- True Negative (Armstrong 1969): ACCEPT ✅
- Partial hallucination (Armstrong 1971): ACCEPT ⚠️ missed

### Known Limitation
Cosine similarity cannot detect numeric-level hallucinations.
Semantic embeddings score "Armstrong 1971" as highly similar
to "Armstrong 1969" because the entity and event are identical.
Only the numeric value differs — invisible to cosine distance.
This directly motivated NLI layer in v0.2.

---