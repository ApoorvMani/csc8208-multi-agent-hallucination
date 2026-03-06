"""
cosine_layer.py
===============
Layer 1 of the triple-layer hallucination detection pipeline.

Computes semantic similarity between the primary agent's answer
and all 9 verification agent answers using sentence embeddings
and cosine similarity.

Improvements over v0.1 aggregation_module.py:
  - Adds VARIANCE alongside mean similarity
    High variance = agents wildly inconsistent = stronger signal
  - Produces independent verdict (ACCEPT/FLAG/REGENERATE)
    not just a score — feeds cross-validation module
  - Per-agent similarity scores retained for heatmap (Figure 1)
  - Named cosine_layer.py to reflect its role as Layer 1 only

Why cosine similarity?
  Two semantically equivalent answers can be expressed in
  entirely different words. Cosine similarity on sentence
  embeddings captures meaning rather than surface word overlap.
  (Reimers & Gurevych, 2019 — Sentence-BERT)

Why all-MiniLM-L6-v2?
  Strong performance on semantic textual similarity benchmarks,
  fast inference, small footprint (80MB). Well suited for
  local deployment without GPU requirements.

What cosine catches:
  Semantically divergent answers — completely wrong person,
  completely different topic, obviously incorrect claims.

What cosine misses:
  Numeric-level hallucinations where surrounding text is
  correct but a specific number or date is wrong.
  (e.g. Armstrong 1969 vs Armstrong 1971 — semantically
  almost identical, factually wrong)
  This limitation directly motivates Layer 2 (NLI).

Academic basis:
  Manakul et al. (2023) SelfCheckGPT — consistency-based
  hallucination detection via multi-sample comparison.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ── Model (loaded once, reused across calls) ──────────────────────────────────
print("[Cosine Layer] Loading embedding model...")
_EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
print("[Cosine Layer] Model ready.")

# ── Thresholds ────────────────────────────────────────────────────────────────
# Agreement score thresholds for Layer 1 verdict
THRESHOLD_ACCEPT     = 0.85   # >= 0.85 → ACCEPT
THRESHOLD_FLAG       = 0.65   # >= 0.65 → FLAG, else REGENERATE

# Per-agent similarity threshold for AGREE/DISAGREE vote
SIMILARITY_THRESHOLD = 0.75


def compute_cosine_layer(
    primary_result:       dict,
    verification_results: list,
) -> dict:
    """
    Computes cosine similarity between primary and all verifiers.
    Produces mean agreement score, variance, per-agent scores,
    full similarity matrix, and an independent Layer 1 verdict.

    Args:
        primary_result:       Output from primary_agent.py
        verification_results: Output list from verification_agents.py

    Returns:
        Dict containing all Layer 1 outputs.
    """
    print("\n[Cosine Layer] Starting similarity analysis...")

    # ── Collect valid answers ─────────────────────────────────────────────────
    valid_verifiers = [
        v for v in verification_results
        if v["answer"] and not v.get("error")
    ]

    if not valid_verifiers:
        print("[Cosine Layer] ERROR: No valid verifier answers.")
        return _error_result("No valid verifier answers")

    # Build ordered list: primary first, then verifiers
    all_agents  = [primary_result] + valid_verifiers
    all_answers = [a["answer"] for a in all_agents]
    all_ids     = ["primary"] + [v["agent"] for v in valid_verifiers]

    n = len(all_answers)
    print(f"  Comparing {n} agent answers using sentence embeddings...")

    # ── Compute embeddings ────────────────────────────────────────────────────
    embeddings = _EMBEDDING_MODEL.encode(all_answers)

    # ── Full similarity matrix ────────────────────────────────────────────────
    sim_matrix = cosine_similarity(embeddings)

    # ── Primary vs each verifier ──────────────────────────────────────────────
    primary_vs_verifiers = {}
    verifier_similarities = []

    for i, verifier in enumerate(valid_verifiers):
        sim = float(sim_matrix[0][i + 1])  # row 0 = primary
        primary_vs_verifiers[verifier["agent"]] = round(sim, 4)
        verifier_similarities.append(sim)

    # ── Mean agreement score ──────────────────────────────────────────────────
    # Use upper triangle of similarity matrix (off-diagonal pairwise scores)
    upper_triangle = []
    for i in range(n):
        for j in range(i + 1, n):
            upper_triangle.append(sim_matrix[i][j])

    agreement_score = float(np.mean(upper_triangle))
    variance        = float(np.var(upper_triangle))

    # ── Per-agent AGREE/DISAGREE votes ───────────────────────────────────────
    agent_votes = {}
    for agent_id, sim in primary_vs_verifiers.items():
        agent_votes[agent_id] = "AGREE" if sim >= SIMILARITY_THRESHOLD else "DISAGREE"

    agree_count    = sum(1 for v in agent_votes.values() if v == "AGREE")
    disagree_count = sum(1 for v in agent_votes.values() if v == "DISAGREE")

    # ── Risk score (inverse of agreement) ────────────────────────────────────
    risk_score = round(1.0 - agreement_score, 4)

    # ── Independent verdict ───────────────────────────────────────────────────
    if agreement_score >= THRESHOLD_ACCEPT:
        verdict    = "ACCEPT"
        risk_level = "LOW"
        verdict_label = "✅ Agents strongly agree — low hallucination risk"
    elif agreement_score >= THRESHOLD_FLAG:
        verdict    = "FLAG"
        risk_level = "MODERATE"
        verdict_label = "⚠️  Partial agreement — possible hallucination"
    else:
        verdict    = "REGENERATE"
        risk_level = "HIGH"
        verdict_label = "🚨 Low agreement — high hallucination risk"

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  Agreement Score : {agreement_score:.4f}")
    print(f"  Variance        : {variance:.4f}  "
          f"{'(high — agents inconsistent)' if variance > 0.02 else '(low — agents consistent)'}")
    print(f"  Risk Score      : {risk_score:.4f}")
    print(f"  Risk Level      : {risk_level}")
    print(f"  Verdict         : {verdict}")
    print(f"  {verdict_label}")

    print(f"\n  Votes: {agree_count} AGREE, {disagree_count} DISAGREE")
    print(f"\n  Primary vs Each Verifier:")

    bar_max = 20
    for agent_id, sim in primary_vs_verifiers.items():
        bar_len = int(sim * bar_max)
        bar     = "█" * bar_len
        vote    = agent_votes[agent_id]
        print(f"    {agent_id:<12}: {sim:.4f}  {bar:<20}  {vote}")

    return {
        "layer":                 "cosine",
        "agreement_score":       round(agreement_score, 4),
        "variance":              round(variance, 4),
        "risk_score":            risk_score,
        "risk_level":            risk_level,
        "verdict":               verdict,
        "verdict_label":         verdict_label,
        "agent_votes":           agent_votes,
        "agree_count":           agree_count,
        "disagree_count":        disagree_count,
        "primary_vs_verifiers":  primary_vs_verifiers,
        "similarity_matrix":     sim_matrix.tolist(),   # for heatmap
        "agent_ids":             all_ids,               # for heatmap labels
    }


def _error_result(reason: str) -> dict:
    """Returns a safe error result when layer cannot compute."""
    return {
        "layer":                "cosine",
        "agreement_score":      0.0,
        "variance":             0.0,
        "risk_score":           1.0,
        "risk_level":           "HIGH",
        "verdict":              "REGENERATE",
        "verdict_label":        f"🚨 Layer 1 error: {reason}",
        "agent_votes":          {},
        "agree_count":          0,
        "disagree_count":       0,
        "primary_vs_verifiers": {},
        "similarity_matrix":    [],
        "agent_ids":            [],
        "error":                reason,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import hashlib
    import datetime
    from primary_agent import query_primary_agent
    from verification_agents import run_all_verification_agents

    print("\n" + "=" * 60)
    print("COSINE LAYER — TEST")
    print("=" * 60)

    # Test 1: Factual question (should score LOW)
    print("\n--- Test 1: Factual question ---")
    question = "What is the chemical formula for water?"

    primary = query_primary_agent(question)
    verifiers = run_all_verification_agents(question, include_byzantine=False)
    result = compute_cosine_layer(primary, verifiers)

    print(f"\n  Expected: LOW risk")
    print(f"  Got:      {result['risk_level']} — {result['verdict']}")

    # Test 2: Simulated hallucination (should score MODERATE or HIGH)
    print("\n\n--- Test 2: Simulated hallucination ---")
    question = "Who invented the telephone and in what year?"

    fake_answer = "Nikola Tesla invented the telephone in 1892."
    primary_fake = {
        "agent":     "primary",
        "model":     "simulated",
        "question":  question,
        "answer":    fake_answer,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hash":      hashlib.sha256(fake_answer.encode()).hexdigest(),
    }

    verifiers = run_all_verification_agents(question, include_byzantine=False)
    result = compute_cosine_layer(primary_fake, verifiers)

    print(f"\n  Injected: {fake_answer}")
    print(f"  Expected: MODERATE or HIGH risk")
    print(f"  Got:      {result['risk_level']} — {result['verdict']}")
    print(f"  Variance: {result['variance']} "
          f"({'high' if result['variance'] > 0.02 else 'low'})")