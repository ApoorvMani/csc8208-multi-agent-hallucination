"""
nli_layer.py
============
Layer 2 of the triple-layer hallucination detection pipeline.

Uses Natural Language Inference (NLI) to detect factual
contradictions between the primary agent's answer and each
verification agent's answer.

Why NLI?
  Cosine similarity measures semantic closeness in vector space.
  It cannot distinguish "Armstrong walked on the moon in 1969"
  from "Armstrong walked on the moon in 1971" because the
  surrounding context is identical — only the number differs.

  NLI operates on logical relationship:
    ENTAILMENT    → verifier agrees with primary (supports it)
    NEUTRAL       → verifier neither agrees nor disagrees
    CONTRADICTION → verifier directly contradicts primary

  A year difference (1969 vs 1971) produces CONTRADICTION
  because the two claims cannot both be true simultaneously.
  This directly addresses the v0.1 missed detection case.

Model: cross-encoder/nli-deberta-v3-small
  - State-of-the-art NLI performance on MultiNLI benchmark
  - Free, local, no API key required
  - Runs on CPU (no GPU required)
  - Downloaded automatically from HuggingFace on first run (~180MB)
  - Used as cross-encoder: takes (premise, hypothesis) pair
    and classifies the RELATIONSHIP between them directly.
  - CRITICAL: must use text-classification pipeline with
    text_pair argument — NOT zero-shot-classification.
    zero-shot-classification classifies against label strings,
    not against actual hypothesis sentences.

Academic basis:
  He et al. (2021) DeBERTa — Decoding-enhanced BERT with
  Disentangled Attention. Strong NLI performance.
  Applied here to hallucination detection as Layer 2.
"""

from transformers import pipeline
import torch

# ── Model (loaded once at import time) ────────────────────────────────────────
print("[NLI Layer] Loading DeBERTa NLI model...")
print("[NLI Layer] (First run downloads ~180MB from HuggingFace — please wait)")

# IMPORTANT: text-classification + text_pair is the correct
# way to use a cross-encoder NLI model.
# The model takes (premise, hypothesis) and returns the
# relationship label with a confidence score.
_NLI_PIPELINE = pipeline(
    task="text-classification",
    model="cross-encoder/nli-deberta-v3-small",
    device=0 if torch.cuda.is_available() else -1,
)

print("[NLI Layer] Model ready.")

# Label mapping — cross-encoder/nli-deberta-v3-small returns:
# LABEL_0 = CONTRADICTION
# LABEL_1 = ENTAILMENT
# LABEL_2 = NEUTRAL
# (order confirmed from model card on HuggingFace)
LABEL_MAP = {
    "LABEL_0": "CONTRADICTION",
    "LABEL_1": "ENTAILMENT",
    "LABEL_2": "NEUTRAL",
}

# ── Thresholds ────────────────────────────────────────────────────────────────
THRESHOLD_ACCEPT = 0.20   # risk_score <= 0.20 → ACCEPT
THRESHOLD_FLAG   = 0.45   # risk_score <= 0.45 → FLAG, else REGENERATE


def classify_pair(premise: str, hypothesis: str) -> dict:
    """
    Classifies the NLI relationship between premise and hypothesis.

    The model receives both texts as a pair and returns which
    logical relationship holds between them.

    Args:
        premise:    The primary agent's answer (what we are evaluating)
        hypothesis: A verification agent's answer (the reference check)

    Returns:
        Dict with dominant label and confidence score
    """
    result = _NLI_PIPELINE({
        "text":      premise,
        "text_pair": hypothesis,
    })

    raw_label = result["label"]   # e.g. "LABEL_0"
    score     = result["score"]   # confidence in that label

    label = LABEL_MAP.get(raw_label, raw_label).upper()

    return {
        "label": label,
        "score": round(score, 4),
    }


def compute_nli_layer(
    primary_result:       dict,
    verification_results: list,
) -> dict:
    """
    Runs NLI classification between primary and all verifiers.
    Produces contradiction count, entailment count, risk score,
    and an independent Layer 2 verdict.

    Args:
        primary_result:       Output from primary_agent.py
        verification_results: Output list from verification_agents.py

    Returns:
        Dict containing all Layer 2 outputs.
    """
    print("\n[NLI Layer] Starting contradiction analysis...")

    primary_answer = primary_result.get("answer", "").strip()

    if not primary_answer:
        print("[NLI Layer] ERROR: Primary answer is empty.")
        return _error_result("Primary answer is empty")

    valid_verifiers = [
        v for v in verification_results
        if v.get("answer") and not v.get("error")
    ]

    if not valid_verifiers:
        print("[NLI Layer] ERROR: No valid verifier answers.")
        return _error_result("No valid verifier answers")

    print(f"  Classifying {len(valid_verifiers)} verifier answers against primary...")
    print(f"  Primary: {primary_answer[:80]}...")

    # ── Classify each verifier against primary ────────────────────────────────
    agent_classifications = {}
    entailment_count    = 0
    neutral_count       = 0
    contradiction_count = 0
    contradiction_scores = []

    for verifier in valid_verifiers:
        agent_id        = verifier["agent"]
        verifier_answer = verifier["answer"].strip()

        classification = classify_pair(
            premise    = primary_answer,
            hypothesis = verifier_answer,
        )

        label = classification["label"].upper()
        score = classification["score"]

        agent_classifications[agent_id] = classification

        # Count by label and build risk contribution
        if label == "ENTAILMENT":
            entailment_count += 1
            contradiction_scores.append(1.0 - score)  # low risk
        elif label == "NEUTRAL":
            neutral_count += 1
            contradiction_scores.append(0.5)           # moderate risk
        elif label == "CONTRADICTION":
            contradiction_count += 1
            contradiction_scores.append(score)         # high risk

        icon = {
            "ENTAILMENT":    "✅",
            "NEUTRAL":       "➖",
            "CONTRADICTION": "🚨",
        }.get(label, "?")

        print(
            f"  {icon} {agent_id:<12} → {label:<15} "
            f"(confidence: {score:.4f})"
        )

    n = len(valid_verifiers)

    # ── Risk score ────────────────────────────────────────────────────────────
    mean_risk  = sum(contradiction_scores) / n if n > 0 else 0.0
    risk_score = round(mean_risk, 4)

    # ── Independent verdict ───────────────────────────────────────────────────
    if risk_score <= THRESHOLD_ACCEPT:
        verdict       = "ACCEPT"
        risk_level    = "LOW"
        verdict_label = "✅ No significant contradictions detected"
    elif risk_score <= THRESHOLD_FLAG:
        verdict       = "FLAG"
        risk_level    = "MODERATE"
        verdict_label = "⚠️  Some contradictions detected — review recommended"
    else:
        verdict       = "REGENERATE"
        risk_level    = "HIGH"
        verdict_label = "🚨 Strong contradictions detected — likely hallucination"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Results:")
    print(f"  ENTAILMENT   : {entailment_count} / {n} agents support primary")
    print(f"  NEUTRAL      : {neutral_count} / {n} agents neither support nor contradict")
    print(f"  CONTRADICTION: {contradiction_count} / {n} agents contradict primary")
    print(f"\n  Risk Score  : {risk_score:.4f}")
    print(f"  Risk Level  : {risk_level}")
    print(f"  Verdict     : {verdict}")
    print(f"  {verdict_label}")

    return {
        "layer":                 "nli",
        "entailment_count":      entailment_count,
        "neutral_count":         neutral_count,
        "contradiction_count":   contradiction_count,
        "risk_score":            risk_score,
        "risk_level":            risk_level,
        "verdict":               verdict,
        "verdict_label":         verdict_label,
        "agent_classifications": agent_classifications,
        "n_verifiers":           n,
    }


def _error_result(reason: str) -> dict:
    """Returns a safe error result when layer cannot compute."""
    return {
        "layer":                 "nli",
        "entailment_count":      0,
        "neutral_count":         0,
        "contradiction_count":   0,
        "risk_score":            1.0,
        "risk_level":            "HIGH",
        "verdict":               "REGENERATE",
        "verdict_label":         f"🚨 Layer 2 error: {reason}",
        "agent_classifications": {},
        "n_verifiers":           0,
        "error":                 reason,
    }


# ── Quick test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import hashlib
    import datetime
    from primary_agent import query_primary_agent
    from verification_agents import run_all_verification_agents

    print("\n" + "=" * 60)
    print("NLI LAYER — TEST")
    print("=" * 60)

    # ── Test 1: Factual question — should ACCEPT ──────────────────────────────
    print("\n--- Test 1: Factual question (expect LOW / ACCEPT) ---")
    question = "What is the chemical formula for water?"

    primary   = query_primary_agent(question)
    verifiers = run_all_verification_agents(question, include_byzantine=False)
    result    = compute_nli_layer(primary, verifiers)

    print(f"\n  Expected : LOW — ACCEPT")
    print(f"  Got      : {result['risk_level']} — {result['verdict']}")

    # ── Test 2: Wrong year — the case cosine missed in v0.1 ───────────────────
    print("\n\n--- Test 2: Wrong year injection (expect MODERATE or HIGH) ---")
    print("  Injecting: 'Neil Armstrong first walked on the Moon in 1971.'")
    print("  (correct year is 1969 — cosine missed this, NLI should catch it)\n")

    question = "When did Neil Armstrong first walk on the Moon?"

    fake_primary = {
        "agent":     "primary",
        "agent_id":  "primary",
        "model":     "simulated",
        "question":  question,
        "answer":    "Neil Armstrong first walked on the Moon in 1971.",
        "hash":      hashlib.sha256("test".encode()).hexdigest(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    verifiers = run_all_verification_agents(question, include_byzantine=False)
    result    = compute_nli_layer(fake_primary, verifiers)

    print(f"\n  Expected      : MODERATE or HIGH")
    print(f"  Got           : {result['risk_level']} — {result['verdict']}")
    print(f"  Contradictions: {result['contradiction_count']} / {result['n_verifiers']}")
    print(f"\n  Key result: NLI caught what cosine missed ✅"
          if result["contradiction_count"] > 0
          else "\n  Key result: NLI also missed this — check label mapping.")