"""
regeneration_module.py
======================
Triggered when the decision engine produces FLAG or REGENERATE.

Asks the primary agent to self-correct its answer by providing
it with full evidence from all three detection layers:
  - All 9 verification agent answers
  - Panel judge scores and justifications per agent
  - Cross-validation pattern and confidence level
  - Identified hallucination type
  - Layer-by-layer risk scores

The corrected answer is then re-scored through all 3 layers.
The improvement delta is measured per layer and overall.

Research question (Mujeeb):
  "Do LLMs actually self-correct when told they hallucinated?
   Which models self-correct? By how much?"

This module directly answers that question by:
  1. Measuring pre-regen vs post-regen risk score
  2. Classifying outcome: IMPROVED / UNCHANGED / DEGRADED
  3. Recording which model was used as primary
  4. Tracking self-correction rate across runs
     → feeds Figure 5 and Table 3 in evaluation

Academic basis:
  Madaan et al. (2023) Self-Refine: Iterative Refinement
  with Self-Feedback. Extended here to multi-agent feedback
  rather than self-feedback alone.
"""

import ollama
import hashlib
import datetime
from cosine_layer      import compute_cosine_layer
from nli_layer         import compute_nli_layer
from judge_layer       import compute_judge_layer
from cross_validation  import compute_cross_validation
from fusion_module     import compute_fusion
from decision_engine   import make_decision
from trajectory_tracker import build_trajectory


# ── Improvement thresholds ────────────────────────────────────────────────────
IMPROVEMENT_THRESHOLD  = 0.05   # risk must drop by at least this to be IMPROVED
DEGRADATION_THRESHOLD  = 0.05   # risk must rise by at least this to be DEGRADED


def build_regeneration_prompt(
    question:             str,
    original_answer:      str,
    verification_results: list,
    judge_result:         dict,
    cross_validation:     dict,
    fusion_result:        dict,
) -> str:
    """
    Builds the self-correction prompt with full evidence.

    The primary agent receives:
      - Its original answer
      - All verifier answers
      - Its judge scores and justification
      - The hallucination type identified
      - Which layers flagged and why

    Args:
        question:             The original question
        original_answer:      The primary agent's original answer
        verification_results: All verification agent outputs
        judge_result:         Layer 3 judge output
        cross_validation:     Cross-validation output
        fusion_result:        Fusion module output

    Returns:
        Complete self-correction prompt string
    """
    # ── Verifier answers block ────────────────────────────────────────────────
    verifier_block = ""
    for v in verification_results:
        if v.get("answer") and not v.get("error"):
            verifier_block += f"\n  [{v['agent']} / {v['model']}]: {v['answer']}\n"

    # ── Judge scores for primary ──────────────────────────────────────────────
    primary_scores = judge_result.get("aggregated_scores", {}).get("primary", {})
    halluc_score   = primary_scores.get("avg_hallucination_score", "N/A")
    factual_score  = primary_scores.get("avg_factual_accuracy",    "N/A")
    justification  = primary_scores.get("justification", "No justification available.")

    # ── Cross-validation summary ──────────────────────────────────────────────
    halluc_type    = cross_validation.get("hallucination_type", "unknown")
    confidence     = cross_validation.get("confidence_level",   "unknown")
    pattern        = cross_validation.get("pattern",            "unknown")
    flagging       = cross_validation.get("flagging_layers",    [])
    final_risk     = fusion_result.get("final_risk_score",      0.0)

    prompt = f"""You previously answered the following question:

QUESTION: "{question}"

YOUR ORIGINAL ANSWER:
"{original_answer}"

DETECTION RESULT: Your answer has been flagged for potential hallucination.

EVIDENCE FROM {len(verification_results)} INDEPENDENT VERIFICATION AGENTS:
{verifier_block}

YOUR JUDGE PANEL SCORES (averaged across {len(verification_results)} independent judges):
  Hallucination Score : {halluc_score} / 10  (0=none, 10=severe)
  Factual Accuracy    : {factual_score} / 10
  Judge justification : {justification}

DETECTION LAYER ANALYSIS:
  Hallucination type  : {halluc_type}
  Detection pattern   : {pattern}
  Layers that flagged : {flagging}
  Confidence level    : {confidence}
  Final risk score    : {final_risk:.4f} / 1.0

Based on this evidence, please provide a CORRECTED answer to the question.

Rules for your corrected answer:
  1. Address the specific hallucination type identified: {halluc_type}
  2. Align with what the majority of verification agents said
  3. Be factually accurate — fix any wrong names, dates, or numbers
  4. Be concise and direct
  5. Do NOT repeat or defend your original answer
  6. Do NOT mention the detection process or this prompt

Provide only your corrected factual answer:"""

    return prompt


def run_regeneration(
    question:             str,
    primary_result:       dict,
    verification_results: list,
    cosine_result:        dict,
    nli_result:           dict,
    judge_result:         dict,
    cross_validation:     dict,
    fusion_result:        dict,
    decision:             dict,
    topology:             str = "star",
) -> dict:
    """
    Runs the self-correction loop for the primary agent.
    Re-scores the corrected answer through all 3 layers.
    Measures improvement delta.

    Args:
        All pipeline stage outputs.

    Returns:
        Dict containing corrected answer, re-scored results,
        improvement delta, and outcome classification.
        Returns None fields if regeneration was not triggered.
    """
    action = decision.get("action", "ACCEPT")

    # ── Check if regeneration is needed ──────────────────────────────────────
    if action == "ACCEPT":
        print("\n[Regeneration] Not triggered — decision was ACCEPT")
        return {
            "regeneration_triggered": False,
            "original_answer":        primary_result.get("answer"),
            "corrected_answer":       None,
            "final_answer":           primary_result.get("answer"),
            "pre_regen_risk":         fusion_result.get("final_risk_score"),
            "post_regen_risk_score":  None,
            "improvement_delta":      None,
            "outcome":                "NOT_TRIGGERED",
            "layer_deltas":           {},
        }

    print(f"\n[Regeneration] Triggered — decision was {action}")
    print(f"  Pre-regen risk score: {fusion_result.get('final_risk_score'):.4f}")
    print(f"  Hallucination type  : {cross_validation.get('hallucination_type')}")
    print(f"  Building correction prompt with full evidence...")

    # ── Build prompt ──────────────────────────────────────────────────────────
    original_answer = primary_result.get("answer", "")
    prompt = build_regeneration_prompt(
        question             = question,
        original_answer      = original_answer,
        verification_results = verification_results,
        judge_result         = judge_result,
        cross_validation     = cross_validation,
        fusion_result        = fusion_result,
    )

    # ── Call primary agent for self-correction ────────────────────────────────
    model = primary_result.get("model", "mistral")
    print(f"\n  Asking {model} to self-correct...")

    try:
        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a factual assistant. "
                        "When shown evidence that your answer was wrong, "
                        "you correct yourself honestly and concisely."
                    )
                },
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            options={"temperature": 0.3},
        )
        corrected_answer = response["message"]["content"].strip()

    except Exception as e:
        print(f"  [Regeneration] ERROR calling model: {e}")
        return _failed_result(primary_result, fusion_result, str(e))

    print(f"\n  Original  : {original_answer[:120]}...")
    print(f"  Corrected : {corrected_answer[:120]}...")

    # ── Hash corrected answer ─────────────────────────────────────────────────
    corrected_hash = hashlib.sha256(
        f"{question}||{corrected_answer}".encode()
    ).hexdigest()

    corrected_primary = {
        "agent":     "primary",
        "agent_id":  "primary",
        "model":     model,
        "question":  question,
        "answer":    corrected_answer,
        "hash":      corrected_hash,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # ── Re-score through all 3 layers ─────────────────────────────────────────
    print(f"\n  Re-scoring corrected answer through all 3 layers...")

    post_cosine = compute_cosine_layer(corrected_primary, verification_results)
    post_nli    = compute_nli_layer(corrected_primary,    verification_results)
    post_judge  = compute_judge_layer(
        question             = question,
        primary_result       = corrected_primary,
        verification_results = verification_results,
        topology             = topology,
    )

    post_cv     = compute_cross_validation(post_cosine, post_nli, post_judge)
    post_fusion = compute_fusion(post_cosine, post_nli, post_judge, post_cv)
    post_decision = make_decision(post_fusion, post_cv)

    post_regen_risk = post_fusion.get("final_risk_score", 0.0)
    pre_regen_risk  = fusion_result.get("final_risk_score", 0.0)

    # ── Improvement delta ─────────────────────────────────────────────────────
    improvement_delta = round(pre_regen_risk - post_regen_risk, 4)

    # Positive delta = improvement (risk went down)
    # Negative delta = degradation (risk went up)

    if improvement_delta >= IMPROVEMENT_THRESHOLD:
        outcome = "IMPROVED"
        outcome_label = f"✅ IMPROVED — risk reduced by {improvement_delta:.4f}"
    elif improvement_delta <= -DEGRADATION_THRESHOLD:
        outcome = "DEGRADED"
        outcome_label = f"⚠️  DEGRADED — risk increased by {abs(improvement_delta):.4f}"
    else:
        outcome = "UNCHANGED"
        outcome_label = f"➖ UNCHANGED — delta within threshold ({improvement_delta:.4f})"

    # ── Per-layer improvement deltas ──────────────────────────────────────────
    layer_deltas = {
        "cosine": round(
            cosine_result.get("risk_score", 0) - post_cosine.get("risk_score", 0), 4
        ),
        "nli": round(
            nli_result.get("risk_score", 0) - post_nli.get("risk_score", 0), 4
        ),
        "judge": round(
            judge_result.get("risk_score", 0) - post_judge.get("risk_score", 0), 4
        ),
    }

    # ── Build post-regen trajectory ───────────────────────────────────────────
    post_regen_for_traj = {
        "regeneration_triggered": True,
        "post_regen_risk_score":  post_regen_risk,
        "outcome":                outcome,
    }
    build_trajectory(
        post_cosine, post_nli, post_judge, post_fusion, post_regen_for_traj
    )

    # ── Print summary ─────────────────────────────────────────────────────────
    print(f"\n  REGENERATION SUMMARY:")
    print(f"  Model            : {model}")
    print(f"  Pre-regen risk   : {pre_regen_risk:.4f}")
    print(f"  Post-regen risk  : {post_regen_risk:.4f}")
    print(f"  Delta            : {improvement_delta:+.4f}")
    print(f"  Outcome          : {outcome}")
    print(f"  {outcome_label}")
    print(f"\n  Per-layer deltas :")
    for layer, delta in layer_deltas.items():
        arrow = "↓" if delta > 0 else ("↑" if delta < 0 else "→")
        print(f"    {layer:<8}: {delta:+.4f}  {arrow}")
    print(f"\n  Post-regen decision: {post_decision['action']}")

    return {
        "regeneration_triggered": True,
        "original_answer":        original_answer,
        "corrected_answer":       corrected_answer,
        "final_answer":           corrected_answer,
        "pre_regen_risk":         pre_regen_risk,
        "post_regen_risk_score":  post_regen_risk,
        "improvement_delta":      improvement_delta,
        "outcome":                outcome,
        "outcome_label":          outcome_label,
        "layer_deltas":           layer_deltas,
        "post_cosine":            post_cosine,
        "post_nli":               post_nli,
        "post_judge":             post_judge,
        "post_cv":                post_cv,
        "post_fusion":            post_fusion,
        "post_decision":          post_decision,
        "model":                  model,
        "corrected_hash":         corrected_hash,
    }


def _failed_result(primary_result, fusion_result, error_msg):
    """Returns a safe result when regeneration call fails."""
    return {
        "regeneration_triggered": True,
        "original_answer":        primary_result.get("answer"),
        "corrected_answer":       None,
        "final_answer":           primary_result.get("answer"),
        "pre_regen_risk":         fusion_result.get("final_risk_score"),
        "post_regen_risk_score":  fusion_result.get("final_risk_score"),
        "improvement_delta":      0.0,
        "outcome":                "FAILED",
        "layer_deltas":           {},
        "error":                  error_msg,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import hashlib
    import datetime
    from primary_agent        import query_primary_agent
    from verification_agents  import run_all_verification_agents

    print("\n" + "=" * 60)
    print("REGENERATION MODULE — TEST")
    print("=" * 60)
    print("Testing: Thomas Edison hallucination injection")
    print("Expected: model self-corrects to Alexander Graham Bell")
    print("=" * 60)

    question    = "Who invented the telephone?"
    fake_answer = "Thomas Edison invented the telephone in 1877."

    # Simulated hallucinated primary
    fake_primary = {
        "agent":     "primary",
        "agent_id":  "primary",
        "model":     "mistral",
        "question":  question,
        "answer":    fake_answer,
        "hash":      hashlib.sha256(fake_answer.encode()).hexdigest(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }

    # Run verification agents
    verifiers = run_all_verification_agents(question, include_byzantine=False)

    # Run all 3 detection layers
    cosine = compute_cosine_layer(fake_primary, verifiers)
    nli    = compute_nli_layer(fake_primary,    verifiers)
    judge  = compute_judge_layer(
        question             = question,
        primary_result       = fake_primary,
        verification_results = verifiers,
        topology             = "star",
    )

    cv       = compute_cross_validation(cosine, nli, judge)
    fusion   = compute_fusion(cosine, nli, judge, cv)
    decision = make_decision(fusion, cv)

    print(f"\n  Pre-regen decision: {decision['action']}")
    print(f"  Pre-regen risk    : {fusion['final_risk_score']:.4f}")

    # Run regeneration
    regen = run_regeneration(
        question             = question,
        primary_result       = fake_primary,
        verification_results = verifiers,
        cosine_result        = cosine,
        nli_result           = nli,
        judge_result         = judge,
        cross_validation     = cv,
        fusion_result        = fusion,
        decision             = decision,
        topology             = "star",
    )

    print(f"\n{'='*60}")
    print(f"FINAL RESULT:")
    print(f"  Original answer  : {regen['original_answer']}")
    print(f"  Corrected answer : {regen['corrected_answer']}")
    print(f"  Outcome          : {regen['outcome']}")
    print(f"  Pre-regen risk   : {regen['pre_regen_risk']:.4f}")
    print(f"  Post-regen risk  : {regen['post_regen_risk_score']:.4f}")
    print(f"  Improvement delta: {regen['improvement_delta']:+.4f}")