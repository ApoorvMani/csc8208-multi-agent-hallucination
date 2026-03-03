"""
explanation_generator.py
========================
Produces a plain English summary of every pipeline decision.

The explanation is written for a non-technical reader —
it describes what the system found, why it made its decision,
and what happened if regeneration was triggered.

Example output:
  "Answer ACCEPTED with HIGH confidence. All 3 detection
   layers agreed — no hallucination detected. Cosine
   similarity: 0.922 (LOW risk). NLI found no contradictions.
   Panel judge scored 8.5/10 factual accuracy. Hallucination
   type: none."

  "Answer FLAGGED with HIGH confidence. All 3 layers
   independently flagged. NLI detected factual contradiction
   in 7/9 agents. Panel judge scored 3.2/10 factual accuracy.
   Hallucination type: confirmed factual error. Regeneration
   improved risk from 0.71 to 0.08. Outcome: IMPROVED."

This text is written into the blockchain audit record and
printed to the terminal at the end of every pipeline run.
It also answers Mujeeb's key question when regeneration
is triggered: did the model self-correct, and by how much?
"""


def generate_explanation(
    decision:              dict,
    cross_validation:      dict,
    cosine_result:         dict,
    nli_result:            dict,
    judge_result:          dict,
    fusion_result:         dict,
    regeneration_result:   dict = None,
    trajectory:            dict = None,
) -> str:
    """
    Generates a plain English explanation of the full pipeline decision.

    Args:
        decision:            Output from decision_engine.make_decision()
        cross_validation:    Output from cross_validation.compute_cross_validation()
        cosine_result:       Output from cosine_layer.compute_cosine_layer()
        nli_result:          Output from nli_layer.compute_nli_layer()
        judge_result:        Output from judge_layer.compute_judge_layer()
        fusion_result:       Output from fusion_module.compute_fusion()
        regeneration_result: Output from regeneration_module.run_regeneration()
                             Pass None if regeneration was not triggered.
        trajectory:          Output from trajectory_tracker.build_trajectory()
                             Used for step-change annotation. Optional.

    Returns:
        Plain English explanation string suitable for terminal
        output and blockchain audit record.
    """
    # ── Unpack key fields ─────────────────────────────────────────────────────
    action           = decision.get("action",             "UNKNOWN")
    risk_level       = decision.get("risk_level",         "UNKNOWN")
    final_risk       = fusion_result.get("final_risk_score", 0.0)
    confidence       = cross_validation.get("confidence_level",   "UNKNOWN")
    halluc_type      = cross_validation.get("hallucination_type", "none")
    pattern          = cross_validation.get("pattern",            "UNKNOWN")
    flagging_layers  = cross_validation.get("flagging_layers",    [])
    override         = decision.get("override_applied",   False)

    cosine_risk      = cosine_result.get("risk_score",             0.0)
    cosine_verdict   = cosine_result.get("verdict",               "UNKNOWN")
    agreement_score  = cosine_result.get("agreement_score",        0.0)

    nli_risk         = nli_result.get("risk_score",                0.0)
    nli_verdict      = nli_result.get("verdict",                  "UNKNOWN")
    contradiction_n  = nli_result.get("contradiction_count",       0)
    entailment_n     = nli_result.get("entailment_count",          0)
    total_pairs      = contradiction_n + entailment_n + nli_result.get("neutral_count", 0)

    judge_risk       = judge_result.get("risk_score",              0.0)
    judge_verdict    = judge_result.get("verdict",                "UNKNOWN")
    primary_halluc   = judge_result.get("primary_hallucination_score", 0.0)
    primary_factual  = judge_result.get("primary_factual_accuracy",    0.0)
    panel_verdict    = judge_result.get("primary_verdict",        "UNKNOWN")

    # ── Action headline ───────────────────────────────────────────────────────
    action_icons = {
        "ACCEPT":     "✅",
        "FLAG":       "⚠️ ",
        "REGENERATE": "🚨",
    }
    icon = action_icons.get(action, "❓")

    lines = []
    lines.append(f"{icon} Answer {action} — {confidence} confidence, {risk_level} risk.")

    # ── Override note ─────────────────────────────────────────────────────────
    if override:
        lines.append(
            "  Note: Decision downgraded from REGENERATE to FLAG "
            "because only 1 layer flagged (low confidence). "
            "Human review recommended."
        )

    # ── Final risk score ──────────────────────────────────────────────────────
    lines.append(f"  Final fused risk score: {final_risk:.4f} / 1.00.")

    # ── Cross-validation pattern ──────────────────────────────────────────────
    if flagging_layers:
        layers_str = ", ".join(flagging_layers)
        lines.append(
            f"  Detection pattern: {pattern}. "
            f"Flagging layers: [{layers_str}]. "
            f"Hallucination type identified: {halluc_type}."
        )
    else:
        lines.append(
            f"  Detection pattern: {pattern}. "
            f"No layers flagged. Hallucination type: {halluc_type}."
        )

    # ── Layer 1: Cosine ───────────────────────────────────────────────────────
    lines.append(
        f"  Layer 1 (Cosine Similarity): risk {cosine_risk:.4f}, "
        f"verdict {cosine_verdict}. "
        f"Mean agent agreement: {agreement_score:.3f}."
    )

    # ── Layer 2: NLI ──────────────────────────────────────────────────────────
    if total_pairs > 0:
        lines.append(
            f"  Layer 2 (NLI Contradiction): risk {nli_risk:.4f}, "
            f"verdict {nli_verdict}. "
            f"{contradiction_n} contradiction(s), {entailment_n} entailment(s) "
            f"from {total_pairs} agent-pair checks."
        )
    else:
        lines.append(
            f"  Layer 2 (NLI Contradiction): risk {nli_risk:.4f}, "
            f"verdict {nli_verdict}."
        )

    # ── Layer 3: Judge ────────────────────────────────────────────────────────
    lines.append(
        f"  Layer 3 (Consensus Panel Judge): risk {judge_risk:.4f}, "
        f"verdict {judge_verdict}. "
        f"Primary agent scored {primary_factual:.1f}/10 factual accuracy, "
        f"{primary_halluc:.1f}/10 hallucination score. "
        f"Panel consensus: {panel_verdict}."
    )

    # ── Trajectory step changes ───────────────────────────────────────────────
    if trajectory:
        sig_changes = trajectory.get("significant_changes", [])
        if sig_changes:
            lines.append("  Notable layer step-changes (>0.10):")
            for change in sig_changes:
                lines.append(f"    → {change}")

    # ── Regeneration outcome ──────────────────────────────────────────────────
    if regeneration_result and regeneration_result.get("regeneration_triggered"):
        pre_risk  = regeneration_result.get("pre_regen_risk",        final_risk)
        post_risk = regeneration_result.get("post_regen_risk_score", final_risk)
        delta     = regeneration_result.get("improvement_delta",     0.0)
        outcome   = regeneration_result.get("outcome",               "UNKNOWN")
        model     = regeneration_result.get("model",                 "unknown")

        outcome_icons = {
            "IMPROVED":  "✅",
            "UNCHANGED": "➖",
            "DEGRADED":  "⚠️ ",
            "FAILED":    "❌",
        }
        regen_icon = outcome_icons.get(outcome, "❓")

        lines.append(
            f"  Regeneration triggered. Model {model} asked to self-correct."
        )

        if outcome == "FAILED":
            lines.append("  Self-correction call failed — original answer retained.")
        else:
            lines.append(
                f"  {regen_icon} Self-correction outcome: {outcome}. "
                f"Risk score changed from {pre_risk:.4f} to {post_risk:.4f} "
                f"(delta: {delta:+.4f})."
            )

            # Per-layer deltas
            layer_deltas = regeneration_result.get("layer_deltas", {})
            if layer_deltas:
                delta_parts = []
                for layer_name, layer_delta in layer_deltas.items():
                    arrow = "↓" if layer_delta > 0 else ("↑" if layer_delta < 0 else "→")
                    delta_parts.append(f"{layer_name} {arrow}{abs(layer_delta):.3f}")
                lines.append(f"  Per-layer improvement: {', '.join(delta_parts)}.")

            # Post-regen decision
            post_decision = regeneration_result.get("post_decision", {})
            if post_decision:
                post_action = post_decision.get("action", "UNKNOWN")
                lines.append(
                    f"  Post-regeneration decision: {post_action}."
                )
    else:
        if action != "ACCEPT":
            lines.append("  Regeneration was not triggered.")

    # ── Join into paragraph ───────────────────────────────────────────────────
    explanation = "\n".join(lines)
    return explanation


def print_explanation(explanation: str):
    """Prints the explanation with a header box to the terminal."""
    print("\n" + "=" * 70)
    print("  PIPELINE DECISION — PLAIN ENGLISH EXPLANATION")
    print("=" * 70)
    print(explanation)
    print("=" * 70)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from cross_validation import compute_cross_validation
    from fusion_module    import compute_fusion
    from decision_engine  import make_decision

    print("\n" + "=" * 60)
    print("EXPLANATION GENERATOR — TEST")
    print("=" * 60)

    # ── Test 1: Clean answer — ACCEPT ─────────────────────────────────────────
    print("\n--- Test 1: Clean answer ---")

    cosine = {
        "verdict": "ACCEPT", "risk_score": 0.042,
        "agreement_score": 0.922, "variance": 0.01,
    }
    nli = {
        "verdict": "ACCEPT", "risk_score": 0.008,
        "contradiction_count": 0, "entailment_count": 8, "neutral_count": 1,
    }
    judge = {
        "verdict": "ACCEPT", "risk_score": 0.000,
        "primary_hallucination_score": 0.3,
        "primary_factual_accuracy":    9.2,
        "primary_verdict": "TRUSTWORTHY",
    }

    cv       = compute_cross_validation(cosine, nli, judge)
    fusion   = compute_fusion(cosine, nli, judge, cv)
    decision = make_decision(fusion, cv)

    explanation = generate_explanation(
        decision=decision,
        cross_validation=cv,
        cosine_result=cosine,
        nli_result=nli,
        judge_result=judge,
        fusion_result=fusion,
        regeneration_result=None,
    )
    print_explanation(explanation)

    # ── Test 2: Hallucination — REGENERATE + IMPROVED ────────────────────────
    print("\n--- Test 2: Hallucination with regeneration ---")

    cosine = {
        "verdict": "FLAG", "risk_score": 0.710,
        "agreement_score": 0.612, "variance": 0.08,
    }
    nli = {
        "verdict": "REGENERATE", "risk_score": 0.890,
        "contradiction_count": 7, "entailment_count": 1, "neutral_count": 1,
    }
    judge = {
        "verdict": "REGENERATE", "risk_score": 0.812,
        "primary_hallucination_score": 8.1,
        "primary_factual_accuracy":    2.3,
        "primary_verdict": "NOT TRUSTWORTHY",
    }

    cv       = compute_cross_validation(cosine, nli, judge)
    fusion   = compute_fusion(cosine, nli, judge, cv)
    decision = make_decision(fusion, cv)

    regen = {
        "regeneration_triggered": True,
        "pre_regen_risk":         fusion["final_risk_score"],
        "post_regen_risk_score":  0.085,
        "improvement_delta":      round(fusion["final_risk_score"] - 0.085, 4),
        "outcome":                "IMPROVED",
        "model":                  "mistral",
        "layer_deltas": {
            "cosine": 0.61,
            "nli":    0.80,
            "judge":  0.72,
        },
        "post_decision": {"action": "ACCEPT"},
    }

    explanation = generate_explanation(
        decision=decision,
        cross_validation=cv,
        cosine_result=cosine,
        nli_result=nli,
        judge_result=judge,
        fusion_result=fusion,
        regeneration_result=regen,
    )
    print_explanation(explanation)