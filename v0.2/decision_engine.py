"""
decision_engine.py
==================
Converts the final fusion risk score into an action.

Thresholds:
  0.00 – 0.20  →  ACCEPT      (low risk — answer likely correct)
  0.20 – 0.45  →  FLAG        (moderate risk — review recommended)
  0.45 – 1.00  →  REGENERATE  (high risk — self-correction triggered)

These thresholds are hyperparameters tested in evaluate.py
via ROC curve analysis (Figure 6).

Low confidence override:
  If cross-validation confidence is LOW (only 1 layer flagged),
  a REGENERATE verdict is downgraded to FLAG to avoid
  aggressive regeneration on weak evidence. Human review
  is recommended instead.

Future quorum integration:
  When judge_layer.py implements the Byzantine quorum model,
  decision_engine will also accept a quorum_result field.
  If quorum says REGENERATE but fusion says FLAG, the quorum
  escalates the decision. This connects verdict logic directly
  to Byzantine fault tolerance theory.
"""


# ── Decision thresholds ───────────────────────────────────────────────────────

THRESHOLD_ACCEPT = 0.20   # below this → ACCEPT
THRESHOLD_FLAG   = 0.45   # below this → FLAG, else REGENERATE


def make_decision(
    fusion_result:    dict,
    cross_validation: dict,
) -> dict:
    """
    Maps fusion risk score to a pipeline action.

    Args:
        fusion_result:    Output from fusion_module.compute_fusion()
        cross_validation: Output from cross_validation.compute_cross_validation()

    Returns:
        Dict with action, risk_level, label, and explanation.
    """
    print("\n[Decision Engine] Computing final decision...")

    final_risk       = fusion_result.get("final_risk_score", 0.0)
    confidence_level = cross_validation.get("confidence_level", "HIGH")
    halluc_type      = cross_validation.get("hallucination_type", "none")

    # ── Base decision from risk score ─────────────────────────────────────────
    if final_risk < THRESHOLD_ACCEPT:
        action     = "ACCEPT"
        risk_level = "LOW"
        label      = "✅ Answer accepted — risk below threshold"

    elif final_risk < THRESHOLD_FLAG:
        action     = "FLAG"
        risk_level = "MODERATE"
        label      = "⚠️  Answer flagged — moderate hallucination risk"

    else:
        action     = "REGENERATE"
        risk_level = "HIGH"
        label      = "🚨 Regeneration triggered — high hallucination risk"

    # ── Low confidence override ───────────────────────────────────────────────
    # Only 1 layer flagged — downgrade REGENERATE to FLAG
    # Avoids aggressive action on weak single-layer evidence
    override_applied = False
    if confidence_level == "LOW" and action == "REGENERATE":
        action           = "FLAG"
        risk_level       = "MODERATE"
        label            = "⚠️  Flagged (downgraded from REGENERATE — low confidence)"
        override_applied = True

    # ── Print result ──────────────────────────────────────────────────────────
    print(f"\n  Final Risk Score  : {final_risk:.4f}")
    print(f"  Confidence Level  : {confidence_level}")
    print(f"  Hallucination Type: {halluc_type}")
    print(f"  Action            : {action}")
    print(f"  Risk Level        : {risk_level}")
    if override_applied:
        print(f"  Override          : REGENERATE → FLAG (low confidence)")
    print(f"  {label}")

    return {
        "action":             action,
        "risk_level":         risk_level,
        "label":              label,
        "final_risk_score":   final_risk,
        "confidence_level":   confidence_level,
        "hallucination_type": halluc_type,
        "override_applied":   override_applied,
        "thresholds": {
            "accept": THRESHOLD_ACCEPT,
            "flag":   THRESHOLD_FLAG,
        },
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from cross_validation import compute_cross_validation
    from fusion_module    import compute_fusion

    print("\n" + "=" * 60)
    print("DECISION ENGINE — TEST")
    print("=" * 60)

    cases = [
        {
            "label":  "Clean answer",
            "cosine": {"verdict": "ACCEPT", "risk_score": 0.042},
            "nli":    {"verdict": "ACCEPT", "risk_score": 0.008},
            "judge":  {"verdict": "ACCEPT", "risk_score": 0.000},
            "expect": "ACCEPT",
        },
        {
            "label":  "Moderate risk",
            "cosine": {"verdict": "FLAG",   "risk_score": 0.350},
            "nli":    {"verdict": "ACCEPT", "risk_score": 0.180},
            "judge":  {"verdict": "FLAG",   "risk_score": 0.300},
            "expect": "FLAG",
        },
        {
            "label":  "Confirmed hallucination",
            "cosine": {"verdict": "FLAG",       "risk_score": 0.710},
            "nli":    {"verdict": "REGENERATE", "risk_score": 0.890},
            "judge":  {"verdict": "REGENERATE", "risk_score": 0.812},
            "expect": "REGENERATE",
        },
        {
            "label":  "Low confidence — downgrade REGENERATE to FLAG",
            "cosine": {"verdict": "ACCEPT",     "risk_score": 0.150},
            "nli":    {"verdict": "ACCEPT",     "risk_score": 0.180},
            "judge":  {"verdict": "REGENERATE", "risk_score": 0.600},
            "expect": "FLAG",
        },
    ]

    all_passed = True
    for case in cases:
        print(f"\n--- {case['label']} ---")
        cv     = compute_cross_validation(case["cosine"], case["nli"], case["judge"])
        fusion = compute_fusion(case["cosine"], case["nli"], case["judge"], cv)
        result = make_decision(fusion, cv)

        passed = result["action"] == case["expect"]
        print(f"\n  Expected : {case['expect']}")
        print(f"  Got      : {result['action']}  "
              f"(override={result['override_applied']})")
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    print(f"  {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
    print(f"{'=' * 60}")