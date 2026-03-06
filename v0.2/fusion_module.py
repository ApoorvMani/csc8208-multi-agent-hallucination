"""
fusion_module.py
================
Combines the three independent layer risk scores into a
single weighted fusion score.

Formula (standard weights):
  final_risk = 0.25 × cosine + 0.35 × nli + 0.40 × judge

Why these weights?
  Judge (0.40) — highest weight. Provides the most
  semantically rich signal: multi-dimensional scoring,
  written justifications, reasoning quality assessment.

  NLI (0.35) — second weight. Purpose-built for
  contradiction detection. Catches numeric errors that
  cosine misses. Operates on logical relationship.

  Cosine (0.25) — lowest weight. Fast first-pass filter.
  Good at semantic divergence but blind to numeric errors.
  Supports the other two layers rather than leading.

Adaptive weighting:
  Cross-validation confidence level adjusts weights.
  HIGH confidence   → standard weights
  MODERATE          → agreeing pair's stronger layer gets +0.05
  LOW               → standard weights, flag for human review

  When cosine + NLI agree (without judge), NLI weight is
  boosted rather than cosine because NLI is purpose-built
  for factual contradiction detection and is more reliable
  on the class of errors where cosine and NLI both fire
  (semantic divergence with logical contradiction).

Academic basis:
  Weighted fusion of heterogeneous detectors is standard
  practice in ensemble learning (Dietterich, 2000).
  Applied here to combine three qualitatively different
  hallucination signals into a single calibrated score.
"""


# ── Standard weights ──────────────────────────────────────────────────────────

STANDARD_WEIGHTS = {
    "cosine": 0.25,
    "nli":    0.35,
    "judge":  0.40,
}


def compute_fusion(
    cosine_result:    dict,
    nli_result:       dict,
    judge_result:     dict,
    cross_validation: dict,
) -> dict:
    """
    Fuses three layer risk scores into a single weighted score.
    Applies confidence-level weight adjustments from
    cross-validation module.

    Args:
        cosine_result:    Output from cosine_layer
        nli_result:       Output from nli_layer
        judge_result:     Output from judge_layer
        cross_validation: Output from cross_validation module

    Returns:
        Dict containing fusion score, weights used, and metadata.
    """
    print("\n[Fusion Module] Computing weighted risk score...")

    # ── Extract risk scores ───────────────────────────────────────────────────
    cosine_risk = cosine_result.get("risk_score", 0.0)
    nli_risk    = nli_result.get("risk_score",    0.0)
    judge_risk  = judge_result.get("risk_score",  0.0)

    # ── Get weight adjustment from cross-validation ───────────────────────────
    weight_adjustment = cross_validation.get("weight_adjustment", "standard")
    confidence_level  = cross_validation.get("confidence_level",  "HIGH")
    halluc_type       = cross_validation.get("hallucination_type", "none")
    weight_note       = cross_validation.get("weight_note", "")

    # ── Apply weight adjustments ──────────────────────────────────────────────
    weights = STANDARD_WEIGHTS.copy()

    if weight_adjustment == "boost_nli":
        # NLI + Judge agree, OR Cosine + NLI agree:
        # Boost NLI, reduce cosine (already the weakest layer)
        weights["nli"]    = round(weights["nli"]    + 0.05, 2)
        weights["cosine"] = round(weights["cosine"] - 0.05, 2)

    elif weight_adjustment == "boost_judge":
        # Cosine + Judge agree:
        # Boost judge (richest signal), reduce cosine
        weights["judge"]  = round(weights["judge"]  + 0.05, 2)
        weights["cosine"] = round(weights["cosine"] - 0.05, 2)

    # "standard" and "human_review" use STANDARD_WEIGHTS unchanged

    # Sanity check — weights must sum to 1.0
    weight_sum = round(sum(weights.values()), 2)
    if weight_sum != 1.0:
        # Correct any floating point drift
        weights["judge"] = round(1.0 - weights["cosine"] - weights["nli"], 2)

    # ── Weighted fusion ───────────────────────────────────────────────────────
    final_risk = (
        weights["cosine"] * cosine_risk +
        weights["nli"]    * nli_risk    +
        weights["judge"]  * judge_risk
    )
    final_risk = round(final_risk, 4)

    # ── Print breakdown ───────────────────────────────────────────────────────
    print(f"\n  Risk Scores:")
    print(f"  Layer 1 (Cosine) : {cosine_risk:.4f}  × {weights['cosine']} "
          f"= {weights['cosine'] * cosine_risk:.4f}")
    print(f"  Layer 2 (NLI)    : {nli_risk:.4f}  × {weights['nli']} "
          f"= {weights['nli'] * nli_risk:.4f}")
    print(f"  Layer 3 (Judge)  : {judge_risk:.4f}  × {weights['judge']} "
          f"= {weights['judge'] * judge_risk:.4f}")
    print(f"  {'─' * 48}")
    print(f"  Final Risk Score : {final_risk:.4f}")
    print(f"  Confidence Level : {confidence_level}")
    print(f"  Hallucination    : {halluc_type}")
    if weight_adjustment != "standard":
        print(f"  Weight note      : {weight_note}")

    return {
        "cosine_risk":       cosine_risk,
        "nli_risk":          nli_risk,
        "judge_risk":        judge_risk,
        "weights_used":      weights,
        "weight_sum":        weight_sum,
        "weight_adjustment": weight_adjustment,
        "confidence_level":  confidence_level,
        "hallucination_type": halluc_type,
        "final_risk_score":  final_risk,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from cross_validation import compute_cross_validation

    print("\n" + "=" * 60)
    print("FUSION MODULE — TEST")
    print("=" * 60)

    cases = [
        {
            "label":  "Clean answer — all low risk",
            "cosine": {"verdict": "ACCEPT", "risk_score": 0.042},
            "nli":    {"verdict": "ACCEPT", "risk_score": 0.008},
            "judge":  {"verdict": "ACCEPT", "risk_score": 0.000},
            "expect": 0.012,   # 0.25×0.042 + 0.35×0.008 + 0.40×0.000
            "expect_weights": STANDARD_WEIGHTS,
        },
        {
            "label":  "Confirmed hallucination — all high risk",
            "cosine": {"verdict": "FLAG",       "risk_score": 0.710},
            "nli":    {"verdict": "REGENERATE", "risk_score": 0.890},
            "judge":  {"verdict": "REGENERATE", "risk_score": 0.812},
            "expect": 0.819,
            "expect_weights": STANDARD_WEIGHTS,
        },
        {
            "label":  "NLI + Judge agree — NLI weight boosted",
            "cosine": {"verdict": "ACCEPT", "risk_score": 0.150},
            "nli":    {"verdict": "FLAG",   "risk_score": 0.610},
            "judge":  {"verdict": "FLAG",   "risk_score": 0.500},
            "expect_nli_weight": 0.40,
            "expect_cosine_weight": 0.20,
        },
        {
            "label":  "Cosine + NLI agree — NLI weight boosted (not cosine)",
            "cosine": {"verdict": "FLAG",   "risk_score": 0.350},
            "nli":    {"verdict": "FLAG",   "risk_score": 0.550},
            "judge":  {"verdict": "ACCEPT", "risk_score": 0.150},
            "expect_nli_weight": 0.40,
            "expect_cosine_weight": 0.20,
        },
    ]

    all_passed = True
    for case in cases:
        print(f"\n--- {case['label']} ---")
        cv     = compute_cross_validation(case["cosine"], case["nli"], case["judge"])
        result = compute_fusion(case["cosine"], case["nli"], case["judge"], cv)

        passed = True

        if "expect" in case:
            score_ok = abs(result["final_risk_score"] - case["expect"]) < 0.02
            print(f"\n  Expected score : ~{case['expect']}")
            print(f"  Got            : {result['final_risk_score']}")
            if not score_ok:
                passed = False

        if "expect_nli_weight" in case:
            nli_ok    = result["weights_used"]["nli"]    == case["expect_nli_weight"]
            cosine_ok = result["weights_used"]["cosine"] == case["expect_cosine_weight"]
            print(f"\n  Expected NLI weight    : {case['expect_nli_weight']}")
            print(f"  Got NLI weight         : {result['weights_used']['nli']}")
            print(f"  Expected cosine weight : {case['expect_cosine_weight']}")
            print(f"  Got cosine weight      : {result['weights_used']['cosine']}")
            if not (nli_ok and cosine_ok):
                passed = False

        # Weights must always sum to 1.0
        weight_sum = round(sum(result["weights_used"].values()), 2)
        sum_ok = weight_sum == 1.0
        print(f"  Weights sum to : {weight_sum} {'✅' if sum_ok else '❌'}")
        if not sum_ok:
            passed = False

        print(f"  {'✅ PASS' if passed else '❌ FAIL'}")
        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    print(f"  {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
    print(f"{'=' * 60}")