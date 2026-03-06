"""
cross_validation.py
===================
Novel contribution of this framework.

Compares the three independent layer verdicts and determines:
  1. Confidence level (HIGH / MODERATE / LOW)
  2. Hallucination type (what kind of error was detected)

Why cross-validation?
  Each detection layer has different strengths and blindspots:

  Layer 1 (Cosine):  Catches semantic divergence — completely
                     wrong answers, wrong person, wrong topic.
                     Misses: numeric errors (1969 vs 1971).

  Layer 2 (NLI):     Catches factual contradictions — including
                     numeric errors cosine misses.
                     Misses: subtle reasoning flaws where the
                     answer is logically coherent but wrong.

  Layer 3 (Judge):   Catches reasoning flaws, subtle errors,
                     incomplete answers, and fabricated context.
                     Most comprehensive but slowest.

  When layers AGREE — confidence is HIGH.
  The same error was independently detected by multiple
  different methods, reducing the chance of false positive.

  When layers DISAGREE — the pattern reveals the ERROR TYPE.
  Which layer flagged tells us what kind of hallucination
  it is. This is the novel contribution: not just detecting
  hallucination but CHARACTERISING it.

Hallucination type taxonomy:
  none                   → no layer flagged, answer likely correct
  semantic divergence    → only cosine flagged
                           answer is semantically different from
                           verifiers — completely wrong entity/topic
  factual contradiction  → only NLI flagged
                           answer contradicts verifiers on a
                           specific fact (date, number, name)
  reasoning flaw         → only judge flagged
                           answer is semantically close and passes
                           NLI but reasoning is flawed or incomplete
  clear factual error    → cosine + NLI both flagged
                           strong signal: both semantic and logical
                           contradiction detected
  semantic and reasoning → cosine + judge flagged
                           semantically divergent AND reasoning flaw
  confirmed factual error→ NLI + judge flagged
                           logical contradiction confirmed by panel
  confirmed hallucination→ all 3 layers flagged
                           highest confidence, most severe case

Academic basis:
  This cross-validation approach extends the multi-layer
  detection principle. No existing paper in the reviewed
  literature characterises hallucination type through
  cross-layer triangulation. This is the primary novel
  academic contribution of this framework.
"""


# ── Confidence thresholds ─────────────────────────────────────────────────────

HIGH_CONFIDENCE_THRESHOLD    = 3   # all 3 layers agree
MODERATE_CONFIDENCE_THRESHOLD = 2  # any 2 layers agree


# ── Hallucination type map ────────────────────────────────────────────────────
# Maps frozenset of flagging layers → hallucination type label
# Covers all 8 possible combinations (2^3 subsets of 3 layers)

HALLUCINATION_TYPE_MAP = {
    frozenset():                           "none",
    frozenset(["cosine"]):                 "semantic divergence",
    frozenset(["nli"]):                    "factual contradiction",
    frozenset(["judge"]):                  "reasoning flaw",
    frozenset(["cosine", "nli"]):          "clear factual error",
    frozenset(["cosine", "judge"]):        "semantic and reasoning error",
    frozenset(["nli", "judge"]):           "confirmed factual error",
    frozenset(["cosine", "nli", "judge"]): "confirmed hallucination",
}


def compute_cross_validation(
    cosine_result: dict,
    nli_result:    dict,
    judge_result:  dict,
) -> dict:
    """
    Compares three independent layer verdicts.
    Determines confidence level and hallucination type.

    Args:
        cosine_result: Output from cosine_layer.compute_cosine_layer()
        nli_result:    Output from nli_layer.compute_nli_layer()
        judge_result:  Output from judge_layer.compute_judge_layer()

    Returns:
        Dict containing cross-validation analysis.
    """
    print("\n[Cross-Validation] Comparing layer verdicts...")

    # ── Extract verdicts ──────────────────────────────────────────────────────
    cosine_verdict = cosine_result.get("verdict", "ACCEPT")
    nli_verdict    = nli_result.get("verdict",    "ACCEPT")
    judge_verdict  = judge_result.get("verdict",  "ACCEPT")

    layer_verdicts = {
        "cosine": cosine_verdict,
        "nli":    nli_verdict,
        "judge":  judge_verdict,
    }

    # Extract risk scores for reference
    layer_risk_scores = {
        "cosine": cosine_result.get("risk_score", 0.0),
        "nli":    nli_result.get("risk_score",    0.0),
        "judge":  judge_result.get("risk_score",  0.0),
    }

    print(f"  Layer 1 (Cosine): {cosine_verdict:<12} "
          f"risk={layer_risk_scores['cosine']:.4f}")
    print(f"  Layer 2 (NLI)   : {nli_verdict:<12} "
          f"risk={layer_risk_scores['nli']:.4f}")
    print(f"  Layer 3 (Judge) : {judge_verdict:<12} "
          f"risk={layer_risk_scores['judge']:.4f}")

    # ── Identify which layers flagged ─────────────────────────────────────────
    # A layer "flagged" if its verdict is FLAG or REGENERATE
    flagging_layers = set()
    for layer_name, verdict in layer_verdicts.items():
        if verdict in ("FLAG", "REGENERATE"):
            flagging_layers.add(layer_name)

    n_flagging = len(flagging_layers)

    # ── Confidence level ──────────────────────────────────────────────────────
    if n_flagging == 0:
        confidence_level = "HIGH"
        confidence_label = "HIGH confidence ACCEPT — no layer detected issues"
    elif n_flagging == HIGH_CONFIDENCE_THRESHOLD:
        confidence_level = "HIGH"
        confidence_label = "HIGH confidence FLAG — all 3 layers independently flagged"
    elif n_flagging == MODERATE_CONFIDENCE_THRESHOLD:
        confidence_level = "MODERATE"
        confidence_label = f"MODERATE confidence — {n_flagging}/3 layers flagged"
    else:
        # n_flagging == 1
        confidence_level = "LOW"
        confidence_label = "LOW confidence — only 1 layer flagged (human review recommended)"

    # ── Hallucination type ────────────────────────────────────────────────────
    hallucination_type = HALLUCINATION_TYPE_MAP.get(
        frozenset(flagging_layers),
        "unknown pattern"
    )

    # ── Overall pattern string ────────────────────────────────────────────────
    if not flagging_layers:
        pattern = "ACCEPT | ACCEPT | ACCEPT"
    else:
        parts = []
        for layer in ["cosine", "nli", "judge"]:
            verdict = layer_verdicts[layer]
            if verdict == "ACCEPT":
                parts.append(f"{layer.upper()}:ACCEPT")
            else:
                parts.append(f"{layer.upper()}:{verdict}")
        pattern = " | ".join(parts)

    # ── Adaptive weight recommendation ────────────────────────────────────────
    # Cross-validation adjusts fusion weights based on which layers agree.
    # HIGH confidence   → standard weights (0.25 / 0.35 / 0.40)
    # MODERATE          → boost the agreeing pair's stronger layer
    # LOW               → flag for human review, use standard weights
    #
    # Cosine + NLI agreement (without judge) defaults to boost_nli because:
    # NLI is purpose-built for contradiction detection and is more reliable
    # than cosine on factual errors. Boosting NLI over cosine reflects this.

    if confidence_level == "HIGH":
        weight_adjustment = "standard"
        weight_note = "Standard weights apply (0.25 / 0.35 / 0.40)"

    elif confidence_level == "MODERATE":
        if "nli" in flagging_layers and "judge" in flagging_layers:
            # NLI + Judge agree — strongest combination
            weight_adjustment = "boost_nli"
            weight_note = "NLI weight boosted (+0.05) — NLI + Judge agree"
        elif "cosine" in flagging_layers and "judge" in flagging_layers:
            # Cosine + Judge agree — judge is the stronger signal
            weight_adjustment = "boost_judge"
            weight_note = "Judge weight boosted (+0.05) — Cosine + Judge agree"
        else:
            # Cosine + NLI agree (without judge)
            # Boost NLI: more reliable than cosine on factual errors
            weight_adjustment = "boost_nli"
            weight_note = "NLI weight boosted (+0.05) — Cosine + NLI agree (NLI more reliable)"

    else:
        # LOW confidence — single layer flagged
        weight_adjustment = "human_review"
        weight_note = "Low confidence — human review recommended"

    # ── Print results ─────────────────────────────────────────────────────────
    print(f"\n  Flagging layers  : "
          f"{list(flagging_layers) if flagging_layers else 'none'}")
    print(f"  Layers flagged   : {n_flagging} / 3")
    print(f"  Confidence level : {confidence_level}")
    print(f"  Hallucination type: {hallucination_type}")
    print(f"  Pattern          : {pattern}")
    print(f"  Weight adjustment: {weight_note}")

    icon = {
        "HIGH":     "✅" if n_flagging == 0 else "🚨",
        "MODERATE": "⚠️",
        "LOW":      "🔍",
    }.get(confidence_level, "?")

    print(f"\n  {icon} {confidence_label}")
    print(f"  Type: {hallucination_type.upper()}")

    return {
        "layer_verdicts":     layer_verdicts,
        "layer_risk_scores":  layer_risk_scores,
        "flagging_layers":    list(flagging_layers),
        "n_flagging":         n_flagging,
        "confidence_level":   confidence_level,
        "confidence_label":   confidence_label,
        "hallucination_type": hallucination_type,
        "pattern":            pattern,
        "weight_adjustment":  weight_adjustment,
        "weight_note":        weight_note,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CROSS-VALIDATION MODULE — TEST")
    print("=" * 60)

    cases = [
        {
            "label":   "All layers ACCEPT",
            "cosine":  {"verdict": "ACCEPT", "risk_score": 0.042},
            "nli":     {"verdict": "ACCEPT", "risk_score": 0.008},
            "judge":   {"verdict": "ACCEPT", "risk_score": 0.000},
            "expect_conf": "HIGH",
            "expect_type": "none",
        },
        {
            "label":   "Only NLI flags — numeric error",
            "cosine":  {"verdict": "ACCEPT", "risk_score": 0.073},
            "nli":     {"verdict": "FLAG",   "risk_score": 0.612},
            "judge":   {"verdict": "ACCEPT", "risk_score": 0.200},
            "expect_conf": "LOW",
            "expect_type": "factual contradiction",
        },
        {
            "label":   "All layers FLAG — confirmed hallucination",
            "cosine":  {"verdict": "FLAG",       "risk_score": 0.710},
            "nli":     {"verdict": "REGENERATE", "risk_score": 0.890},
            "judge":   {"verdict": "REGENERATE", "risk_score": 0.812},
            "expect_conf": "HIGH",
            "expect_type": "confirmed hallucination",
        },
        {
            "label":   "Only judge flags — reasoning flaw",
            "cosine":  {"verdict": "ACCEPT", "risk_score": 0.150},
            "nli":     {"verdict": "ACCEPT", "risk_score": 0.180},
            "judge":   {"verdict": "FLAG",   "risk_score": 0.400},
            "expect_conf": "LOW",
            "expect_type": "reasoning flaw",
        },
        {
            "label":   "Cosine + NLI flag — clear factual error",
            "cosine":  {"verdict": "FLAG",   "risk_score": 0.350},
            "nli":     {"verdict": "FLAG",   "risk_score": 0.550},
            "judge":   {"verdict": "ACCEPT", "risk_score": 0.150},
            "expect_conf": "MODERATE",
            "expect_type": "clear factual error",
        },
        {
            "label":   "NLI + Judge flag — confirmed factual error",
            "cosine":  {"verdict": "ACCEPT",     "risk_score": 0.100},
            "nli":     {"verdict": "FLAG",        "risk_score": 0.650},
            "judge":   {"verdict": "REGENERATE", "risk_score": 0.700},
            "expect_conf": "MODERATE",
            "expect_type": "confirmed factual error",
        },
    ]

    all_passed = True
    for case in cases:
        print(f"\n--- {case['label']} ---")
        result = compute_cross_validation(
            case["cosine"], case["nli"], case["judge"]
        )
        conf_ok = result["confidence_level"]   == case["expect_conf"]
        type_ok = result["hallucination_type"] == case["expect_type"]
        passed  = conf_ok and type_ok

        print(f"\n  Expected: {case['expect_conf']} — {case['expect_type']}")
        print(f"  Got     : {result['confidence_level']} — "
              f"{result['hallucination_type']}")
        print(f"  {'✅ PASS' if passed else '❌ FAIL'}")

        if not passed:
            all_passed = False

    print(f"\n{'=' * 60}")
    print(f"  {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
    print(f"{'=' * 60}")