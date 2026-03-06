"""
trajectory_tracker.py
=====================
Records the risk score after each detection layer and
after regeneration (if triggered).

Produces a confidence trajectory per pipeline run:
  Layer 1 (Cosine) : X.XX
  Layer 2 (NLI)    : X.XX
  Layer 3 (Judge)  : X.XX
  Fused            : X.XX
  Post-Regen       : X.XX  (if regeneration was triggered)

Why track trajectory?
  If all three layers produce identical scores, they are
  redundant — one layer would be sufficient.
  If the trajectory shows a step-change between layers,
  the layers are complementary — each catches something
  the previous layer missed.

  The Armstrong/1971 case demonstrates this perfectly:
    Layer 1 (Cosine) : 0.073  ACCEPT  ← missed numeric error
    Layer 2 (NLI)    : 0.612  FLAG    ← caught the contradiction
    Layer 3 (Judge)  : 0.200  ACCEPT  ← borderline
    Fused            : 0.295  FLAG    ← fusion catches it

  The step-change at Layer 2 proves NLI adds value beyond
  cosine similarity. Without the trajectory, this would not
  be visible. This feeds Figure 2 (line graph) in the
  evaluation section and directly justifies the multi-layer
  design decision.

Post-regeneration tracking:
  When regeneration is triggered, the corrected answer is
  re-scored through all three layers. The post-regen score
  is recorded here as the trajectory endpoint.

  IMPORTANT: regeneration_module.py must include the key
  "post_regen_risk_score" in its return dict for this to
  populate correctly. See regeneration_module.py.

Academic justification:
  Confidence trajectory proves layers are complementary,
  not redundant. This is the empirical evidence for the
  multi-layer design decision and directly supports the
  claim that triple-layer detection outperforms any
  single-layer approach.
"""


def build_trajectory(
    cosine_result:       dict,
    nli_result:          dict,
    judge_result:        dict,
    fusion_result:       dict,
    regeneration_result: dict = None,
) -> dict:
    """
    Builds the confidence trajectory for one pipeline run.

    Args:
        cosine_result:       Output from cosine_layer
        nli_result:          Output from nli_layer
        judge_result:        Output from judge_layer
        fusion_result:       Output from fusion_module
        regeneration_result: Output from regeneration_module.
                             Must contain "post_regen_risk_score"
                             if regeneration was triggered.
                             Pass None if not triggered.

    Returns:
        Dict containing trajectory scores, step changes,
        and significant change annotations.
    """
    # ── Extract scores ────────────────────────────────────────────────────────
    layer1_score = round(cosine_result.get("risk_score",      0.0), 4)
    layer2_score = round(nli_result.get("risk_score",          0.0), 4)
    layer3_score = round(judge_result.get("risk_score",        0.0), 4)
    fused_score  = round(fusion_result.get("final_risk_score", 0.0), 4)

    # Post-regeneration score
    # Populated only if regeneration was triggered AND
    # regeneration_module returned "post_regen_risk_score"
    post_regen_score = None
    regen_triggered  = False

    if regeneration_result:
        regen_triggered = regeneration_result.get("regeneration_triggered", False)
        if regen_triggered:
            post_regen_score = regeneration_result.get("post_regen_risk_score")
            if post_regen_score is None:
                print("[Trajectory] WARNING: regeneration_triggered=True but "
                      "post_regen_risk_score not found in regeneration_result. "
                      "Check regeneration_module.py return dict.")

    # ── Step changes between layers ───────────────────────────────────────────
    # Positive delta = risk increased (layer found more hallucination evidence)
    # Negative delta = risk decreased (layer found less evidence)
    delta_l1_l2    = round(layer2_score - layer1_score, 4)
    delta_l2_l3    = round(layer3_score - layer2_score, 4)
    delta_l3_fused = round(fused_score  - layer3_score, 4)

    delta_regen = None
    if post_regen_score is not None:
        delta_regen = round(post_regen_score - fused_score, 4)

    # ── Significant step changes (>0.10 threshold) ────────────────────────────
    # A step change > 0.10 indicates a layer added meaningful detection signal
    # beyond the previous layer — evidence that layers are complementary.
    significant_changes = []

    if abs(delta_l1_l2) > 0.10:
        direction = "↑ increase" if delta_l1_l2 > 0 else "↓ decrease"
        significant_changes.append(
            f"NLI {direction} {abs(delta_l1_l2):.3f} vs Cosine "
            f"({'NLI caught what cosine missed' if delta_l1_l2 > 0 else 'NLI more lenient'})"
        )

    if abs(delta_l2_l3) > 0.10:
        direction = "↑ increase" if delta_l2_l3 > 0 else "↓ decrease"
        significant_changes.append(
            f"Judge {direction} {abs(delta_l2_l3):.3f} vs NLI "
            f"({'Judge caught what NLI missed' if delta_l2_l3 > 0 else 'Judge more lenient'})"
        )

    if delta_regen is not None and abs(delta_regen) > 0.10:
        direction = "↓ improved" if delta_regen < 0 else "↑ worsened"
        significant_changes.append(
            f"Regeneration {direction} risk by {abs(delta_regen):.3f}"
        )

    # ── Print trajectory ──────────────────────────────────────────────────────
    print("\n[Trajectory] Confidence trajectory:")

    BAR_MAX = 30

    def bar(score):
        score  = max(0.0, min(1.0, score))
        filled = int(score * BAR_MAX)
        return "█" * filled + "░" * (BAR_MAX - filled)

    print(f"  Layer 1 Cosine : {layer1_score:.4f}  {bar(layer1_score)}")
    print(f"  Layer 2 NLI    : {layer2_score:.4f}  {bar(layer2_score)}"
          f"  Δ={delta_l1_l2:+.4f}")
    print(f"  Layer 3 Judge  : {layer3_score:.4f}  {bar(layer3_score)}"
          f"  Δ={delta_l2_l3:+.4f}")
    print(f"  Fused          : {fused_score:.4f}  {bar(fused_score)}"
          f"  Δ={delta_l3_fused:+.4f}")

    if post_regen_score is not None:
        print(f"  Post-Regen     : {post_regen_score:.4f}  {bar(post_regen_score)}"
              f"  Δ={delta_regen:+.4f}")

    if significant_changes:
        print(f"\n  Significant step changes (>0.10):")
        for change in significant_changes:
            print(f"    → {change}")
    else:
        print(f"\n  No significant step changes — layers consistent")

    return {
        "layer1_cosine":      layer1_score,
        "layer2_nli":         layer2_score,
        "layer3_judge":       layer3_score,
        "fused":              fused_score,
        "post_regen":         post_regen_score,
        "regen_triggered":    regen_triggered,
        "delta_l1_l2":        delta_l1_l2,
        "delta_l2_l3":        delta_l2_l3,
        "delta_l3_fused":     delta_l3_fused,
        "delta_regen":        delta_regen,
        "significant_changes": significant_changes,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from cross_validation import compute_cross_validation
    from fusion_module    import compute_fusion

    print("\n" + "=" * 60)
    print("TRAJECTORY TRACKER — TEST")
    print("=" * 60)

    all_passed = True

    # ── Scenario 1: Armstrong/1971 — NLI step-change expected ────────────────
    print("\n--- Scenario 1: Armstrong 1971 (NLI step-change expected) ---")

    mock_cosine = {"verdict": "ACCEPT", "risk_score": 0.073}
    mock_nli    = {"verdict": "FLAG",   "risk_score": 0.612}
    mock_judge  = {"verdict": "ACCEPT", "risk_score": 0.200}

    cv     = compute_cross_validation(mock_cosine, mock_nli, mock_judge)
    fusion = compute_fusion(mock_cosine, mock_nli, mock_judge, cv)
    traj   = build_trajectory(mock_cosine, mock_nli, mock_judge, fusion)

    step_change_detected = abs(traj["delta_l1_l2"]) > 0.10
    print(f"\n  NLI step change  : {traj['delta_l1_l2']:+.4f}")
    print(f"  Step > 0.10      : {step_change_detected}")
    print(f"  Significant      : {traj['significant_changes']}")
    print(f"  {'✅ PASS' if step_change_detected else '❌ FAIL'}")
    if not step_change_detected:
        all_passed = False

    # ── Scenario 2: Clean answer — flat trajectory ────────────────────────────
    print("\n\n--- Scenario 2: Clean answer (no significant changes expected) ---")

    mock_cosine = {"verdict": "ACCEPT", "risk_score": 0.042}
    mock_nli    = {"verdict": "ACCEPT", "risk_score": 0.008}
    mock_judge  = {"verdict": "ACCEPT", "risk_score": 0.000}

    cv     = compute_cross_validation(mock_cosine, mock_nli, mock_judge)
    fusion = compute_fusion(mock_cosine, mock_nli, mock_judge, cv)
    traj   = build_trajectory(mock_cosine, mock_nli, mock_judge, fusion)

    no_changes = len(traj["significant_changes"]) == 0
    print(f"\n  Significant changes: {traj['significant_changes']}")
    print(f"  {'✅ PASS' if no_changes else '❌ FAIL'}")
    if not no_changes:
        all_passed = False

    # ── Scenario 3: Regeneration improvement ─────────────────────────────────
    print("\n\n--- Scenario 3: With regeneration (large negative delta expected) ---")

    mock_cosine = {"verdict": "FLAG",       "risk_score": 0.710}
    mock_nli    = {"verdict": "REGENERATE", "risk_score": 0.890}
    mock_judge  = {"verdict": "REGENERATE", "risk_score": 0.812}
    mock_regen  = {
        "regeneration_triggered": True,
        "post_regen_risk_score":  0.085,
        "outcome":                "IMPROVED",
    }

    cv     = compute_cross_validation(mock_cosine, mock_nli, mock_judge)
    fusion = compute_fusion(mock_cosine, mock_nli, mock_judge, cv)
    traj   = build_trajectory(mock_cosine, mock_nli, mock_judge, fusion, mock_regen)

    improvement = traj["delta_regen"] is not None and traj["delta_regen"] < -0.50
    print(f"\n  Pre-regen fused : {traj['fused']:.4f}")
    print(f"  Post-regen      : {traj['post_regen']:.4f}")
    print(f"  Delta           : {traj['delta_regen']:+.4f}")
    print(f"  {'✅ PASS — significant improvement' if improvement else '❌ FAIL'}")
    if not improvement:
        all_passed = False

    # ── Scenario 4: Missing post_regen_risk_score warning ────────────────────
    print("\n\n--- Scenario 4: Missing post_regen_risk_score (warning expected) ---")

    mock_regen_missing = {
        "regeneration_triggered": True,
        # post_regen_risk_score intentionally missing
    }

    traj = build_trajectory(mock_cosine, mock_nli, mock_judge, fusion, mock_regen_missing)
    key_missing = traj["post_regen"] is None
    print(f"\n  post_regen is None: {key_missing}")
    print(f"  {'✅ PASS — warning printed, graceful handling' if key_missing else '❌ FAIL'}")
    if not key_missing:
        all_passed = False

    print(f"\n{'=' * 60}")
    print(f"  {'✅ All tests passed' if all_passed else '❌ Some tests failed'}")
    print(f"{'=' * 60}")