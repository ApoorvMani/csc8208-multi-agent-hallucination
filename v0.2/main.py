"""
main.py
=======
Full pipeline entry point for the Hallucination Detection Framework v0.2.

Usage:
  python main.py
    → Interactive mode: prompts for a question

  python main.py --question "What is the boiling point of water?"
    → Runs the full pipeline on the given question

  python main.py --simulate "Thomas Edison invented the telephone in 1877."
    → Injects a known hallucination as the primary answer.
      Verifiers still run normally. Tests detection + regeneration.

  python main.py --question "..." --topology ring
    → Runs with ring topology instead of star (default)
      Choices: star, ring, complete

  python main.py --question "..." --no-byzantine
    → Excludes the Byzantine adversarial agent from verification

  python main.py --question "..." --no-regen
    → Disables regeneration even if FLAG or REGENERATE is triggered

  python main.py --validate-chain
    → Validates the blockchain audit log for tampering. No inference run.

Pipeline stages (in order):
  1.  Primary agent         → query_primary_agent()
  2.  Verification agents   → run_all_verification_agents()
  3.  Layer 1: Cosine       → compute_cosine_layer()
  4.  Layer 2: NLI          → compute_nli_layer()
  5.  Layer 3: Judge panel  → compute_judge_layer()
  6.  Cross-validation      → compute_cross_validation()
  7.  Fusion                → compute_fusion()
  8.  Decision engine       → make_decision()
  9.  Trajectory tracker    → build_trajectory()
  10. Regeneration          → run_regeneration()  [if FLAG or REGENERATE]
  11. Explanation generator → generate_explanation()
  12. Blockchain logger     → build_audit_record() + blockchain.add_block()

CSC8208 Newcastle University — MSc Cybersecurity
Module: CSC8208 | Assessment 2 | 2025/2026
"""

import argparse
import datetime
import hashlib
import sys
import time

# ── Pipeline imports ──────────────────────────────────────────────────────────
from agent_registry       import print_roster
from primary_agent        import query_primary_agent
from verification_agents  import run_all_verification_agents
from cosine_layer         import compute_cosine_layer
from nli_layer            import compute_nli_layer
from judge_layer          import compute_judge_layer
from cross_validation     import compute_cross_validation
from fusion_module        import compute_fusion
from decision_engine      import make_decision
from trajectory_tracker   import build_trajectory
from regeneration_module  import run_regeneration
from explanation_generator import generate_explanation, print_explanation
from blockchain_logger    import Blockchain, build_audit_record
from topology_manager     import describe_topology


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════════════════════╗
║       HALLUCINATION DETECTION FRAMEWORK  v0.2                       ║
║       Multi-Agent Aggregation and Voting System                     ║
║       CSC8208 — Newcastle University — MSc Cybersecurity            ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run_pipeline(
    question:          str,
    topology:          str  = "star",
    include_byzantine: bool = True,
    allow_regen:       bool = True,
    simulated_answer:  str  = None,
) -> dict:
    """
    Runs the complete hallucination detection pipeline.

    Args:
        question:          The question to evaluate.
        topology:          "star", "ring", or "complete".
        include_byzantine: Whether to include the Byzantine adversarial agent.
        allow_regen:       Whether to run regeneration on FLAG/REGENERATE.
        simulated_answer:  If set, injects this as the primary agent answer
                           instead of querying the model. Used for controlled
                           hallucination injection tests.

    Returns:
        Full pipeline result dict containing all stage outputs.
    """
    pipeline_start = time.time()

    print(BANNER)
    print(f"  Question  : {question}")
    print(f"  Topology  : {topology.upper()} — {describe_topology(topology)}")
    print(f"  Byzantine : {'Enabled' if include_byzantine else 'Disabled'}")
    print(f"  Regen     : {'Enabled' if allow_regen else 'Disabled'}")
    if simulated_answer:
        print(f"  Mode      : SIMULATION — injecting known answer")
    print()

    # ── Stage 1: Primary agent ────────────────────────────────────────────────
    print("━" * 70)
    print("  STAGE 1 — PRIMARY AGENT")
    print("━" * 70)

    if simulated_answer:
        # Inject simulated answer (for controlled hallucination tests)
        sim_hash = hashlib.sha256(
            f"{question}||{simulated_answer}".encode()
        ).hexdigest()

        primary_result = {
            "agent":     "primary",
            "agent_id":  "primary",
            "model":     "mistral",
            "question":  question,
            "answer":    simulated_answer,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "hash":      sim_hash,
        }
        print(f"\n  [SIMULATION] Injected answer: {simulated_answer}")
        print(f"  Hash: {sim_hash[:16]}...")
    else:
        primary_result = query_primary_agent(question)

    # ── Stage 2: Verification agents ──────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 2 — VERIFICATION AGENTS")
    print("━" * 70)

    verification_results = run_all_verification_agents(
        question,
        include_byzantine=include_byzantine,
    )

    valid_verifiers = [
        v for v in verification_results
        if v.get("answer") and not v.get("error")
    ]
    print(f"\n  Valid verifiers: {len(valid_verifiers)} / {len(verification_results)}")

    if len(valid_verifiers) < 2:
        print("\n  ⚠️  CRITICAL: Fewer than 2 verifiers available.")
        print("  Check that Ollama models are running (ollama list).")
        print("  Pipeline cannot produce reliable results with < 2 verifiers.")

    # ── Stage 3: Layer 1 — Cosine similarity ──────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 3 — LAYER 1: COSINE SIMILARITY")
    print("━" * 70)

    cosine_result = compute_cosine_layer(primary_result, verification_results)

    print(f"\n  Agreement score : {cosine_result.get('agreement_score', 0):.4f}")
    print(f"  Risk score      : {cosine_result.get('risk_score', 0):.4f}")
    print(f"  Verdict         : {cosine_result.get('verdict')}")

    # ── Stage 4: Layer 2 — NLI contradiction ──────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 4 — LAYER 2: NLI CONTRADICTION")
    print("━" * 70)

    nli_result = compute_nli_layer(primary_result, verification_results)

    print(f"\n  Contradictions  : {nli_result.get('contradiction_count', 0)}")
    print(f"  Entailments     : {nli_result.get('entailment_count', 0)}")
    print(f"  Risk score      : {nli_result.get('risk_score', 0):.4f}")
    print(f"  Verdict         : {nli_result.get('verdict')}")

    # ── Stage 5: Layer 3 — Consensus panel judge ───────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 5 — LAYER 3: CONSENSUS PANEL JUDGE")
    print("━" * 70)

    judge_result = compute_judge_layer(
        question             = question,
        primary_result       = primary_result,
        verification_results = verification_results,
        topology             = topology,
    )

    # ── Stage 6: Cross-validation ──────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 6 — CROSS-VALIDATION")
    print("━" * 70)

    cv_result = compute_cross_validation(cosine_result, nli_result, judge_result)

    print(f"\n  Pattern           : {cv_result.get('pattern')}")
    print(f"  Confidence level  : {cv_result.get('confidence_level')}")
    print(f"  Hallucination type: {cv_result.get('hallucination_type')}")
    print(f"  Flagging layers   : {cv_result.get('flagging_layers')}")

    # ── Stage 7: Fusion ────────────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 7 — WEIGHTED ADAPTIVE FUSION")
    print("━" * 70)

    fusion_result = compute_fusion(cosine_result, nli_result, judge_result, cv_result)

    weights = fusion_result.get("weights_used", {})
    print(f"\n  Weights used:")
    print(f"    Cosine : {weights.get('cosine', 0.25):.2f}")
    print(f"    NLI    : {weights.get('nli', 0.35):.2f}")
    print(f"    Judge  : {weights.get('judge', 0.40):.2f}")
    print(f"  Final risk score: {fusion_result.get('final_risk_score', 0):.4f}")

    # ── Stage 8: Decision engine ───────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 8 — DECISION ENGINE")
    print("━" * 70)

    decision = make_decision(fusion_result, cv_result)

    # ── Stage 9: Trajectory ────────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 9 — CONFIDENCE TRAJECTORY")
    print("━" * 70)

    trajectory = build_trajectory(
        cosine_result       = cosine_result,
        nli_result          = nli_result,
        judge_result        = judge_result,
        fusion_result       = fusion_result,
        regeneration_result = None,   # updated after regen if triggered
    )

    # ── Stage 10: Regeneration ─────────────────────────────────────────────────
    regeneration_result = None

    if allow_regen and decision.get("action") in ("FLAG", "REGENERATE"):
        print("\n" + "━" * 70)
        print("  STAGE 10 — REGENERATION (SELF-CORRECTION)")
        print("━" * 70)

        regeneration_result = run_regeneration(
            question             = question,
            primary_result       = primary_result,
            verification_results = verification_results,
            cosine_result        = cosine_result,
            nli_result           = nli_result,
            judge_result         = judge_result,
            cross_validation     = cv_result,
            fusion_result        = fusion_result,
            decision             = decision,
            topology             = topology,
        )

        # Rebuild trajectory with post-regen score
        trajectory = build_trajectory(
            cosine_result       = cosine_result,
            nli_result          = nli_result,
            judge_result        = judge_result,
            fusion_result       = fusion_result,
            regeneration_result = regeneration_result,
        )

    else:
        print("\n" + "━" * 70)
        print("  STAGE 10 — REGENERATION")
        print("━" * 70)
        if not allow_regen:
            print("\n  Skipped (--no-regen flag set)")
        else:
            print(f"\n  Not triggered — decision was {decision.get('action')}")

        regeneration_result = {
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

    # ── Stage 11: Explanation ──────────────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 11 — PLAIN ENGLISH EXPLANATION")
    print("━" * 70)

    explanation = generate_explanation(
        decision            = decision,
        cross_validation    = cv_result,
        cosine_result       = cosine_result,
        nli_result          = nli_result,
        judge_result        = judge_result,
        fusion_result       = fusion_result,
        regeneration_result = regeneration_result,
        trajectory          = trajectory,
    )
    print_explanation(explanation)

    # ── Stage 12: Blockchain audit log ─────────────────────────────────────────
    print("\n" + "━" * 70)
    print("  STAGE 12 — BLOCKCHAIN AUDIT LOG")
    print("━" * 70)

    blockchain = Blockchain()

    audit_record = build_audit_record(
        question             = question,
        primary_result       = primary_result,
        verification_results = verification_results,
        cosine_report        = cosine_result,
        nli_report           = nli_result,
        judge_report         = judge_result,
        cross_validation     = cv_result,
        fusion_result        = fusion_result,
        decision             = decision,
        trajectory           = trajectory,
        regeneration_result  = regeneration_result,
        explanation          = explanation,
        topology             = topology,
    )

    new_block = blockchain.add_block(audit_record)

    print(f"\n  Block index  : {new_block.index}")
    print(f"  Block hash   : {new_block.hash[:32]}...")
    print(f"  Chain length : {blockchain.get_length()} blocks")

    # Quick chain validation after every run
    chain_check = blockchain.validate_chain()
    print(f"  Chain status : {chain_check['message']}")

    # ── Pipeline complete ──────────────────────────────────────────────────────
    elapsed = round(time.time() - pipeline_start, 2)

    print("\n" + "═" * 70)
    print("  PIPELINE COMPLETE")
    print("═" * 70)

    # Determine final answer to display
    final_answer = primary_result.get("answer")
    if regeneration_result and regeneration_result.get("regeneration_triggered"):
        final_answer = regeneration_result.get("final_answer", final_answer)
        regen_outcome = regeneration_result.get("outcome", "UNKNOWN")
    else:
        regen_outcome = "NOT_TRIGGERED"

    action     = decision.get("action")
    risk_level = decision.get("risk_level")
    final_risk = fusion_result.get("final_risk_score", 0.0)

    print(f"\n  Question     : {question}")
    print(f"  Final answer : {final_answer[:200]}{'...' if len(final_answer) > 200 else ''}")
    print(f"  Decision     : {action}")
    print(f"  Risk level   : {risk_level}")
    print(f"  Final risk   : {final_risk:.4f}")
    print(f"  Confidence   : {cv_result.get('confidence_level')}")
    print(f"  Halluc type  : {cv_result.get('hallucination_type')}")

    if regen_outcome != "NOT_TRIGGERED":
        print(f"  Regen outcome: {regen_outcome}")
        if regeneration_result.get("improvement_delta") is not None:
            delta = regeneration_result["improvement_delta"]
            print(f"  Risk delta   : {delta:+.4f}")

    print(f"\n  Total time   : {elapsed}s")
    print(f"  Audit block  : #{new_block.index} in audit_blockchain.json")
    print()

    return {
        "question":             question,
        "topology":             topology,
        "primary_result":       primary_result,
        "verification_results": verification_results,
        "cosine_result":        cosine_result,
        "nli_result":           nli_result,
        "judge_result":         judge_result,
        "cv_result":            cv_result,
        "fusion_result":        fusion_result,
        "decision":             decision,
        "trajectory":           trajectory,
        "regeneration_result":  regeneration_result,
        "explanation":          explanation,
        "block_index":          new_block.index,
        "elapsed_s":            elapsed,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Hallucination Detection Framework v0.2\n"
            "CSC8208 — Newcastle University\n\n"
            "Examples:\n"
            "  python main.py\n"
            '  python main.py --question "Who was the first person on the moon?"\n'
            '  python main.py --simulate "Thomas Edison invented the telephone."\n'
            '  python main.py --question "..." --topology ring\n'
            '  python main.py --validate-chain'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--question", "-q",
        type=str,
        default=None,
        help="Question to run through the pipeline.",
    )

    parser.add_argument(
        "--simulate", "-s",
        type=str,
        default=None,
        help=(
            "Inject a known hallucination as the primary agent's answer. "
            "Verification agents still answer normally. "
            "Used for controlled detection testing."
        ),
    )

    parser.add_argument(
        "--topology", "-t",
        type=str,
        default="star",
        choices=["star", "ring", "complete"],
        help="Agent topology for Layer 3 judge panel (default: star).",
    )

    parser.add_argument(
        "--no-byzantine",
        action="store_true",
        help="Exclude the Byzantine adversarial agent from verification.",
    )

    parser.add_argument(
        "--no-regen",
        action="store_true",
        help="Disable regeneration even if FLAG or REGENERATE is triggered.",
    )

    parser.add_argument(
        "--validate-chain",
        action="store_true",
        help=(
            "Validate the blockchain audit log for tampering. "
            "No pipeline run is performed."
        ),
    )

    parser.add_argument(
        "--show-roster",
        action="store_true",
        help="Print the agent roster and exit.",
    )

    return parser.parse_args()


# ── Default test questions ─────────────────────────────────────────────────────
# All neutral — science, geography, history (per Mujeeb's guidance).
# No politics, no controversial figures.

DEFAULT_QUESTIONS = [
    "What is the boiling point of water at sea level in Celsius?",
    "Who was the first person to walk on the moon, and in what year?",
    "What is the speed of light in a vacuum?",
    "Where is the Amazon rainforest located?",
    "What is the chemical symbol for gold?",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Validate chain only ───────────────────────────────────────────────────
    if args.validate_chain:
        print(BANNER)
        print("Validating blockchain audit log...\n")
        bc = Blockchain()
        bc.print_chain_summary()
        result = bc.validate_chain()
        print(f"\n{result['message']}")
        sys.exit(0 if result["valid"] else 1)

    # ── Show roster only ──────────────────────────────────────────────────────
    if args.show_roster:
        print(BANNER)
        print_roster()
        sys.exit(0)

    # ── Resolve question ──────────────────────────────────────────────────────
    question = args.question

    if not question and not args.simulate:
        # Interactive mode
        print(BANNER)
        print("Interactive mode — enter a question to evaluate.\n")
        print("Suggested questions (neutral topics):")
        for i, q in enumerate(DEFAULT_QUESTIONS, 1):
            print(f"  {i}. {q}")
        print()

        user_input = input("Enter your question (or a number 1-5): ").strip()

        if user_input.isdigit():
            idx = int(user_input) - 1
            if 0 <= idx < len(DEFAULT_QUESTIONS):
                question = DEFAULT_QUESTIONS[idx]
                print(f"\nUsing: {question}\n")
            else:
                print("Invalid number. Please run again and choose 1-5.")
                sys.exit(1)
        elif user_input:
            question = user_input
        else:
            print("No question provided. Exiting.")
            sys.exit(1)

    # ── Simulate mode: question required ─────────────────────────────────────
    if args.simulate and not question:
        print(
            "\nSimulation mode requires --question as well.\n"
            "Example:\n"
            '  python main.py --question "Who invented the telephone?" '
            '--simulate "Thomas Edison invented the telephone in 1877."\n'
        )
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────────────────────
    try:
        result = run_pipeline(
            question          = question,
            topology          = args.topology,
            include_byzantine = not args.no_byzantine,
            allow_regen       = not args.no_regen,
            simulated_answer  = args.simulate,
        )
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n  ❌ Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()