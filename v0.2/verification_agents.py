"""
verification_agents.py
======================
Runs all 9 verification agents independently against a question.

Key design principle — STRICT INDEPENDENCE:
  No agent sees any other agent's answer during generation.
  This eliminates the echo-chamber convergence risk present
  in debate-based systems (Du et al., 2023) where agents
  can reinforce each other's hallucinations before voting.

  Agents only see each other's answers AFTER all votes are
  cast and the risk score is already computed. This ensures
  divergence between agents is a genuine independent signal
  of epistemic uncertainty — not social influence.

Agent diversity strategy:
  - 5 real model architectures (llama3.2, qwen2.5, deepseek-r1,
    gemma2, phi3) — architectural diversity decorrelates outputs
  - 3 temperature variants — behavioural diversity
  - 1 Byzantine adversarial agent — resilience testing

API routing:
  All agents currently use Ollama (local inference).
  When external APIs are added, this module routes
  automatically based on agent_registry provider field.

Academic basis:
  Manakul et al. (2023) SelfCheckGPT — sampling multiple
  responses to measure self-consistency. This framework
  extends that principle to multi-model multi-style sampling.
"""

import ollama
import hashlib
import datetime
import time
from agent_registry import get_verifiers, get_agent


# ── API router ────────────────────────────────────────────────────────────────

def call_agent(agent_id: str, question: str) -> str:
    """
    Routes a question to the correct provider based on agent registry.
    Currently supports Ollama (local).
    Future: openai, anthropic, grok, google, mistral, cohere.

    Args:
        agent_id: Agent ID from registry
        question:  The question to answer

    Returns:
        Raw answer string from the model
    """
    config = get_agent(agent_id)
    provider = config["provider"]

    if provider == "ollama":
        return _call_ollama(config, question)

    # ── Future API providers (uncomment when ready) ────────────────────────
    # elif provider == "openai":
    #     return _call_openai(config, question)
    # elif provider == "anthropic":
    #     return _call_anthropic(config, question)
    # elif provider == "grok":
    #     return _call_grok(config, question)
    # elif provider == "google":
    #     return _call_google(config, question)

    else:
        raise ValueError(
            f"Unknown provider '{provider}' for agent '{agent_id}'. "
            f"Supported: ollama. Future: openai, anthropic, grok, google."
        )


def _call_ollama(config: dict, question: str) -> str:
    """Calls a local Ollama model."""
    response = ollama.chat(
        model=config["model"],
        messages=[
            {
                "role": "system",
                "content": config["system_prompt"],
            },
            {
                "role": "user",
                "content": question,
            },
        ],
        options={
            "temperature": config["temperature"],
        }
    )
    return response["message"]["content"].strip()


# ── Future provider stubs ─────────────────────────────────────────────────────

# def _call_openai(config: dict, question: str) -> str:
#     import openai
#     client = openai.OpenAI(api_key=os.getenv(config["api_key_env"]))
#     response = client.chat.completions.create(
#         model=config["model"],
#         messages=[
#             {"role": "system", "content": config["system_prompt"]},
#             {"role": "user", "content": question},
#         ],
#         temperature=config["temperature"],
#     )
#     return response.choices[0].message.content.strip()

# def _call_anthropic(config: dict, question: str) -> str:
#     import anthropic
#     client = anthropic.Anthropic(api_key=os.getenv(config["api_key_env"]))
#     response = client.messages.create(
#         model=config["model"],
#         max_tokens=1024,
#         system=config["system_prompt"],
#         messages=[{"role": "user", "content": question}],
#     )
#     return response.content[0].text.strip()

# def _call_grok(config: dict, question: str) -> str:
#     import openai  # Grok uses OpenAI-compatible API
#     client = openai.OpenAI(
#         api_key=os.getenv(config["api_key_env"]),
#         base_url="https://api.x.ai/v1",
#     )
#     response = client.chat.completions.create(
#         model=config["model"],
#         messages=[
#             {"role": "system", "content": config["system_prompt"]},
#             {"role": "user", "content": question},
#         ],
#         temperature=config["temperature"],
#     )
#     return response.choices[0].message.content.strip()


# ── Single agent runner ───────────────────────────────────────────────────────

def run_verification_agent(agent_id: str, question: str) -> dict:
    """
    Runs a single verification agent and returns a structured result.

    Args:
        agent_id: Agent ID from registry
        question:  The question to answer

    Returns:
        Dict with agent metadata, answer, hash, and timing
    """
    config = get_agent(agent_id)

    print(f"  [{agent_id}] ({config['style']} style) "
          f"querying {config['model']}...")

    start_time = time.time()

    try:
        answer = call_agent(agent_id, question)
        elapsed = round(time.time() - start_time, 2)
        answer_hash = hashlib.sha256(answer.encode()).hexdigest()

        print(f"  [{agent_id}] Answer: {answer[:80]}...")

        return {
            "agent":        agent_id,
            "model":        config["model"],
            "provider":     config["provider"],
            "temperature":  config["temperature"],
            "style":        config["style"],
            "role":         config["role"],
            "trust_weight": config["trust_weight"],
            "question":     question,
            "answer":       answer,
            "hash":         answer_hash,
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "elapsed_s":    elapsed,
            "error":        None,
            "adversarial":  config.get("adversarial", False),
        }

    except Exception as e:
        elapsed = round(time.time() - start_time, 2)
        print(f"  [{agent_id}] ERROR: {e}")
        return {
            "agent":        agent_id,
            "model":        config["model"],
            "provider":     config["provider"],
            "temperature":  config["temperature"],
            "style":        config["style"],
            "role":         config["role"],
            "trust_weight": config["trust_weight"],
            "question":     question,
            "answer":       "",
            "hash":         "",
            "timestamp":    datetime.datetime.utcnow().isoformat(),
            "elapsed_s":    elapsed,
            "error":        str(e),
            "adversarial":  config.get("adversarial", False),
        }


# ── Full verification layer ───────────────────────────────────────────────────

def run_all_verification_agents(
    question: str,
    include_byzantine: bool = True,
) -> list:
    """
    Runs all 9 verification agents independently against the question.

    Agents answer in strict isolation — no agent sees any other
    agent's answer at this stage. Independence is enforced by
    sequential execution with no shared state between calls.

    Args:
        question:           The question to verify
        include_byzantine:  Whether to include the Byzantine agent.
                            Set False for baseline (non-adversarial) runs.

    Returns:
        List of result dicts, one per agent.
        Failed agents are included with error field set.
    """
    verifiers = get_verifiers()

    print(f"\n[Verification Layer] Running {len(verifiers)} agents...")
    print(f'  Question: "{question}"')
    print("-" * 60)

    results = []
    failed = 0

    for agent_id, config in verifiers.items():

        # Skip Byzantine agent if not requested
        if not include_byzantine and config.get("adversarial", False):
            print(f"  [{agent_id}] Skipping (Byzantine disabled)")
            continue

        result = run_verification_agent(agent_id, question)
        results.append(result)

        if result["error"]:
            failed += 1

    # Summary
    successful = len(results) - failed
    print(f"\n[Verification Layer] Complete.")
    print(f"  Successful: {successful} / {len(results)}")
    if failed > 0:
        print(f"  Failed:     {failed} agents (check model availability)")

    return results


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from agent_registry import print_roster

    print_roster()

    # Neutral test question (per Mujeeb's guidance — no controversial topics)
    question = (
        "What is the speed of light and what unit is it measured in?"
    )

    print(f"\nTest question: {question}\n")

    # Run without Byzantine agent first (clean baseline)
    results = run_all_verification_agents(
        question,
        include_byzantine=False,
    )

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        status = "✅" if not r["error"] else "❌"
        print(f"  {status} {r['agent']:<12} "
              f"{r['model']:<14} "
              f"temp={r['temperature']}  "
              f"{r['elapsed_s']}s")
        if r["answer"]:
            print(f"     Answer: {r['answer'][:100]}...")
        if r["error"]:
            print(f"     Error:  {r['error']}")
    print()

    # Now run WITH Byzantine agent
    print("\n--- Now running WITH Byzantine agent ---\n")
    results_with_byzantine = run_all_verification_agents(
        question,
        include_byzantine=True,
    )

    print("\nByzantine agent answer:")
    for r in results_with_byzantine:
        if r.get("adversarial"):
            print(f"  {r['agent']}: {r['answer'][:150]}")