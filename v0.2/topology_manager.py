"""
topology_manager.py
===================
Controls information flow between verification agents
during the consensus panel judge phase (Layer 3).

Three topologies are implemented:

  STAR (default)
  ──────────────
  All agents report independently to a central aggregator.
  No agent sees any other agent's answer.
  Fastest. Most independent. Best baseline.
  This is the implicit topology of v0.1.

        [Aggregator]
       / | | | | \ \
      1  2 3 4 5  6 7 8 9

  RING
  ────
  Each agent only sees its two immediate neighbours' answers
  before submitting its final judgement.
  Information travels around the chain.
  Tests: does local information sharing help or hurt detection?
  Malicious agent influence is localised to neighbours.

      1 — 2 — 3 — 4 — 5
      |                 |
      9 — 8 — 7 — 6 ——

  COMPLETE
  ────────
  Every agent sees every other agent's answer before judging.
  Maximum information sharing.
  Risk: echo chamber — agents may converge on shared wrong answer.
  Closest to Du et al. (2023) debate paradigm.
  Tests: does full visibility improve or degrade detection?

      1 ——— 2 ——— 3
      | \ / | \ / |
      |  X  |  X  |
      | / \ | / \ |
      4 ——— 5 ——— 6 ... (all connected)

Research question:
  "Under which network topology does a multi-agent LLM system
   demonstrate greatest resilience to adversarial hallucination
   injection — and why?"

  Hypothesis: Star topology provides superior adversarial
  resilience because malicious agent influence cannot
  propagate through the network before votes are cast.

Academic basis:
  Network science literature on information diffusion.
  Byzantine fault tolerance in distributed systems.
  Applied here empirically to LLM hallucination detection.
"""

from agent_registry import get_verifiers

# ── Topology constants ────────────────────────────────────────────────────────
STAR     = "star"
RING     = "ring"
COMPLETE = "complete"

VALID_TOPOLOGIES = [STAR, RING, COMPLETE]


# ── Topology builder ──────────────────────────────────────────────────────────

def build_topology(
    topology: str,
    agent_results: list,
) -> dict:
    """
    Builds the information-sharing graph for a given topology.
    Returns a dict mapping each agent to the list of other
    agents' answers it is allowed to see before judging.

    In STAR: no agent sees anyone else (empty list).
    In RING: each agent sees its two neighbours.
    In COMPLETE: each agent sees all other agents.

    Args:
        topology:      "star", "ring", or "complete"
        agent_results: List of agent result dicts (from verification_agents.py)
                       Each must have "agent" and "answer" keys.

    Returns:
        Dict mapping agent_id → list of {agent, answer} dicts
        that this agent is permitted to see.
    """
    if topology not in VALID_TOPOLOGIES:
        raise ValueError(
            f"Unknown topology '{topology}'. "
            f"Choose from: {VALID_TOPOLOGIES}"
        )

    # Filter to valid results only
    valid_results = [
        r for r in agent_results
        if r.get("answer") and not r.get("error")
    ]

    agent_ids = [r["agent"] for r in valid_results]
    n         = len(agent_ids)

    if topology == STAR:
        return _build_star(valid_results, agent_ids)

    elif topology == RING:
        return _build_ring(valid_results, agent_ids, n)

    elif topology == COMPLETE:
        return _build_complete(valid_results, agent_ids)


def _build_star(valid_results: list, agent_ids: list) -> dict:
    """
    Star topology — no agent sees any other agent.
    Each agent reports independently to the central aggregator.
    """
    return {
        agent_id: []   # empty — sees nobody
        for agent_id in agent_ids
    }


def _build_ring(valid_results: list, agent_ids: list, n: int) -> dict:
    """
    Ring topology — each agent sees its two immediate neighbours.
    Agents are arranged in a circular chain.
    """
    result_map = {r["agent"]: r for r in valid_results}
    visibility = {}

    for i, agent_id in enumerate(agent_ids):
        left_neighbour  = agent_ids[(i - 1) % n]
        right_neighbour = agent_ids[(i + 1) % n]

        visibility[agent_id] = [
            {
                "agent":  left_neighbour,
                "answer": result_map[left_neighbour]["answer"],
            },
            {
                "agent":  right_neighbour,
                "answer": result_map[right_neighbour]["answer"],
            },
        ]

    return visibility


def _build_complete(valid_results: list, agent_ids: list) -> dict:
    """
    Complete graph topology — every agent sees every other agent.
    Maximum information sharing before judging.
    """
    result_map = {r["agent"]: r for r in valid_results}
    visibility = {}

    for agent_id in agent_ids:
        visibility[agent_id] = [
            {
                "agent":  other_id,
                "answer": result_map[other_id]["answer"],
            }
            for other_id in agent_ids
            if other_id != agent_id  # does not see its own answer
        ]

    return visibility


# ── Context builder for judge prompts ─────────────────────────────────────────

def get_agent_context(
    agent_id:   str,
    visibility: dict,
) -> str:
    """
    Builds the context string that an agent sees before judging.
    This is injected into the judge prompt for that agent.

    In STAR: returns empty string (agent sees nothing extra).
    In RING/COMPLETE: returns the answers of visible neighbours.

    Args:
        agent_id:   The judging agent's ID
        visibility: Output from build_topology()

    Returns:
        Formatted string of visible answers, or empty string.
    """
    visible = visibility.get(agent_id, [])

    if not visible:
        return ""

    lines = ["\n\nOther agents you can see before judging:"]
    for entry in visible:
        lines.append(f"\n  [{entry['agent']}]: {entry['answer']}")

    return "\n".join(lines)


# ── Topology description for reports ─────────────────────────────────────────

def describe_topology(topology: str) -> str:
    """Returns a one-line description of the topology."""
    descriptions = {
        STAR:     "Star — all agents independent, report to central aggregator",
        RING:     "Ring — each agent sees two neighbours before judging",
        COMPLETE: "Complete — every agent sees all other agents before judging",
    }
    return descriptions.get(topology, "Unknown topology")


def print_topology(topology: str, visibility: dict):
    """Prints topology structure to terminal."""
    print(f"\n[Topology] {describe_topology(topology)}")
    print(f"  Mode: {topology.upper()}")
    print(f"  Agents: {len(visibility)}")

    if topology == STAR:
        print("  Each agent answers independently (no shared context)")
    elif topology == RING:
        print("  Visibility:")
        for agent_id, visible in visibility.items():
            neighbours = [v["agent"] for v in visible]
            print(f"    {agent_id} sees: {neighbours}")
    elif topology == COMPLETE:
        sample_agent = list(visibility.keys())[0]
        sees = [v["agent"] for v in visibility[sample_agent]]
        print(f"  Every agent sees all others: {sees[:3]}... "
              f"({len(sees)} total)")


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from verification_agents import run_all_verification_agents

    print("\n" + "=" * 60)
    print("TOPOLOGY MANAGER — TEST")
    print("=" * 60)

    # Get some agent results to work with
    question = "What is the capital of France?"
    print(f"\nQuestion: {question}")

    results = run_all_verification_agents(question, include_byzantine=False)
    valid   = [r for r in results if r.get("answer") and not r.get("error")]

    # Test all three topologies
    for topo in VALID_TOPOLOGIES:
        print(f"\n{'─' * 40}")
        visibility = build_topology(topo, valid)
        print_topology(topo, visibility)

        # Show context for first agent
        first_agent = list(visibility.keys())[0]
        context     = get_agent_context(first_agent, visibility)

        if context:
            print(f"\n  Context seen by {first_agent}:")
            print(f"  {context[:200]}...")
        else:
            print(f"\n  {first_agent} sees no other answers (star topology)")

    print("\n✅ Topology manager working correctly.")