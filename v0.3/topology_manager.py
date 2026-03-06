"""
topology_manager.py - Defines who can see whose answers during discussion.
Ring = see left and right neighbours only.
Star = everyone talks through one center agent.
Complete / Mesh = everyone sees everyone.
"""

import networkx as nx

def build_topology(agent_ids, topology="mesh"):
    """
    Build a graph where agents are nodes and edges mean 'can see each other'.
    Returns a networkx Graph object.
    """

    # create an empty graph — no nodes, no edges yet
    G = nx.Graph()

    # add all our agents as nodes
    G.add_nodes_from(agent_ids)

    # --- build edges based on topology type ---

    if topology == "ring":
        # ring: each agent connects to the next one, last connects back to first
        # like people sitting in a circle — you can only talk to the person
        # on your left and right
        for i in range(len(agent_ids)):
            next_i = (i + 1) % len(agent_ids)
            G.add_edge(agent_ids[i], agent_ids[next_i])

    elif topology == "star":
        # star: first agent is the center, connected to everyone else
        # like a teacher in a classroom — everyone talks to the teacher
        # but students dont talk to each other directly
        center = agent_ids[0]
        for i in range(1, len(agent_ids)):
            G.add_edge(center, agent_ids[i])

    elif topology in ("complete", "mesh"):
        # complete / mesh: every agent connected to every other agent
        # like a group chat — everyone sees everyone
        # mesh is just another name for complete, same thing
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                G.add_edge(agent_ids[i], agent_ids[j])

    # hand back the finished graph
    return G

def get_neighbours(G, agent_id):
    """
    Ask the graph: who can this agent see?
    Returns a list of neighbour agent IDs.
    """

    # G.neighbors() is a built-in networkx function
    # it looks at all edges touching this agent and returns the other end
    return sorted(list(G.neighbors(agent_id)))

def describe_topology(G, agent_ids, topology):
    """
    Returns a plain english summary of the topology — goes straight into the log.
    Tells you who can see who, useful for reading results later.
    """

    lines = []
    lines.append(f"topology: {topology}")
    lines.append(f"nodes: {agent_ids}")
    lines.append("connections:")
    for agent_id in agent_ids:
        neighbours = get_neighbours(G, agent_id)
        lines.append(f"  {agent_id} sees: {neighbours}")
    return "\n".join(lines)


# --- test it ---
if __name__ == "__main__":

    agents = ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"]

    for topo in ["ring", "star", "mesh"]:
        print(f"\n=== {topo.upper()} ===")
        G = build_topology(agents, topo)
        print(describe_topology(G, agents, topo))
