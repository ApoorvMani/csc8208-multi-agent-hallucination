"""
experiment_config.py - defines all the experiment setups we want to run.

each config is one experiment — a specific combination of:
  - question
  - topology (mesh / ring / star)
  - node order (which agent appears first in prompts)
  - prompt variant (in case we want to test different judge prompts later)

the idea is to run the same question across different configs and compare.
changing the node order with the same topology lets us test positional bias.
changing the topology with the same node order lets us test network effects.
"""


# --- experiment configs ---
# each entry is a dict describing one full experiment run
# add more here to expand the experiment set

EXPERIMENTS = [

    # --- topology experiments ---
    # same question, same node order, different topologies
    # this isolates the effect of network structure

    {
        "id": "exp_01_mesh",
        "description": "mesh topology — everyone sees everyone (baseline)",
        "question": "Who was the first person to walk on the moon?",
        "topology": "mesh",
        "node_order": ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        "prompt_variant": "default",
        "total_rounds": 5
    },
    {
        "id": "exp_02_ring",
        "description": "ring topology — agents only see left and right neighbours",
        "question": "Who was the first person to walk on the moon?",
        "topology": "ring",
        "node_order": ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        "prompt_variant": "default",
        "total_rounds": 5
    },
    {
        "id": "exp_03_star",
        "description": "star topology — all agents go through agent_0 as hub",
        "question": "Who was the first person to walk on the moon?",
        "topology": "star",
        "node_order": ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        "prompt_variant": "default",
        "total_rounds": 5
    },

    # --- positional bias experiments ---
    # same question, same topology (mesh), different node orders
    # this tests whether appearing first in the prompt gives more influence

    {
        "id": "exp_04_mesh_reordered",
        "description": "mesh — node order reversed to test positional bias",
        "question": "Who was the first person to walk on the moon?",
        "topology": "mesh",
        "node_order": ["agent_4", "agent_3", "agent_2", "agent_1", "agent_0"],
        "prompt_variant": "default",
        "total_rounds": 5
    },
    {
        "id": "exp_05_mesh_shuffled",
        "description": "mesh — node order shuffled (agent_2 first)",
        "question": "Who was the first person to walk on the moon?",
        "topology": "mesh",
        "node_order": ["agent_2", "agent_0", "agent_4", "agent_1", "agent_3"],
        "prompt_variant": "default",
        "total_rounds": 5
    },

    # --- different question experiments ---
    # same mesh topology, default order, different questions
    # tests whether findings generalise across topics

    {
        "id": "exp_06_climate",
        "description": "mesh — different topic (climate science)",
        "question": "What is the main cause of climate change?",
        "topology": "mesh",
        "node_order": ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        "prompt_variant": "default",
        "total_rounds": 5
    },
    {
        "id": "exp_07_geography",
        "description": "mesh — different topic (geography)",
        "question": "What is the longest river in the world?",
        "topology": "mesh",
        "node_order": ["agent_0", "agent_1", "agent_2", "agent_3", "agent_4"],
        "prompt_variant": "default",
        "total_rounds": 5
    },

]


def get_experiment(experiment_id):
    """ returns the config dict for a given experiment id """
    for exp in EXPERIMENTS:
        if exp["id"] == experiment_id:
            return exp
    raise KeyError(f"experiment '{experiment_id}' not found")

def get_all_experiments():
    """ returns the full list of experiment configs """
    return EXPERIMENTS

def get_experiment_ids():
    """ returns just the ids — useful for listing available experiments """
    return [exp["id"] for exp in EXPERIMENTS]

def describe_experiments():
    """ prints a summary of all experiments — for picking which ones to run """
    print("=== AVAILABLE EXPERIMENTS ===")
    for exp in EXPERIMENTS:
        print(f"  {exp['id']}: {exp['description']}")
        print(f"    question: {exp['question']}")
        print(f"    topology: {exp['topology']} | order: {exp['node_order']}")
        print()


# --- test it ---
if __name__ == "__main__":
    describe_experiments()
    print(f"total experiments defined: {len(EXPERIMENTS)}")
