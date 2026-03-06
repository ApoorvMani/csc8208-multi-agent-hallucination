"""
agent_registry.py - Set up 5 agents and their settings.
all agents use same model — we want to test topology not model differnces.
"""

AGENTS = {} # dictonary to keep list of agents

# same model and temp for all — position in network is the only variable
AGENTS["agent_0"] = {
    "id": "agent_0",
    "model": "llama3.2",
    "temperature": 0.5,
    "system_prompt": "You are a knowlegeable assistant. Answer questions accurately and concisely."
}
AGENTS["agent_1"] = {
    "id": "agent_1",
    "model": "llama3.2",
    "temperature": 0.5,
    "system_prompt": "You are a knowlegeable assistant. Answer questions accurately and concisely."
}
AGENTS["agent_2"] = {
    "id": "agent_2",
    "model": "llama3.2",
    "temperature": 0.5,
    "system_prompt": "You are a knowlegeable assistant. Answer questions accurately and concisely."
}
AGENTS["agent_3"] = {
    "id": "agent_3",
    "model": "llama3.2",
    "temperature": 0.5,
    "system_prompt": "You are a knowlegeable assistant. Answer questions accurately and concisely."
}
AGENTS["agent_4"] = {
    "id": "agent_4",
    "model": "llama3.2",
    "temperature": 0.5,
    "system_prompt": "You are a knowlegeable assistant. Answer questions accurately and concisely."
}

def get_agent(agent_id):
    """ returns asked agents full config dict."""
    if agent_id not in AGENTS:
        raise KeyError(f"Agent '{agent_id}' not found. Available {list(AGENTS.keys())}")

    return AGENTS[agent_id]

def get_agent_ids():
    """ return a sorted list of all agent ids"""
    return sorted(AGENTS.keys())

def get_all_agents():
    """ returns the full agents dict """
    return AGENTS

def describe_agent(agent_id):
    """ one line summary of an agent for logs """
    a = get_agent(agent_id)
    return f"{a['id']} | model: {a['model']} | temp: {a['temperature']}"
