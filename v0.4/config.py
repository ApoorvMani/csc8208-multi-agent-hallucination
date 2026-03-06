"""
config.py - all experiment settings in one place

CSC8208 Multi-Agent Hallucination Detection Framework — Newcastle University
Experiment 1: triangle topology, same model, 10 rounds, natural hallucination detection
"""

# the question all agents will answer
QUESTION = "Who invented the telephone and in what year?"

# how many rounds to run
TOTAL_ROUNDS = 100

# model used by all agents — same model so topology is the only variable
MODEL = "llama3.2"

# system prompt — same for all agents all rounds
SYSTEM_PROMPT = "You are a factual assistant. Answer questions accurately and concisely."

# agent configs — id, model, temperature
AGENTS = {
    "agent_0": {"model": MODEL, "temperature": 0.5},
    "agent_1": {"model": MODEL, "temperature": 0.5},
    "agent_2": {"model": MODEL, "temperature": 0.5},
}

# triangle topology — every agent sees both other agents
TOPOLOGY = {
    "agent_0": ["agent_1", "agent_2"],
    "agent_1": ["agent_0", "agent_2"],
    "agent_2": ["agent_0", "agent_1"],
}
