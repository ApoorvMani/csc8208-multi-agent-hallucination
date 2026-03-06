"""
logger.py - logs everything that happens during an experiment run.

two outputs get written to disk:
  1. event log  — a timestamped list of things that happened, in plain english,
                  in the order they happened. good for reading top to bottom.
  2. full record — the complete structured data for everything: all rounds,
                   all agents, all scores, all decisions, topology, config.
                   good for analysis and building graphs.

both are saved as a single JSON file with a timestamp in the filename
so you can run many experiments and keep them all separate.

file saved to: logs/ folder (created automatically if it doesnt exist)
"""

import json
import os
from datetime import datetime


# --- where logs get saved ---
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


class ExperimentLogger:
    """
    One instance per experiment run.
    Call log_event() as things happen, then save() at the end.
    """

    def __init__(self, experiment_id=None):
        # generate an experiment id from the current timestamp if none given
        self.start_time = datetime.now()
        self.experiment_id = experiment_id or self.start_time.strftime("%Y%m%d_%H%M%S")

        # --- event log ---
        # list of dicts, one per event, in chronological order
        # each event has: timestamp, event_type, message, and optional data
        self.events = []

        # --- full structured record ---
        # everything organised by topic — built up as experiment runs
        self.record = {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time.isoformat(),
            "end_time": None,
            "config": {},          # topology, node order, question, prompt variant
            "topology": {},        # full topology description (who sees who)
            "rounds": {},          # rounds[round_num] = {agents: {agent_id: {...}}}
            "convergence": {},     # from convergence_tracker
            "influence": {},       # from influence_tracker
            "deviation": {},       # from deviation_tracker
            "summary": {}          # final high level summary
        }

    # --- event logging ---

    def log_event(self, event_type, message, data=None):
        """
        Log one thing that happened.

        event_type — short category string e.g. "ROUND_START", "AGENT_DECISION", "WARNING"
        message    — plain english description of what happened and why
        data       — optional dict of any extra structured data for this event
        """

        event = {
            "timestamp": datetime.now().isoformat(),
            "event_type": event_type,
            "message": message
        }

        if data:
            event["data"] = data

        self.events.append(event)

        # also print to terminal so you can watch the experiment live
        ts = datetime.now().strftime("%H:%M:%S")
        print(f"  [{ts}] [{event_type}] {message}")

    # --- structured record builders ---
    # these write into the full record dict, called as experiment progresses

    def log_config(self, question, topology_name, agent_ids, node_order, prompt_variant="default"):
        """ record the experiment setup — what we ran and how """

        self.record["config"] = {
            "question": question,
            "topology": topology_name,
            "agent_ids": agent_ids,
            "node_order": node_order,          # order agents appear in prompts — for positional bias
            "prompt_variant": prompt_variant,
            "total_agents": len(agent_ids)
        }

        self.log_event(
            "EXPERIMENT_START",
            f"starting experiment | topology: {topology_name} | agents: {agent_ids}",
            {"question": question, "node_order": node_order}
        )

    def log_topology(self, topology_description, adjacency):
        """
        Record the full topology structure.
        topology_description — plain english from topology_manager.describe_topology()
        adjacency            — dict of {agent_id: [list of neighbours]}
        """

        self.record["topology"] = {
            "description": topology_description,
            "adjacency": adjacency       # who can see who
        }

        self.log_event(
            "TOPOLOGY_SET",
            f"topology recorded — adjacency: {adjacency}"
        )

    def log_round_start(self, round_num, total_rounds=5):
        """ mark the start of a round """

        if round_num not in self.record["rounds"]:
            self.record["rounds"][round_num] = {"agents": {}}

        self.log_event(
            "ROUND_START",
            f"round {round_num} of {total_rounds} starting"
        )

    def log_agent_response(self, round_num, agent_id, prompt, raw_response, parsed):
        """
        Record everything about one agents response in one round.

        prompt        — the full prompt this agent recieved
        raw_response  — the raw text the model returned
        parsed        — the structured dict from response_parser
        """

        if round_num not in self.record["rounds"]:
            self.record["rounds"][round_num] = {"agents": {}}

        # store the full agent record for this round
        self.record["rounds"][round_num]["agents"][agent_id] = {
            "prompt": prompt,
            "raw_response": raw_response,
            "evaluations": parsed.get("evaluations", {}),
            "step1_summary": parsed.get("step1_summary", ""),
            "decision": parsed.get("decision"),
            "reasoning": parsed.get("reasoning", ""),
            "revised_answer": parsed.get("revised_answer", ""),
            "parse_warnings": parsed.get("parse_warnings", []),
            "timestamp": datetime.now().isoformat()
        }

        # log a readable summary of what happened
        decision = parsed.get("decision", "UNKNOWN")
        warnings = parsed.get("parse_warnings", [])
        warning_note = f" | {len(warnings)} parse warnings" if warnings else ""

        self.log_event(
            "AGENT_DECISION",
            f"{agent_id} round {round_num}: {decision}{warning_note}",
            {
                "decision": decision,
                "reasoning_snippet": parsed.get("reasoning", "")[:120]  # first 120 chars
            }
        )

    def log_round1_answer(self, agent_id, answer, duration_s):
        """
        Special case for round 1 — agents just answer the question, no evaluation.
        Stored separately since there are no scores or decisions in round 1.
        """

        if 1 not in self.record["rounds"]:
            self.record["rounds"][1] = {"agents": {}}

        self.record["rounds"][1]["agents"][agent_id] = {
            "answer": answer,
            "duration_s": duration_s,
            "timestamp": datetime.now().isoformat()
        }

        self.log_event(
            "ROUND1_ANSWER",
            f"{agent_id} answered in {duration_s}s",
            {"answer_snippet": answer[:120]}
        )

    def log_convergence(self, convergence_summary):
        """ store the full convergence summary from convergence_tracker """

        self.record["convergence"] = convergence_summary

        all_conv = convergence_summary.get("all_converged", False)
        group_round = convergence_summary.get("group_convergence_round")

        if all_conv:
            self.log_event(
                "CONVERGENCE",
                f"all agents converged by round {group_round}"
            )
        else:
            # find who didnt converge
            never = [
                aid for aid, info in convergence_summary.get("per_agent", {}).items()
                if not info.get("converged")
            ]
            self.log_event(
                "CONVERGENCE",
                f"not all agents converged — still revising at end: {never}"
            )

    def log_influence(self, influence_summary):
        """ store the full influence summary from influence_tracker """

        self.record["influence"] = influence_summary

        paradox_count = influence_summary.get("paradox_count", 0)
        resistance_count = influence_summary.get("resistance_count", 0)

        self.log_event(
            "INFLUENCE_ANALYSIS",
            f"trustworthy-but-changed paradox cases: {paradox_count} | "
            f"resistance cases (doubted but held firm): {resistance_count}"
        )

    def log_deviation(self, deviation_summary):
        """ store the full deviation summary from deviation_tracker """

        self.record["deviation"] = deviation_summary

        consensus = deviation_summary.get("group_consensus_score")
        most_drifted = deviation_summary.get("most_drifted_agent")
        least_drifted = deviation_summary.get("least_drifted_agent")

        self.log_event(
            "DEVIATION_ANALYSIS",
            f"group consensus score: {consensus} | "
            f"most drifted: {most_drifted} | least drifted: {least_drifted}"
        )

    def log_summary(self, summary_dict):
        """ final high level summary of the whole experiment """

        self.record["summary"] = summary_dict
        self.record["end_time"] = datetime.now().isoformat()

        duration = (datetime.now() - self.start_time).total_seconds()

        self.log_event(
            "EXPERIMENT_END",
            f"experiment complete in {round(duration, 1)}s",
            summary_dict
        )

    def log_warning(self, message, data=None):
        """ log anything unexpected that happened but didnt crash the run """
        self.log_event("WARNING", message, data)

    def log_error(self, message, data=None):
        """ log a proper error — something went wrong """
        self.log_event("ERROR", message, data)

    # --- saving ---

    def save(self):
        """
        Writes everything to disk as a single JSON file.
        Filename includes timestamp, topology, and a snippet of the question
        so you can find the right log file without opening them.

        returns the path to the saved file.
        """

        # make sure logs folder exists
        os.makedirs(LOG_DIR, exist_ok=True)

        # build a short question snippet for the filename — first 30 chars, no spaces
        question = self.record.get("config", {}).get("question", "unknown")
        question_slug = question[:30].replace(" ", "_").replace("?", "").lower()
        topology = self.record.get("config", {}).get("topology", "unknown")

        filename = f"experiment_{self.experiment_id}_{topology}_{question_slug}.json"
        filepath = os.path.join(LOG_DIR, filename)

        # build the final output — event log + full record together
        output = {
            "event_log": self.events,
            "full_record": self.record
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n  [LOGGER] saved to: {filepath}")
        return filepath

    def get_event_log(self):
        """ returns the event log list — useful for inspecting without saving """
        return self.events

    def get_record(self):
        """ returns the full structured record dict """
        return self.record


# --- test it ---
if __name__ == "__main__":

    logger = ExperimentLogger()

    # simulate a small experiment
    logger.log_config(
        question="Who was the first person to walk on the moon?",
        topology_name="mesh",
        agent_ids=["agent_0", "agent_1", "agent_2"],
        node_order=["agent_0", "agent_1", "agent_2"]
    )

    logger.log_topology(
        topology_description="mesh: every agent sees every other agent",
        adjacency={"agent_0": ["agent_1", "agent_2"], "agent_1": ["agent_0", "agent_2"], "agent_2": ["agent_0", "agent_1"]}
    )

    logger.log_round_start(1)
    logger.log_round1_answer("agent_0", "Neil Armstrong was the first person to walk on the moon.", 3.2)
    logger.log_round1_answer("agent_1", "Buzz Aldrin walked on the moon first.", 2.8)

    logger.log_round_start(2)
    fake_parsed = {
        "decision": "REVISE",
        "reasoning": "agent_0 had the correct answer so I updated mine.",
        "revised_answer": "Neil Armstrong was the first person to walk on the moon in 1969.",
        "evaluations": {"agent_0": {"verdict": "TRUSTWORTHY", "hallucination_score": 1, "quality_score": 9, "example": ""}},
        "step1_summary": "agent_0 was most reliable.",
        "parse_warnings": []
    }
    logger.log_agent_response(2, "agent_1", "...", "...", fake_parsed)

    logger.log_warning("agent_2 response took longer than expected")

    logger.log_summary({"note": "test run complete"})

    path = logger.save()
    print(f"log saved to: {path}")
