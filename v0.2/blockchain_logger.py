"""
blockchain_logger.py
====================
A local Python blockchain implementation for tamper-evident
audit logging of all hallucination detection decisions.

Design rationale:
  All agents in this framework run on a single local machine.
  A distributed ledger (Ethereum, Hyperledger) solves a
  distributed trust problem — consensus across untrusted nodes.
  Our threat model is different: integrity of the audit record
  on a single deployment against post-hoc tampering.

  SHA-256 chain hashing provides identical tamper-evidence
  guarantees to a blockchain without distributed infrastructure
  overhead. Any modification to a historical block changes its
  hash, breaking the chain from that point forward.

  In a production multi-machine deployment, this local chain
  would be replaced with a distributed ledger. The cryptographic
  principle is identical — only the consensus mechanism differs.

Structure:
  Genesis Block (index 0)
      ↓ previous_hash
  Block 1 (first pipeline run)
      ↓ previous_hash
  Block 2 (second pipeline run)
      ↓ ...

Each block contains:
  - index, timestamp
  - all pipeline data (question, agent answers, scores, decision)
  - previous_hash (links to prior block)
  - hash (SHA-256 of all above — tamper-evident fingerprint)

Tampering detection:
  validate_chain() checks every block's hash matches its
  content AND its previous_hash matches the prior block.
  Any break in the chain = tampering detected.

Academic basis:
  Nakamoto (2008) Bitcoin whitepaper — blockchain as
  append-only tamper-evident ledger.
  Applied here to AI decision auditing rather than
  financial transactions.
"""

import hashlib
import json
import os
import datetime
from typing import Optional


# ── Block ─────────────────────────────────────────────────────────────────────

class Block:
    """
    A single block in the audit chain.
    Contains one complete pipeline run record.
    """

    def __init__(
        self,
        index:         int,
        data:          dict,
        previous_hash: str,
    ):
        self.index         = index
        self.timestamp     = datetime.datetime.utcnow().isoformat()
        self.data          = data
        self.previous_hash = previous_hash
        self.hash          = self._compute_hash()

    def _compute_hash(self) -> str:
        """
        Computes SHA-256 hash of this block's contents.
        Any change to index, timestamp, data, or previous_hash
        produces a completely different hash — tamper-evident.
        """
        block_string = json.dumps({
            "index":         self.index,
            "timestamp":     self.timestamp,
            "data":          self.data,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)

        return hashlib.sha256(block_string.encode()).hexdigest()

    def to_dict(self) -> dict:
        """Serialises block to dictionary for JSON storage."""
        return {
            "index":         self.index,
            "timestamp":     self.timestamp,
            "data":          self.data,
            "previous_hash": self.previous_hash,
            "hash":          self.hash,
        }


# ── Blockchain ────────────────────────────────────────────────────────────────

class Blockchain:
    """
    Local append-only blockchain for audit logging.
    Initialises with a genesis block on first use.
    Persists to JSON file on disk.
    """

    CHAIN_FILE = "audit_blockchain.json"

    def __init__(self, chain_file: Optional[str] = None):
        self.chain_file = chain_file or self.CHAIN_FILE
        self.chain: list[Block] = []

        # Load existing chain or create new one
        if os.path.exists(self.chain_file):
            self._load_chain()
            print(f"[Blockchain] Loaded existing chain "
                  f"({len(self.chain)} blocks) from {self.chain_file}")
        else:
            self._create_genesis_block()
            print(f"[Blockchain] New chain initialised "
                  f"with genesis block → {self.chain_file}")

    # ── Genesis block ─────────────────────────────────────────────────────────

    def _create_genesis_block(self):
        """
        Creates the first block in the chain (index 0).
        previous_hash is "0"*64 by convention — no prior block.
        """
        genesis = Block(
            index         = 0,
            data          = {
                "type":        "genesis",
                "system":      "Hallucination Detection Framework",
                "module":      "CSC8208 Newcastle University",
                "version":     "v0.2",
                "description": "Audit chain initialised",
            },
            previous_hash = "0" * 64,
        )
        self.chain.append(genesis)
        self._save_chain()

    # ── Add block ─────────────────────────────────────────────────────────────

    def add_block(self, pipeline_data: dict) -> Block:
        """
        Adds a new block containing one complete pipeline run.
        Links to previous block via previous_hash.

        Args:
            pipeline_data: Complete pipeline result dict containing
                           question, agent answers, layer scores,
                           decision, regeneration outcome, etc.

        Returns:
            The newly created block.
        """
        previous_block = self.chain[-1]

        new_block = Block(
            index         = len(self.chain),
            data          = pipeline_data,
            previous_hash = previous_block.hash,
        )

        self.chain.append(new_block)
        self._save_chain()

        print(f"[Blockchain] Block {new_block.index} added")
        print(f"  Hash: {new_block.hash[:24]}...")
        print(f"  Chain length: {len(self.chain)} blocks")

        return new_block

    # ── Validate chain ────────────────────────────────────────────────────────

    def validate_chain(self) -> dict:
        """
        Validates the entire chain for tamper-evidence.

        Checks:
          1. Each block's hash matches its recomputed hash
             (detects modification of block content)
          2. Each block's previous_hash matches the prior
             block's hash (detects insertion/deletion)

        Returns:
            Dict with valid (bool), message, and any broken blocks.
        """
        broken_blocks = []

        for i in range(1, len(self.chain)):
            current  = self.chain[i]
            previous = self.chain[i - 1]

            # Check 1: recompute hash directly from block fields
            # This avoids timestamp issues from creating a new Block object
            block_string = json.dumps({
                "index":         current.index,
                "timestamp":     current.timestamp,
                "data":          current.data,
                "previous_hash": current.previous_hash,
            }, sort_keys=True)
            recomputed_hash = hashlib.sha256(block_string.encode()).hexdigest()

            if current.hash != recomputed_hash:
                broken_blocks.append({
                    "block":  i,
                    "reason": "Block content modified — hash mismatch",
                })

            # Check 2: chain link still intact
            if current.previous_hash != previous.hash:
                broken_blocks.append({
                    "block":  i,
                    "reason": "Chain broken — previous_hash mismatch",
                })

        if broken_blocks:
            return {
                "valid":          False,
                "message":        f"⚠ TAMPERING DETECTED — {len(broken_blocks)} broken block(s)",
                "broken_blocks":  broken_blocks,
                "chain_length":   len(self.chain),
            }

        return {
            "valid":         True,
            "message":       f"✅ Chain valid — {len(self.chain)} blocks verified",
            "broken_blocks": [],
            "chain_length":  len(self.chain),
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save_chain(self):
        """Saves the full chain to JSON file."""
        with open(self.chain_file, "w") as f:
            json.dump(
                [block.to_dict() for block in self.chain],
                f,
                indent=2,
            )

    def _load_chain(self):
        """Loads chain from JSON file and reconstructs Block objects."""
        with open(self.chain_file, "r") as f:
            raw_chain = json.load(f)

        self.chain = []
        for raw_block in raw_chain:
            block               = Block.__new__(Block)
            block.index         = raw_block["index"]
            block.timestamp     = raw_block["timestamp"]
            block.data          = raw_block["data"]
            block.previous_hash = raw_block["previous_hash"]
            block.hash          = raw_block["hash"]
            self.chain.append(block)

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_chain(self) -> list:
        """Returns full chain as list of dicts."""
        return [block.to_dict() for block in self.chain]

    def get_block(self, index: int) -> Optional[dict]:
        """Returns a single block by index."""
        if index < 0 or index >= len(self.chain):
            return None
        return self.chain[index].to_dict()

    def get_latest_block(self) -> dict:
        """Returns the most recently added block."""
        return self.chain[-1].to_dict()

    def get_length(self) -> int:
        """Returns number of blocks in chain."""
        return len(self.chain)

    def print_chain_summary(self):
        """Prints a summary of the chain to terminal."""
        print("\n" + "=" * 60)
        print(f"  BLOCKCHAIN AUDIT LOG — {len(self.chain)} blocks")
        print("=" * 60)
        for block in self.chain:
            data_type = block.data.get("type", "pipeline_run")
            question  = block.data.get("question", "")[:40]
            decision  = block.data.get("decision", {}).get("action", "")
            print(
                f"  Block {block.index:>3}  "
                f"{block.timestamp[:19]}  "
                f"{data_type:<14}  "
                f"{decision:<12}  "
                f"{question}"
            )
        print("=" * 60)

        # Run validation
        result = self.validate_chain()
        print(f"\n  Validation: {result['message']}\n")


# ── Convenience function for pipeline use ─────────────────────────────────────

def build_audit_record(
    question:              str,
    primary_result:        dict,
    verification_results:  list,
    cosine_report:         dict,
    nli_report:            dict,
    judge_report:          dict,
    cross_validation:      dict,
    fusion_result:         dict,
    decision:              dict,
    trajectory:            dict,
    regeneration_result:   Optional[dict],
    explanation:           str,
    topology:              str = "star",
) -> dict:
    """
    Builds a complete, structured audit record for one pipeline run.
    This is what gets written to the blockchain as block data.

    Args:
        All pipeline stage outputs.

    Returns:
        Structured dict ready to pass to blockchain.add_block()
    """
    return {
        "type":      "pipeline_run",
        "question":  question,
        "topology":  topology,

        "primary": {
            "answer":    primary_result["answer"],
            "model":     primary_result["model"],
            "hash":      primary_result["hash"],
            "timestamp": primary_result["timestamp"],
        },

        "verifiers": [
            {
                "agent":   v["agent"],
                "model":   v["model"],
                "style":   v["style"],
                "answer":  v["answer"],
                "hash":    v["hash"],
            }
            for v in verification_results
            if not v.get("error")
        ],

        "layer_1_cosine": {
            "agreement_score": cosine_report.get("agreement_score"),
            "variance":        cosine_report.get("variance"),
            "risk_score":      cosine_report.get("risk_score"),
            "verdict":         cosine_report.get("verdict"),
        },

        "layer_2_nli": {
            "contradiction_count": nli_report.get("contradiction_count"),
            "entailment_count":    nli_report.get("entailment_count"),
            "risk_score":          nli_report.get("risk_score"),
            "verdict":             nli_report.get("verdict"),
        },

        "layer_3_judge": {
            "primary_hallucination_score": judge_report.get("primary_hallucination_score"),
            "primary_factual_accuracy":    judge_report.get("primary_factual_accuracy"),
            "primary_verdict":             judge_report.get("primary_verdict"),
            "risk_score":                  judge_report.get("risk_score"),
            "verdict":                     judge_report.get("verdict"),
        },

        "cross_validation": {
            "pattern":            cross_validation.get("pattern"),
            "confidence_level":   cross_validation.get("confidence_level"),
            "hallucination_type": cross_validation.get("hallucination_type"),
            "layer_verdicts":     cross_validation.get("layer_verdicts"),
        },

        "fusion": {
            "final_risk_score": fusion_result.get("final_risk_score"),
            "weights_used":     fusion_result.get("weights_used"),
        },

        "decision": {
            "action":      decision.get("action"),
            "risk_level":  decision.get("risk_level"),
            "label":       decision.get("label"),
        },

        "trajectory": trajectory,

        "regeneration": {
            "triggered":         regeneration_result is not None and
                                 regeneration_result.get("regeneration_triggered", False),
            "outcome":           regeneration_result.get("outcome") if regeneration_result else None,
            "improvement_delta": regeneration_result.get("improvement_delta") if regeneration_result else None,
            "final_answer":      regeneration_result.get("final_answer") if regeneration_result else None,
        },

        "explanation": explanation,
    }


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    # Use a test file so we don't pollute the real audit chain
    TEST_FILE = "test_blockchain.json"

    # Clean up any previous test
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)

    print("\n── Creating new blockchain ──")
    bc = Blockchain(chain_file=TEST_FILE)

    print("\n── Adding test blocks ──")

    # Simulate a pipeline run — ACCEPT
    bc.add_block({
        "type":     "pipeline_run",
        "question": "What is the boiling point of water?",
        "decision": {"action": "ACCEPT", "risk_level": "LOW"},
        "fusion":   {"final_risk_score": 0.042},
        "cross_validation": {
            "confidence_level":   "HIGH",
            "hallucination_type": "none",
        },
        "regeneration": {"triggered": False},
    })

    # Simulate a pipeline run — FLAG + REGENERATE
    bc.add_block({
        "type":     "pipeline_run",
        "question": "Who discovered penicillin and in what year?",
        "decision": {"action": "FLAG", "risk_level": "MODERATE"},
        "fusion":   {"final_risk_score": 0.31},
        "cross_validation": {
            "confidence_level":   "MODERATE",
            "hallucination_type": "factual contradiction",
        },
        "regeneration": {
            "triggered":         True,
            "outcome":           "IMPROVED",
            "improvement_delta": 0.24,
        },
    })

    # Print full chain
    bc.print_chain_summary()

    # Validate chain
    print("\n── Validation test ──")
    result = bc.validate_chain()
    print(f"  {result['message']}")

    # Simulate tampering
    print("\n── Tampering simulation ──")
    print("  Modifying block 2 data directly (FLAG → ACCEPT)...")
    bc.chain[2].data["decision"]["action"] = "ACCEPT"  # tamper! FLAG→ACCEPT hides detection

    result = bc.validate_chain()
    print(f"  {result['message']}")
    for broken in result["broken_blocks"]:
        print(f"  Block {broken['block']}: {broken['reason']}")

    # Clean up test file
    if os.path.exists(TEST_FILE):
        os.remove(TEST_FILE)
    print("\n── Test complete. Test file cleaned up. ──\n")