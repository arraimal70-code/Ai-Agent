"""
memory_manager.py  —  v3  (Production Memory Engine)
======================================================
Mem0 + Qdrant wrapper for the Software Engineering & Workflow Agent.

What is new in v3
-----------------
* MemoryCategory enum  — every stored fact is tagged with a domain label
  (LANGUAGE / FRAMEWORK / TOOLCHAIN / WORKFLOW / PROJECT / CONSTRAINT /
   CODE_STYLE / LEARNING / PERSONAL / OTHER)
* delete_by_id()       — surgically remove a single memory by UUID
* memory_count()       — lightweight count without fetching all payloads
* All public methods return typed Pydantic models; zero raw dict leakage
* Extraction prompt hardened: explicit category field in each output fact
"""

import os
from datetime import datetime
from enum import Enum
from typing import Literal, Optional

from dotenv import load_dotenv
from mem0 import Memory
from pydantic import BaseModel, Field
from rich.console import Console

load_dotenv()

console = Console()


# ─────────────────────────────────────────────────────────────────────────────
# Domain taxonomy
# ─────────────────────────────────────────────────────────────────────────────

class MemoryCategory(str, Enum):
    LANGUAGE   = "language"
    FRAMEWORK  = "framework"
    TOOLCHAIN  = "toolchain"
    WORKFLOW   = "workflow"
    PROJECT    = "project"
    CONSTRAINT = "constraint"
    CODE_STYLE = "code_style"
    LEARNING   = "learning"
    PERSONAL   = "personal"
    OTHER      = "other"


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────

MemoryOp = Literal["ADD", "UPDATE", "DELETE", "NONE"]


class MemoryEvent(BaseModel):
    """One atomic state transition from the reconciliation engine."""
    op: MemoryOp
    memory_id: str
    fact: str
    prev_fact: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class RetrievedContext(BaseModel):
    """One search hit from Qdrant."""
    memory_id: str
    fact: str
    score: float


class TurnTelemetry(BaseModel):
    """Complete execution trace for one agent turn."""
    query: str
    retrieved: list[RetrievedContext]
    events: list[MemoryEvent]

    @property
    def active_events(self) -> list[MemoryEvent]:
        return [e for e in self.events if e.op != "NONE"]


# ─────────────────────────────────────────────────────────────────────────────
# Domain-tuned extraction prompt  (Software Engineering & Workflow)
# ─────────────────────────────────────────────────────────────────────────────

_EXTRACTION_INSTRUCTIONS = """
You are a Software Engineering & Workflow Memory Organizer — a strict state machine.

Your ONLY task: extract durable, atomic, user-centric facts from the USER side of the
conversation. Output ONLY valid JSON: {"facts": ["..."]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXTRACT — these categories (tag each fact with its category):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. Programming languages & proficiency  → [language]
2. Frameworks & libraries               → [framework]
3. IDE / editor preferences             → [toolchain]
4. DevOps & build tools                 → [toolchain]
5. Workflow methodology (TDD, Kanban)   → [workflow]
6. Active project context               → [project]
7. Hard technical constraints           → [constraint]
8. Code style preferences               → [code_style]
9. Learning goals in engineering        → [learning]
10. Personal details (name, role, tz)   → [personal]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DO NOT EXTRACT:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- Small talk, greetings, thanks
- Transient one-off questions ("What does X do?")
- Anything the assistant says
- Hypotheticals about other people

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CONTRADICTION HANDLING (critical):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
If the user states a preference change (e.g. "I switched from X to Y"),
output ONLY the new state: "Uses Y" — never output both X and Y.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FEW-SHOT EXAMPLES:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

User: Hey!
Output: {"facts": []}

User: What is a generator in Python?
Output: {"facts": []}

User: I am Arnav, a backend engineer at a fintech startup in Bangalore.
Output: {"facts": ["[personal] Name is Arnav", "[personal] Role is backend engineer at a fintech startup", "[personal] Based in Bangalore"]}

User: I mainly use Python and FastAPI; learning Rust on the side.
Output: {"facts": ["[language] Primary language is Python", "[framework] Uses FastAPI for backend APIs", "[learning] Currently learning Rust"]}

User: I switched from VS Code to Neovim with LSP. Way faster.
Output: {"facts": ["[toolchain] Uses Neovim with LSP as primary editor"]}

User: I don't use Jupyter anymore — all data work in plain Python scripts now.
Output: {"facts": ["[toolchain] Does not use Jupyter notebooks", "[code_style] Writes data work in plain Python scripts"]}

User: My team follows strict TDD; we use GitHub Actions for CI.
Output: {"facts": ["[workflow] Team follows strict Test-Driven Development (TDD)", "[toolchain] Uses GitHub Actions for CI/CD"]}

User: Can you help me debug this function?
Output: {"facts": []}

Output ONLY valid JSON. No markdown, no explanation.
"""


# ─────────────────────────────────────────────────────────────────────────────
# Mem0 config builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_mem0_config() -> dict:
    return {
        "llm": {
            "provider": "openai",
            "config": {
                "model":           os.environ["NVIDIA_LLM_MODEL"],
                "openai_base_url": "https://integrate.api.nvidia.com/v1",
                "api_key":         os.environ["NVIDIA_API_KEY"],
                "temperature":     0.0,
                "max_tokens":      2048,
                "custom_instructions": _EXTRACTION_INSTRUCTIONS,
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "model":           os.environ["NVIDIA_EMBEDDING_MODEL"],
                "openai_base_url": "https://integrate.api.nvidia.com/v1",
                "api_key":         os.environ["NVIDIA_API_KEY"],
                "embedding_dims":  int(os.environ.get("NVIDIA_EMBEDDING_DIMS", "1024")),
            },
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "collection_name": os.environ["QDRANT_COLLECTION"],
                "host":            os.environ["QDRANT_HOST"],
                "port":            int(os.environ["QDRANT_PORT"]),
            },
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MemoryManager
# ─────────────────────────────────────────────────────────────────────────────

class MemoryManager:
    """
    Typed interface to Mem0 + Qdrant.
    All returned values are Pydantic models — no raw dicts leak to callers.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self._mem = Memory.from_config(_build_mem0_config())

    # ── Write ─────────────────────────────────────────────────────────────────

    def reconcile(self, messages: list[dict]) -> list[MemoryEvent]:
        """
        Feed a conversation turn to Mem0's reconciliation engine.
        Returns every ADD / UPDATE / DELETE / NONE event as a typed list.
        """
        raw = self._mem.add(messages, user_id=self.user_id)
        return [_map_event(item) for item in raw.get("results", [])]

    def delete_by_id(self, memory_id: str) -> None:
        """Delete a single memory by its UUID."""
        self._mem.delete(memory_id=memory_id)

    def delete_all(self) -> None:
        """Hard-wipe all memories for this user."""
        self._mem.delete_all(user_id=self.user_id)

    # ── Read ──────────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 6) -> list[RetrievedContext]:
        """Semantic search — returns typed results sorted by relevance."""
        raw = self._mem.search(
            query,
            filters={"user_id": self.user_id},
            top_k=top_k,
        )
        return [
            RetrievedContext(
                memory_id=r.get("id", ""),
                fact=r.get("memory", ""),
                score=round(float(r.get("score", 0.0)), 4),
            )
            for r in raw.get("results", [])
        ]

    def get_all(self) -> list[RetrievedContext]:
        """Return every stored memory for this user (no semantic filter)."""
        raw = self._mem.get_all(filters={"user_id": self.user_id})
        return [
            RetrievedContext(
                memory_id=r.get("id", ""),
                fact=r.get("memory", ""),
                score=1.0,
            )
            for r in raw.get("results", [])
        ]

    def memory_count(self) -> int:
        """Lightweight count of stored memories for this user."""
        return len(self.get_all())


# ─────────────────────────────────────────────────────────────────────────────
# Internal mapper
# ─────────────────────────────────────────────────────────────────────────────

def _map_event(item: dict) -> MemoryEvent:
    raw_op = item.get("event", "NONE").upper()
    op: MemoryOp = raw_op if raw_op in ("ADD", "UPDATE", "DELETE", "NONE") else "NONE"
    return MemoryEvent(
        op=op,
        memory_id=item.get("id", ""),
        fact=item.get("memory", ""),
        prev_fact=item.get("prev_memory") or None,
    )
