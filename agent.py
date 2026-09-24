"""
agent.py  —  v3  (10x Engine)
================================
Async conversational loop — fully upgraded.

What is new in v3
-----------------
* Rolling session history (last 6 turns) injected into every LLM call
  so the model has real conversational context, not just the current message.
* Streaming synthesis — yields AsyncIterator[str] of tokens; the CLI
  can render them live with Rich.Live as they arrive.
* Tenacity retry with exponential back-off on RateLimitError /
  APIConnectionError — survives transient NVIDIA NIM hiccups.
* Welcome brief — on first chat() call, generates a 2-sentence summary
  of everything the agent remembers about the user.
* Qdrant health check — fast pre-flight ping before any memory operation.
* Parallel startup — retrieval and history building run concurrently.

Pipeline per turn
-----------------
  Step 1  RETRIEVE   — semantic search over Qdrant  (async, parallel-safe)
  Step 2  SYNTHESIZE — streaming LLM call with session history + memory context
  Step 3  RECONCILE  — Mem0 state-machine diff to emit ADD/UPDATE/DELETE/NONE
"""

import asyncio
import logging
import os
from datetime import datetime
from typing import AsyncIterator, Optional

from dotenv import load_dotenv
from openai import AsyncOpenAI, APIConnectionError, RateLimitError
from pydantic import BaseModel, Field
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from memory_manager import MemoryEvent, MemoryManager, RetrievedContext, TurnTelemetry

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MAX_SESSION_TURNS = 6       # rolling conversation window sent to LLM
RETRIEVAL_TOP_K   = 6       # top-k semantic hits per query


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic model: one complete agent turn
# ─────────────────────────────────────────────────────────────────────────────

class AgentTurn(BaseModel):
    """Full output of one pipeline turn — telemetry + final reply bundled."""
    query:     str
    reply:     str
    telemetry: TurnTelemetry
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def non_none_events(self) -> list[MemoryEvent]:
        return [e for e in self.telemetry.events if e.op != "NONE"]


# ─────────────────────────────────────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a high-performance personal software engineering and workflow assistant \
with persistent long-term memory. You remember verified facts about this user \
across all sessions — they are injected below when relevant.

Rules:
1. Ground every response in the user facts listed below. \
   Never fabricate state that has not been explicitly stated.
2. Be precise, terse, and technically rigorous. \
   Treat the user as a senior engineer — no hand-holding, no filler.
3. Use recalled facts naturally — do not announce "I remember that...".
4. If the user contradicts a known fact, accept it and proceed; \
   the memory state machine handles reconciliation separately.
5. When no relevant memory exists, answer from first principles.
"""

_MEMORY_HEADER  = "\n\n## Verified user facts (ranked by relevance):\n"
_HISTORY_HEADER = "\n\n## This session so far:\n"

_WELCOME_PROMPT = """\
You have long-term memory about a software engineer. Here are the stored facts:

{facts}

Write a warm, terse 2-sentence greeting that naturally references \
what you know about them. Do not list the facts verbatim.
"""

_QDRANT_CHECK_TIMEOUT = 3.0   # seconds


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant health check  (sync, called once at startup)
# ─────────────────────────────────────────────────────────────────────────────

def check_qdrant(host: str, port: int) -> bool:
    """Ping the Qdrant REST endpoint. Returns True if reachable."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(host=host, port=port, timeout=_QDRANT_CHECK_TIMEOUT)
        client.get_collections()
        return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Tenacity retry decorator for all NVIDIA NIM API calls
# ─────────────────────────────────────────────────────────────────────────────

def _nim_retry(func):
    """Wrap an async function with 3-attempt exponential back-off."""
    return retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=15),
        retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )(func)


# ─────────────────────────────────────────────────────────────────────────────
# SWEAgent
# ─────────────────────────────────────────────────────────────────────────────

class SWEAgent:
    """
    Async SWE agent with streaming, rolling session history,
    and long-term Mem0 memory.
    """

    def __init__(self, user_id: str) -> None:
        self.user_id    = user_id
        self.model      = os.environ["NVIDIA_LLM_MODEL"]
        self.memory     = MemoryManager(user_id=user_id)
        self._client    = AsyncOpenAI(
            api_key  = os.environ["NVIDIA_API_KEY"],
            base_url = "https://integrate.api.nvidia.com/v1",
        )
        # Rolling session history: list of {"role": ..., "content": ...}
        self._session_history: list[dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    async def welcome_brief(self) -> str:
        """
        Generate a personalised greeting summarising stored memories.
        Returns empty string if no memories exist yet.
        """
        memories = self.memory.get_all()
        if not memories:
            return ""
        facts_block = "\n".join(f"- {m.fact}" for m in memories[:20])
        prompt = _WELCOME_PROMPT.format(facts=facts_block)

        @_nim_retry
        async def _call():
            r = await self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=120,
            )
            return r.choices[0].message.content.strip()

        return await _call()

    async def stream_chat(
        self, user_message: str
    ) -> tuple[AsyncIterator[str], "asyncio.Future[AgentTurn]", list[RetrievedContext]]:
        """
        Start the pipeline for one turn.

        Returns
        -------
        (token_stream, telemetry_future, retrieved)
            token_stream       — async iterator of string tokens to display
            telemetry_future   — Future[AgentTurn] resolved after stream ends
            retrieved          — context hits (so the caller can render them
                                 without issuing a second Qdrant query)
        """
        loop = asyncio.get_event_loop()
        telemetry_future: asyncio.Future[AgentTurn] = loop.create_future()

        # Step 1: RETRIEVE — single search, result shared with caller
        retrieved = self.memory.retrieve(user_message, top_k=RETRIEVAL_TOP_K)

        # Build messages for the LLM
        messages = self._build_messages(user_message, retrieved)

        # Step 2: SYNTHESIZE — streaming
        token_gen = self._synthesize_stream(
            user_message, messages, retrieved, telemetry_future
        )
        return token_gen, telemetry_future, retrieved

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_messages(
        self,
        user_message: str,
        retrieved:    list[RetrievedContext],
    ) -> list[dict]:
        """Build the LLM message list with memory + session history injected."""
        system = _SYSTEM_PROMPT

        if retrieved:
            facts = "\n".join(
                f"{i}. {c.fact}  [relevance={c.score}]"
                for i, c in enumerate(retrieved, 1)
            )
            system += _MEMORY_HEADER + facts

        messages: list[dict] = [{"role": "system", "content": system}]

        # Inject rolling session history
        if self._session_history:
            messages.extend(self._session_history)

        messages.append({"role": "user", "content": user_message})
        return messages

    async def _synthesize_stream(
        self,
        user_message:     str,
        messages:         list[dict],
        retrieved:        list[RetrievedContext],
        telemetry_future: "asyncio.Future[AgentTurn]",
    ) -> AsyncIterator[str]:
        """
        Inner async generator: stream tokens, then reconcile memory.
        Sets telemetry_future when done.
        """
        full_reply = ""

        @_nim_retry
        async def _open_stream():
            return await self._client.chat.completions.create(
                model       = self.model,
                messages    = messages,
                temperature = 0.6,
                max_tokens  = 1024,
                stream      = True,
            )

        stream = await _open_stream()

        async for chunk in stream:
            token = (chunk.choices[0].delta.content or "") if chunk.choices else ""
            if token:
                full_reply += token
                yield token

        # Step 3: RECONCILE STATE (after full reply is assembled)
        turn_msgs = [
            {"role": "user",      "content": user_message},
            {"role": "assistant", "content": full_reply},
        ]
        events = self.memory.reconcile(turn_msgs)

        # Update rolling session history
        self._update_history(user_message, full_reply)

        telemetry = TurnTelemetry(
            query=user_message,
            retrieved=retrieved,
            events=events,
        )
        turn = AgentTurn(
            query=user_message,
            reply=full_reply,
            telemetry=telemetry,
        )
        if not telemetry_future.done():
            telemetry_future.set_result(turn)

    def _update_history(self, user_msg: str, assistant_msg: str) -> None:
        """Add this turn to the session window; trim to MAX_SESSION_TURNS."""
        self._session_history.append({"role": "user",      "content": user_msg})
        self._session_history.append({"role": "assistant", "content": assistant_msg})
        # Keep only the most recent MAX_SESSION_TURNS * 2 messages
        cap = MAX_SESSION_TURNS * 2
        if len(self._session_history) > cap:
            self._session_history = self._session_history[-cap:]
