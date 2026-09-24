# ⚙ Self-Learning SWE Agent  —  v3 Production Memory Engine

A production-grade, self-learning AI agent for **software engineering & workflow** with
genuine long-term memory — built on **NVIDIA NIM + Mem0 + Qdrant (Docker)**.

Memories persist across process restarts. Every turn shows you exactly what the agent
learned, updated, or discarded — in real time, as it happens.

---

## What makes it production-grade

| Feature | Detail |
|---------|--------|
| **Streaming output** | Reply tokens appear live via `Rich.Live` — no blocking wait |
| **Rolling session context** | Last 6 turns injected into every LLM call |
| **Startup welcome brief** | LLM summarises your stored profile on boot |
| **Qdrant health check** | Fails fast with a clear error if Docker is not running |
| **Tenacity retry + backoff** | Handles NVIDIA NIM rate limits transparently |
| **prompt_toolkit input** | Up-arrow history, Ctrl-A/E line editing, persistent `~/.swe_agent_history` |
| **JSONL session logging** | Every turn written to `~/.swe_agent_logs/<timestamp>.jsonl` |
| **Pydantic models throughout** | No raw dict leakage between modules |
| **Memory categories** | Facts tagged `[language]`, `[framework]`, `[toolchain]`, etc. |
| **Zero duplicate Qdrant calls** | Retrieval result shared between telemetry panel and LLM context |

---

## Stack

| Layer | Technology |
|-------|-----------|
| LLM + extraction | NVIDIA NIM (`meta/llama-3.1-8b-instruct` or `3.3-70b`) |
| Embeddings | NVIDIA NIM (`nvidia/nv-embedqa-e5-v5`, 1024-dim) |
| Memory framework | Mem0 OSS (`mem0ai`) |
| Vector store | Qdrant in Docker with persistent named volume |
| CLI input | `prompt_toolkit` with file-based history |
| Output | `Rich` — Live streaming, tables, panels |
| Retry | `tenacity` — exponential back-off on rate limits |
| Models | `pydantic` v2 — typed data contracts end-to-end |

---

## Setup

### Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.11+ |
| Docker Desktop | latest |
| NVIDIA API key | [integrate.api.nvidia.com](https://integrate.api.nvidia.com) |

### 1. Start Qdrant

```powershell
docker-compose up -d
# Verify:
curl http://localhost:6333/healthz
```

### 2. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 3. Configure

```powershell
copy .env.example .env
# Edit .env — NVIDIA_API_KEY is already filled in
```

### 4. Run

```powershell
python main.py
python main.py --user arnav    # isolated memory namespace
```

---

## What you see every turn

```
─────────────────────── Pipeline ────────────────────────────
┌─ QUERY ─────────────────────────────────────────────────┐
│  I switched from VS Code to Neovim with LSP.            │
└─────────────────────────────────────────────────────────┘

┌─ RETRIEVED CONTEXT ─────────────────────────────────────┐
│  1. [toolchain] Uses VS Code as primary editor  0.9312  │
│  2. [language]  Primary language is Python      0.7801  │
└─────────────────────────────────────────────────────────┘

─────────────────────── Agent (streaming) ────────────────
┌─ AGENT ──────────────────────────────────────────────── ┐
│  Neovim + LSP is a solid upgrade. Once your config      │   ← tokens stream live
│  stabilises you will notice a meaningful latency drop   │
│  for large files. Which LSP server are you using for... │
└─────────────────────────────────────────────────────────┘

─────────────────────── Memory Operations ────────────────
  [MEMORY OP: UPDATE]  [toolchain] Uses Neovim with LSP as primary editor
               was: [toolchain] Uses VS Code as primary editor
               id=7f3a1b2c-...
```

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `/memories` | Full memory table |
| `/search <query>` | Semantic search over your memories |
| `/forget <n>` | Delete memory by row number from `/memories` |
| `/export` | Dump all memories to `~/.swe_agent_logs/memories_<ts>.json` |
| `/stats` | Memory count by domain category |
| `/history` | This session conversation so far |
| `/clear` | Wipe ALL memories (confirmation required) |
| `/quit` | Exit (session log path shown) |

---

## Test scenarios

| # | Scenario | Expected result |
|---|----------|-----------------|
| 1 | State "I use Rust" in session 1 → restart → ask "what do I work with?" | Agent recalls Rust |
| 2 | Say "I don't like dark mode, I use light mode now" | `UPDATE` fires — no duplicate stored |
| 3 | Say "Hey thanks!" | `[MEMORY OP: NONE]  No durable facts` |
| 4 | Kill process → `docker-compose down` → `up -d` → restart agent | All memories intact |

---

## File map

```
self-learning-agent/
├── docker-compose.yml     Qdrant with persistent named volume
├── memory_manager.py      Mem0 wrapper — Pydantic models, typed interface
├── agent.py               Async engine — streaming, session history, retry
├── main.py                CLI — streaming Live panel, 8 commands, JSONL log
├── .env                   Your keys (pre-filled)
├── .env.example           Template
├── requirements.txt       8 packages
└── README.md
```

---

## Upgrading the LLM

In `.env`:

```
NVIDIA_LLM_MODEL=meta/llama-3.3-70b-instruct
```

Restart — no other changes needed.
