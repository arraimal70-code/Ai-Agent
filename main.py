"""
main.py  —  v3  (10x CLI)
===========================
Interactive CLI for the Self-Learning SWE Agent.

What is new in v3
-----------------
* Streaming output  — reply tokens appear as they arrive inside a Live panel
* prompt_toolkit    — up-arrow history, Ctrl-A/E line editing, persistent
                      ~/.swe_agent_history between runs
* Startup sequence  — Qdrant health check → memory count → welcome brief
* JSONL session log — every turn written to ~/.swe_agent_logs/<ts>.jsonl
* Power commands:
    /memories           list all stored facts in a table
    /search <query>     semantic search over memories
    /forget <n>         delete memory by table row number
    /export             dump all memories to JSON file
    /stats              show memory stats per domain category
    /history            show this session conversation so far
    /clear              wipe ALL memories (with confirmation)
    /quit               exit

Run
---
    python main.py
    python main.py --user arnav
"""

import argparse
import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Rich ──────────────────────────────────────────────────────────────────────
from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# ── prompt_toolkit ────────────────────────────────────────────────────────────
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory

# ── project ───────────────────────────────────────────────────────────────────
from agent import AgentTurn, SWEAgent, check_qdrant
from memory_manager import MemoryEvent

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.WARNING)

_THEME = Theme({
    "add":    "bold green",
    "update": "bold yellow",
    "delete": "bold red",
    "none":   "dim",
    "tag":    "cyan",
    "score":  "dim magenta",
})

console = Console(theme=_THEME)

# Session log directory
_LOG_DIR = Path.home() / ".swe_agent_logs"
_LOG_DIR.mkdir(exist_ok=True)
_SESSION_LOG = _LOG_DIR / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# prompt_toolkit session (persists history to disk)
_HISTORY_FILE = Path.home() / ".swe_agent_history"
_pt_session: PromptSession = PromptSession(
    history       = FileHistory(str(_HISTORY_FILE)),
    auto_suggest  = AutoSuggestFromHistory(),
    mouse_support = False,
)

# Memory op display config
_OP_STYLE = {
    "ADD":    ("add",    "✚ ADD"),
    "UPDATE": ("update", "⟳ UPDATE"),
    "DELETE": ("delete", "✕ DELETE"),
    "NONE":   ("none",   "• NONE"),
}


# ─────────────────────────────────────────────────────────────────────────────
# Banner
# ─────────────────────────────────────────────────────────────────────────────

_BANNER = """\
[bold cyan]
  ╔════════════════════════════════════════════════════════════════╗
  ║   ⚙  Self-Learning SWE Agent  —  v3  Production Memory Engine ║
  ║   NVIDIA NIM  ×  Mem0  ×  Qdrant  ×  Streaming  ×  History   ║
  ╚════════════════════════════════════════════════════════════════╝
[/bold cyan]
[dim]  Commands: /memories /search <q> /forget <n> /export /stats /history /clear /quit[/dim]
"""


# ─────────────────────────────────────────────────────────────────────────────
# JSONL session logger
# ─────────────────────────────────────────────────────────────────────────────

def _log_turn(turn: AgentTurn) -> None:
    """Append one turn to the JSONL session log (non-fatal on error)."""
    try:
        with open(_SESSION_LOG, "a", encoding="utf-8") as fh:
            fh.write(turn.model_dump_json() + "\n")
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry renderer
# ─────────────────────────────────────────────────────────────────────────────

def _render_query_panel(query: str) -> None:
    console.print(Panel(
        Text(query, style="white"),
        title="[bold]QUERY[/bold]",
        border_style="cyan",
        padding=(0, 1),
        expand=False,
    ))


def _render_context_panel(retrieved) -> None:
    if not retrieved:
        console.print(Panel(
            Text("No relevant memories found.", style="dim"),
            title="[bold]RETRIEVED CONTEXT[/bold]",
            border_style="dim",
        ))
        return

    tbl = Table(show_header=True, header_style="bold dim",
                border_style="dim", box=None, padding=(0, 1))
    tbl.add_column("#",     style="dim",   width=3)
    tbl.add_column("Fact",  style="white")
    tbl.add_column("Score", style="score", width=7, justify="right")
    for i, c in enumerate(retrieved, 1):
        tbl.add_row(str(i), c.fact, f"{c.score:.4f}")
    console.print(Panel(tbl,
        title="[bold]RETRIEVED CONTEXT[/bold]",
        border_style="blue",
        padding=(0, 0),
    ))


def _render_memory_events(events: list[MemoryEvent]) -> None:
    active = [e for e in events if e.op != "NONE"]
    if not active:
        console.print(Padding(
            Text("[MEMORY OP: NONE]  No durable facts — state unchanged.", style="dim"),
            (0, 2),
        ))
        return
    for ev in active:
        style, label = _OP_STYLE.get(ev.op, ("white", ev.op))
        line = Text()
        line.append(f"  [MEMORY OP: {ev.op}]", style=style)
        line.append("  ")
        line.append(ev.fact, style="white")
        if ev.prev_fact and ev.op in ("UPDATE", "DELETE"):
            line.append("\n             was: ", style="dim")
            line.append(ev.prev_fact, style="dim italic")
        line.append(f"\n             id={ev.memory_id}", style="dim")
        console.print(line)


# ─────────────────────────────────────────────────────────────────────────────
# Streaming reply renderer  (Rich Live)
# ─────────────────────────────────────────────────────────────────────────────

async def _stream_and_render(
    agent: SWEAgent,
    user_message: str,
) -> AgentTurn:
    """
    Run the full pipeline turn with live streaming display.
    Returns the completed AgentTurn (with telemetry) when done.
    """
    token_stream, telemetry_future, retrieved = await agent.stream_chat(user_message)

    console.print()
    console.print(Rule("[dim]Pipeline[/dim]", style="dim"))
    _render_query_panel(user_message)
    _render_context_panel(retrieved)
    console.print(Rule("[dim]Agent  (streaming)[/dim]", style="dim"))

    # ── Streaming display ─────────────────────────────────────────────────────
    accumulated = ""
    with Live(
        Panel("", title="[bold]AGENT[/bold]", border_style="green", padding=(0, 1)),
        console=console,
        auto_refresh=False,
        vertical_overflow="visible",
    ) as live:
        async for token in token_stream:
            accumulated += token
            live.update(
                Panel(
                    Markdown(accumulated),
                    title="[bold]AGENT[/bold]",
                    border_style="green",
                    padding=(0, 1),
                ),
                refresh=True,
            )

    # ── Memory events (shown after stream ends) ───────────────────────────────
    console.print(Rule("[dim]Memory Operations[/dim]", style="dim"))
    turn: AgentTurn = await telemetry_future
    _render_memory_events(turn.telemetry.events)

    return turn


# ─────────────────────────────────────────────────────────────────────────────
# Slash-command handlers
# ─────────────────────────────────────────────────────────────────────────────

def _cmd_memories(agent: SWEAgent) -> None:
    memories = agent.memory.get_all()
    if not memories:
        console.print(Panel(Text("Memory store is empty.", style="dim"), border_style="dim"))
        return
    tbl = Table(
        title=f"[bold cyan]Memory Store[/bold cyan]  [dim]({len(memories)} entries)[/dim]",
        show_lines=True, border_style="cyan",
    )
    tbl.add_column("#",    style="dim",   width=4)
    tbl.add_column("Fact", style="white")
    tbl.add_column("ID",   style="dim",   width=38)
    for i, m in enumerate(memories, 1):
        tbl.add_row(str(i), m.fact, m.memory_id)
    console.print(tbl)


def _cmd_search(agent: SWEAgent, query: str) -> None:
    if not query.strip():
        console.print("[yellow]Usage:[/yellow]  /search <your query>")
        return
    results = agent.memory.retrieve(query, top_k=8)
    if not results:
        console.print("[dim]No matching memories.[/dim]")
        return
    tbl = Table(title=f'[bold]Search: "{query}"[/bold]',
                show_lines=True, border_style="blue")
    tbl.add_column("#",     style="dim",   width=4)
    tbl.add_column("Fact",  style="white")
    tbl.add_column("Score", style="score", width=7, justify="right")
    for i, r in enumerate(results, 1):
        tbl.add_row(str(i), r.fact, f"{r.score:.4f}")
    console.print(tbl)


def _cmd_forget(agent: SWEAgent, arg: str) -> None:
    memories = agent.memory.get_all()
    if not memories:
        console.print("[dim]No memories to delete.[/dim]")
        return
    try:
        n = int(arg.strip())
        if n < 1 or n > len(memories):
            raise ValueError
    except ValueError:
        console.print(f"[yellow]Usage:[/yellow]  /forget <number 1–{len(memories)}>")
        return
    target = memories[n - 1]
    console.print(f"[red]Delete:[/red]  {target.fact}")
    confirm = console.input("  Type [bold]yes[/bold] to confirm: ").strip().lower()
    if confirm == "yes":
        agent.memory.delete_by_id(target.memory_id)
        console.print(f"[red]  Deleted memory #{n}.[/red]")
    else:
        console.print("[dim]  Cancelled.[/dim]")


def _cmd_export(agent: SWEAgent) -> None:
    memories = agent.memory.get_all()
    if not memories:
        console.print("[dim]Nothing to export.[/dim]")
        return
    out_path = _LOG_DIR / f"memories_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    data = [{"id": m.memory_id, "fact": m.fact} for m in memories]
    out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    console.print(f"[green]Exported {len(memories)} memories →[/green] {out_path}")


def _cmd_stats(agent: SWEAgent) -> None:
    memories = agent.memory.get_all()
    if not memories:
        console.print("[dim]No memories yet.[/dim]")
        return
    # Count categories by looking for [tag] prefix in fact text
    cats: dict[str, int] = {}
    untagged = 0
    for m in memories:
        if m.fact.startswith("[") and "]" in m.fact:
            cat = m.fact[1:m.fact.index("]")]
            cats[cat] = cats.get(cat, 0) + 1
        else:
            untagged += 1
    tbl = Table(title="[bold cyan]Memory Statistics[/bold cyan]",
                border_style="cyan", show_header=True)
    tbl.add_column("Category", style="tag")
    tbl.add_column("Count",    style="bold", justify="right")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        tbl.add_row(cat, str(cnt))
    if untagged:
        tbl.add_row("(untagged)", str(untagged))
    tbl.add_row("─" * 12, "─" * 5)
    tbl.add_row("[bold]TOTAL[/bold]", f"[bold]{len(memories)}[/bold]")
    console.print(tbl)


def _cmd_history(agent: SWEAgent) -> None:
    hist = agent._session_history
    if not hist:
        console.print("[dim]No conversation history this session.[/dim]")
        return
    tbl = Table(title="[bold]Session History[/bold]",
                show_lines=True, border_style="dim")
    tbl.add_column("Role",    style="bold", width=10)
    tbl.add_column("Message", style="white")
    for msg in hist:
        role  = msg["role"].capitalize()
        color = "green" if role == "User" else "blue"
        tbl.add_row(f"[{color}]{role}[/{color}]", msg["content"])
    console.print(tbl)


def _cmd_clear(agent: SWEAgent) -> None:
    console.print(Panel(
        Text(
            "⚠  This permanently deletes ALL memories for this user.\n"
            "   Type [bold]yes[/bold] to confirm, anything else to cancel.",
            style="red",
        ),
        border_style="red",
    ))
    answer = console.input("  > ").strip().lower()
    if answer == "yes":
        agent.memory.delete_all()
        console.print("[bold red]  All memories wiped.[/bold red]")
    else:
        console.print("[dim]  Cancelled.[/dim]")


def _handle_slash(cmd_str: str, agent: SWEAgent) -> bool:
    """Dispatch slash commands. Returns False to quit."""
    parts = cmd_str.strip().split(None, 1)
    cmd   = parts[0].lower()
    arg   = parts[1] if len(parts) > 1 else ""

    if cmd == "/memories":
        _cmd_memories(agent)
    elif cmd == "/search":
        _cmd_search(agent, arg)
    elif cmd == "/forget":
        _cmd_forget(agent, arg)
    elif cmd == "/export":
        _cmd_export(agent)
    elif cmd == "/stats":
        _cmd_stats(agent)
    elif cmd == "/history":
        _cmd_history(agent)
    elif cmd == "/clear":
        _cmd_clear(agent)
    elif cmd in ("/quit", "/exit", "/q"):
        console.print(f"\n[bold cyan]  Session log → {_SESSION_LOG}[/bold cyan]")
        console.print("[bold cyan]  Goodbye — memories persist in Qdrant.[/bold cyan]\n")
        return False
    else:
        console.print(
            f"[yellow]  Unknown command:[/yellow] {cmd}   "
            "[dim](try /memories /search /forget /export /stats /history /clear /quit)[/dim]"
        )
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Startup sequence
# ─────────────────────────────────────────────────────────────────────────────

async def _startup(agent: SWEAgent) -> None:
    """Health check → memory count → optional welcome brief."""

    # 1. Qdrant health check
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    reachable = check_qdrant(host, port)

    if not reachable:
        console.print(Panel(
            Text(
                f"Cannot reach Qdrant at {host}:{port}\n\n"
                "Make sure Docker is running and execute:\n"
                "  docker-compose up -d\n\n"
                "Then restart the agent.",
                style="bold red",
            ),
            title="[bold red]⛔  Qdrant Unreachable[/bold red]",
            border_style="red",
        ))
        raise SystemExit(1)

    count = agent.memory.memory_count()

    status_line = Text()
    status_line.append("  Qdrant ", style="dim")
    status_line.append("✔ connected", style="bold green")
    status_line.append(f"  |  {count} memories stored", style="dim")
    status_line.append(f"  |  log → {_SESSION_LOG.name}", style="dim")
    console.print(status_line)

    # 2. Welcome brief (only if memories exist)
    if count > 0:
        console.print("[dim]  Generating welcome brief…[/dim]", end="\r")
        brief = await agent.welcome_brief()
        if brief:
            console.print(Panel(
                Text(brief, style="italic white"),
                title="[bold]Welcome back[/bold]",
                border_style="cyan",
                padding=(0, 1),
                expand=False,
            ))
    else:
        console.print("[dim]  No prior memories — starting fresh.[/dim]")


# ─────────────────────────────────────────────────────────────────────────────
# Main chat loop
# ─────────────────────────────────────────────────────────────────────────────

async def chat_loop(user_id: str) -> None:
    console.print(_BANNER)
    agent = SWEAgent(user_id=user_id)
    await _startup(agent)
    console.print(Rule(f"[bold]Session  ·  user=[cyan]{user_id}[/cyan][/bold]"))

    running = True
    while running:
        # ── Input via prompt_toolkit ──────────────────────────────────────────
        try:
            user_input: str = await _pt_session.prompt_async(
                HTML("<ansigreen><b>You › </b></ansigreen>"),
            )
            user_input = user_input.strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[bold cyan]  Goodbye![/bold cyan]\n")
            break

        if not user_input:
            continue

        # ── Slash commands ────────────────────────────────────────────────────
        if user_input.startswith("/"):
            running = _handle_slash(user_input, agent)
            continue

        # ── Full pipeline turn ────────────────────────────────────────────────
        try:
            turn = await _stream_and_render(agent, user_input)
            _log_turn(turn)
        except KeyboardInterrupt:
            console.print("\n[dim]  Interrupted.[/dim]")
        except Exception as exc:
            console.print(f"\n[bold red]  Error:[/bold red] {exc}")
            import traceback; traceback.print_exc()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Self-Learning SWE Agent — v3 Production Engine"
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("AGENT_USER_ID", "default_user"),
        help="User ID to scope memories (default: AGENT_USER_ID from .env)",
    )
    args = parser.parse_args()
    asyncio.run(chat_loop(args.user))


if __name__ == "__main__":
    main()
