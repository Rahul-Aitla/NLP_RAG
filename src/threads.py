"""P1.7/P1.8 — server-owned conversation threads with bounded memory.

The server mints every thread id (secrets.token_urlsafe) and binds it to
the browser session id. Clients can neither forge ids nor reach another
session's threads. Per-thread turn/char budgets plus idle TTL bound the
in-process LangGraph MemorySaver growth; the generation call additionally
sees only recent messages (agent.recent_messages).
"""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field


@dataclass
class ThreadRec:
    session_id: str
    created: float
    last_active: float
    turns: int = 0
    chars: int = 0


class ThreadStore:
    def __init__(self, ttl_s: int = 7200):
        self.ttl_s = ttl_s
        self._threads: dict = {}

    def _now(self, now):
        return time.time() if now is None else now

    def create(self, session_id: str, now=None) -> str:
        tid = "th_" + secrets.token_urlsafe(18)
        t = self._now(now)
        self._threads[tid] = ThreadRec(session_id, t, t)
        return tid

    def _expired(self, rec: ThreadRec, now: float) -> bool:
        return (now - rec.last_active) > self.ttl_s

    def valid(self, session_id: str, thread_id: str, now=None) -> bool:
        """True only for a live thread owned by this session."""
        rec = self._threads.get(thread_id)
        if rec is None:
            return False
        t = self._now(now)
        if rec.session_id != session_id or self._expired(rec, t):
            return False
        return True

    def record_turn(self, thread_id: str, chars: int, max_turns: int,
                    max_chars: int, now=None) -> tuple:
        """Account one turn. Returns (ok, reason); over-budget -> (False,..)."""
        rec = self._threads.get(thread_id)
        if rec is None:
            return False, "unknown thread"
        t = self._now(now)
        if self._expired(rec, t):
            return False, "thread expired"
        if rec.turns + 1 > max_turns:
            return False, f"thread exceeded {max_turns} turns; start a new chat"
        if rec.chars + chars > max_chars:
            return False, "thread exceeded its character budget; start a new chat"
        rec.turns += 1
        rec.chars += chars
        rec.last_active = t
        return True, "ok"

    def prune(self, now=None) -> int:
        """Drop idle-expired threads. Returns number removed."""
        t = self._now(now)
        dead = [k for k, r in self._threads.items() if self._expired(r, t)]
        for k in dead:
            del self._threads[k]
        return len(dead)

    def __len__(self):
        return len(self._threads)
