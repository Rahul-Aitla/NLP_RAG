"""P1.7/P1.8/P1.20 — server thread ownership, isolation, memory budgets."""
from src.threads import ThreadStore


def _store(**kw):
    kw.setdefault("ttl_s", 7200)
    return ThreadStore(**kw)


def test_create_valid_lifecycle():
    s = _store()
    tid = s.create("sess1", now=1000.0)
    assert tid.startswith("th_")
    assert s.valid("sess1", tid, now=1001.0) is True


def test_forged_thread_rejected():
    s = _store()
    assert s.valid("sess1", "th_" + "Z" * 24, now=1000.0) is False


def test_cross_session_access_rejected():
    s = _store()
    tid = s.create("sess1", now=1000.0)
    assert s.valid("sess2", tid, now=1001.0) is False


def test_expired_thread_rejected_and_pruned():
    s = _store(ttl_s=60)
    tid = s.create("sess1", now=1000.0)
    assert s.valid("sess1", tid, now=2000.0) is False
    assert s.prune(now=2000.0) == 1
    assert len(s) == 0


def test_new_session_gets_fresh_thread():
    s = _store()
    a = s.create("sess1", now=1000.0)
    b = s.create("sess1", now=1001.0)
    assert a != b and s.valid("sess1", b, now=1002.0)


def test_many_threads_isolated():
    s = _store()
    ids = [s.create(f"sess{i}", now=1000.0) for i in range(50)]
    assert len(set(ids)) == 50
    for i, tid in enumerate(ids):
        assert s.valid(f"sess{i}", tid, now=1001.0) is True
        assert s.valid("sess-other", tid, now=1001.0) is False


def test_thirty_turn_conversation_stays_in_budget():
    """P1.8/P1.20: 30 real turns fit; the 31st is refused, chars accounted."""
    s = _store()
    tid = s.create("sess1", now=1000.0)
    for n in range(30):
        ok, _ = s.record_turn(tid, 600, max_turns=30, max_chars=20000, now=1000.0 + n)
        assert ok is True, f"turn {n + 1} refused inside budget"
    ok, reason = s.record_turn(tid, 600, max_turns=30, max_chars=20000, now=2000.0)
    assert ok is False and "30" in reason
    rec = s._threads[tid]
    assert rec.turns == 30 and rec.chars == 30 * 600 <= 20000


def test_char_budget_enforced_before_turn_budget():
    s = _store()
    tid = s.create("sess1", now=1000.0)
    ok, reason = s.record_turn(tid, 20001, max_turns=30, max_chars=20000, now=1000.0)
    assert ok is False and "character budget" in reason
