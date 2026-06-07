import sqlite3
import os
import re
import time
import random
import threading
from datetime import datetime, timezone

LOCK_DB = "coordinator_locks.db"
HEARTBEAT_INTERVAL = 2
STALE_THRESHOLD = 10
MAX_WAIT = 30
INTENT_TTL_SECONDS = 300
CONFLICT_ESCALATION_THRESHOLD = 4   # DELETE/INSERT conflicts before escalation
CONFLICT_TIME_WINDOW_MINUTES   = 60  # lookback window in minutes
INTENT_CLEANUP_AGE_HOURS       = 24  # completed intents older than this are deleted

INCOMPATIBLE_OPERATIONS = {
    "DELETE": ["UPDATE"],        # DELETE does not block INSERT — INSERT waits for DELETE, not vice versa
    "UPDATE": ["DELETE"],
    "INSERT": ["DELETE"],
    "SELECT": []
}


def _get_conn():
    conn = sqlite3.connect(LOCK_DB, timeout=10)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS write_locks (
            target_table TEXT NOT NULL,
            row_key TEXT NOT NULL DEFAULT '',
            held_by TEXT,
            process_id INTEGER,
            acquired_at TEXT,
            last_heartbeat TEXT,
            queue_position INTEGER DEFAULT 0,
            status TEXT DEFAULT 'holding',
            PRIMARY KEY (target_table, row_key, process_id)
        )
    """)
    conn.commit()
    return conn


def _is_stale(last_heartbeat_str: str) -> bool:
    try:
        last = datetime.fromisoformat(last_heartbeat_str)
        if last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed > STALE_THRESHOLD
    except Exception:
        return True


def _is_my_turn(tables: list, row_key: str = "") -> bool:
    """Returns True if this process is first in queue for all requested tables at the given row granularity."""
    row_key = row_key.strip().lower() if row_key else ""
    pid = os.getpid()
    conn = _get_conn()
    try:
        for table in tables:
            first_waiter = conn.execute(
                "SELECT process_id FROM write_locks "
                "WHERE target_table=? AND row_key=? AND status='waiting' "
                "ORDER BY queue_position ASC LIMIT 1",
                (table, row_key)
            ).fetchone()
            if first_waiter and first_waiter[0] != pid:
                return False
        return True
    finally:
        conn.close()


def acquire(tables: list, held_by: str, row_key: str = "") -> bool:
    """
    Try to acquire locks on all tables at the specified row granularity.
    If row_key is provided, only operations targeting the same row are blocked.
    If row_key is empty, falls back to table-level locking.
    Tables are sorted alphabetically before acquisition — deadlock prevention.
    """
    tables = sorted(tables)
    row_key = row_key.strip().lower() if row_key else ""
    pid = os.getpid()
    now = datetime.now(timezone.utc).isoformat()
    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table in tables:
            holder = conn.execute(
                "SELECT process_id, last_heartbeat FROM write_locks "
                "WHERE target_table=? AND row_key=? AND status='holding'",
                (table, row_key)
            ).fetchone()

            if holder:
                if holder[0] == pid:
                    continue
                if _is_stale(holder[1]):
                    print(f"[LOCK] Stale lock on '{table}' from PID {holder[0]} — taking over")
                    conn.execute(
                        "DELETE FROM write_locks WHERE target_table=? AND row_key=?",
                        (table, row_key)
                    )
                else:
                    existing_waiter = conn.execute(
                        "SELECT process_id FROM write_locks "
                        "WHERE target_table=? AND row_key=? AND status='waiting' AND process_id=?",
                        (table, row_key, pid)
                    ).fetchone()
                    if not existing_waiter:
                        max_pos = conn.execute(
                            "SELECT COALESCE(MAX(queue_position), 0) FROM write_locks "
                            "WHERE target_table=? AND row_key=?", (table, row_key)
                        ).fetchone()[0]
                        conn.execute(
                            "INSERT OR IGNORE INTO write_locks "
                            "(target_table, row_key, held_by, process_id, acquired_at, "
                            "last_heartbeat, queue_position, status) "
                            "VALUES (?,?,?,?,?,?,?,'waiting')",
                            (table, row_key, held_by, pid, now, now, max_pos + 1)
                        )
                        print(f"[LOCK] PID {pid} registered as waiter at "
                              f"position {max_pos + 1} for '{table}' row_key='{row_key}'")
                    conn.execute("COMMIT")
                    return False

            first_waiter = conn.execute(
                "SELECT process_id FROM write_locks "
                "WHERE target_table=? AND row_key=? AND status='waiting' "
                "ORDER BY queue_position ASC LIMIT 1",
                (table, row_key)
            ).fetchone()

            if first_waiter and first_waiter[0] != pid:
                conn.execute("COMMIT")
                return False

            conn.execute(
                "DELETE FROM write_locks "
                "WHERE target_table=? AND row_key=? AND status='waiting' AND process_id=?",
                (table, row_key, pid)
            )
            conn.execute(
                "INSERT INTO write_locks "
                "(target_table, row_key, held_by, process_id, acquired_at, "
                "last_heartbeat, queue_position, status) "
                "VALUES (?,?,?,?,?,?,0,'holding')",
                (table, row_key, held_by, pid, now, now)
            )
            print(f"[LOCK] PID {pid} acquired lock on '{table}' row_key='{row_key}'")

        conn.execute("COMMIT")
        return True
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        return False
    finally:
        conn.close()


def release(tables: list, row_key: str = ""):
    """Release locks in reverse alphabetical order."""
    tables = list(reversed(sorted(tables)))
    row_key = row_key.strip().lower() if row_key else ""
    pid = os.getpid()
    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        for table in tables:
            conn.execute(
                "DELETE FROM write_locks "
                "WHERE target_table=? AND row_key=? AND process_id=?",
                (table, row_key, pid)
            )
            print(f"[LOCK] PID {pid} released lock on '{table}' row_key='{row_key}'")
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
    finally:
        conn.close()


def is_held(tables: list, row_key: str = "") -> tuple:
    """Returns (is_held: bool, state: dict) for the given tables at the given row granularity."""
    row_key = row_key.strip().lower() if row_key else ""
    pid = os.getpid()
    conn = _get_conn()
    try:
        held_tables = []
        state = {}
        for table in sorted(tables):
            row = conn.execute(
                "SELECT held_by, process_id, last_heartbeat FROM write_locks "
                "WHERE target_table=? AND row_key=? AND status='holding'",
                (table, row_key)
            ).fetchone()
            if row and row[1] != pid and not _is_stale(row[2]):
                held_tables.append(table)
                state = {
                    "held_by": row[0],
                    "pid": row[1],
                    "locked_tables": held_tables
                }
        return (bool(held_tables), state)
    finally:
        conn.close()


def get_queue_state(table: str) -> list:
    conn = _get_conn()
    try:
        rows = conn.execute(
            "SELECT process_id, held_by, status, queue_position, last_heartbeat, row_key "
            "FROM write_locks WHERE target_table=? "
            "ORDER BY status DESC, queue_position ASC",
            (table,)
        ).fetchall()
        return [
            {
                "process_id": r[0],
                "held_by": r[1],
                "status": r[2],
                "queue_position": r[3],
                "last_heartbeat": r[4],
                "row_key": r[5]
            }
            for r in rows
        ]
    finally:
        conn.close()


def wait_for_release(tables: list, timeout: int = MAX_WAIT, row_key: str = "") -> bool:
    start = time.time()
    wait = 0.1
    while time.time() - start < timeout:
        held, _ = is_held(tables, row_key=row_key)
        if not held and _is_my_turn(tables, row_key=row_key):
            return True
        jitter = wait * 0.2 * random.uniform(-1, 1)
        sleep_time = min(wait + jitter, 5.0)
        print(f"[LOCK] Waiting {sleep_time:.2f}s — lock held or not my turn yet...")
        time.sleep(sleep_time)
        wait = min(wait * 2, 5.0)
    return False


def _heartbeat_loop(tables: list, stop_event: threading.Event):
    pid = os.getpid()
    while not stop_event.is_set():
        now = datetime.now(timezone.utc).isoformat()
        conn = _get_conn()
        try:
            for table in tables:
                conn.execute(
                    "UPDATE write_locks SET last_heartbeat=? "
                    "WHERE target_table=? AND process_id=? AND status='holding'",
                    (now, table, pid)
                )
            conn.commit()
        except Exception:
            pass
        finally:
            conn.close()
        stop_event.wait(HEARTBEAT_INTERVAL)


def start_heartbeat(tables: list) -> threading.Event:
    stop_event = threading.Event()
    t = threading.Thread(
        target=_heartbeat_loop,
        args=(tables, stop_event),
        daemon=True
    )
    t.start()
    return stop_event


def stop_heartbeat(stop_event: threading.Event):
    stop_event.set()


# ── Intent Registry ───────────────────────────────────────────────────────────

def _ensure_intent_table(conn):
    """Create operation_intents table if it does not exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS operation_intents (
            intent_id TEXT PRIMARY KEY,
            conversation_id TEXT,
            process_id INTEGER,
            target_tables TEXT,
            operation TEXT,
            target_entity TEXT,
            target_column TEXT,
            goal TEXT,
            registered_at TEXT,
            finished_at TEXT,
            status TEXT DEFAULT 'active',
            waiting_for TEXT,
            escalated INTEGER DEFAULT 0
        )
    """)


def _normalize_entity(entity: str) -> str:
    entity = entity.replace(" ", "").lower()
    entity = re.sub(r'\w+_id', 'id', entity)
    return entity


def _is_intent_stale(registered_at_str: str) -> bool:
    try:
        registered_at = datetime.fromisoformat(registered_at_str)
        if registered_at.tzinfo is None:
            registered_at = registered_at.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - registered_at).total_seconds()
        return elapsed > INTENT_TTL_SECONDS
    except Exception:
        return True


def _extract_target_entity(constraints: list, goal: str) -> str:
    """
    Extract primary key condition from constraints or goal string.
    Skips literal 'None' constraints produced by the planner LLM.
    Tries multiple patterns in order:
      1. Explicit id patterns with equals: student_id=1, id=5
      2. Explicit id patterns with space: student_id 1
      3. Name patterns: name='Alice'
      4. Value patterns: city='Cairo'
    Returns normalized string with = separator, empty string if nothing found.
    """
    equals_patterns = [
        r'\w+_?id\s*=\s*\d+',
        r'\bname\s*=\s*["\']?[\w\s]+["\']?',
        r'\w+\s*=\s*["\'][\w\s]+["\']'
    ]
    space_patterns = [
        r'\w+_?id\s+\d+'
    ]

    # check constraints first — skip literal "None" strings
    for c in constraints:
        if c.strip().lower() == "none":
            continue
        for pattern in equals_patterns:
            match = re.search(pattern, c, re.IGNORECASE)
            if match:
                return match.group(0).replace(" ", "").lower()
        for pattern in space_patterns:
            match = re.search(pattern, c, re.IGNORECASE)
            if match:
                # normalize "student_id 1" to "student_id=1"
                parts = match.group(0).strip().split()
                return f"{parts[0].lower()}={parts[-1]}"

    # fall back to goal string
    for pattern in equals_patterns:
        match = re.search(pattern, goal, re.IGNORECASE)
        if match:
            return match.group(0).replace(" ", "").lower()
    for pattern in space_patterns:
        match = re.search(pattern, goal, re.IGNORECASE)
        if match:
            parts = match.group(0).strip().split()
            return f"{parts[0].lower()}={parts[-1]}"

    return ""


def _extract_target_column(constraints: list, goal: str,
                           operation: str, allowed_schema: str = "") -> str:
    if operation != "UPDATE":
        return ""

    schema_columns = []
    if allowed_schema:
        table_blocks = re.findall(
            r'CREATE TABLE[^(]+\(([^;]+)\)',
            allowed_schema,
            re.IGNORECASE | re.DOTALL
        )
        for block in table_blocks:
            for line in block.split('\n'):
                line = line.strip().strip(',')
                if not line:
                    continue
                first_word = line.split()[0].lower() if line.split() else ""
                if first_word and first_word not in (
                    'primary', 'foreign', 'unique', 'check',
                    'constraint', 'index', 'key', '--'
                ):
                    schema_columns.append(first_word)

    if schema_columns:
        search_text = " ".join(constraints) + " " + goal
        search_text_lower = search_text.lower()
        for col in schema_columns:
            if col in search_text_lower and col not in ('id', 'the', 'a', 'an'):
                return col
        return ""

    col_pattern = r'\b(?:update|set|change|modify)\s+(?:the\s+)?(\w+)\b'
    match = re.search(col_pattern, goal, re.IGNORECASE)
    if match:
        return match.group(1).lower()
    return ""


def extract_row_key(sql: str) -> str:
    """
    Extract primary key condition from WHERE clause for row-level locking.
    Returns normalized condition string like 'student_id=3'.
    Returns empty string if no specific row condition found — falls back to table-level lock.
    """
    where_match = re.search(
        r'\bWHERE\b(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|;|$)',
        sql, re.IGNORECASE | re.DOTALL
    )
    if not where_match:
        return ""

    where_clause = where_match.group(1).strip()

    id_match = re.search(r'\w+_?id\s*=\s*\d+', where_clause, re.IGNORECASE)
    if id_match:
        return id_match.group(0).replace(" ", "").lower()

    normalized = where_clause.strip().replace(" ", "").lower()
    return normalized[:100]


def derive_conflict_constraint(db_path: str, target_entity: str,
                               target_tables: list, operation: str = "") -> str:
    import re as _re
    from database import run_sql

    entity_match = _re.match(r'(\w+)=(\d+)', target_entity)
    if not entity_match or not target_tables:
        return (
            f"CONFLICT RESOLUTION: a concurrent agent recently operated "
            f"on {target_entity}. Re-plan based on current database state."
        )

    entity_col = entity_match.group(1)
    entity_val = entity_match.group(2)
    table = target_tables[0]

    try:
        existence_check = run_sql(
            db_path,
            f'SELECT COUNT(*) FROM "{table}" WHERE "{entity_col}"={entity_val};'
        )
        exists = "0" not in existence_check

        if not exists:
            if operation == "INSERT":
                return (
                    f"CONFLICT RESOLUTION: {target_entity} was deleted by a "
                    f"concurrent agent session. You are performing an INSERT — "
                    f"the row no longer exists so your INSERT can proceed normally. "
                    f"Generate the INSERT SQL as planned without any modification."
                )
            else:
                return (
                    f"CONFLICT RESOLUTION: {target_entity} no longer exists "
                    f"in table '{table}' — it was deleted by a concurrent agent "
                    f"session. You MUST NOT generate SQL targeting this entity. "
                    f"If the question cannot be answered without this entity, "
                    f"respond with OPERATION_NOT_PERMITTED."
                )
        else:
            current_state = run_sql(
                db_path,
                f'SELECT * FROM "{table}" WHERE "{entity_col}"={entity_val};'
            )
            return (
                f"CONFLICT RESOLUTION: {target_entity} was modified by a "
                f"concurrent agent session. Current state as of re-planning: "
                f"{current_state}. Your SQL must account for this current "
                f"state — do not assume previous values."
            )
    except Exception as e:
        return (
            f"CONFLICT RESOLUTION: a concurrent agent recently operated on "
            f"{target_entity}. Re-plan based on current database state. "
            f"(State check failed: {e})"
        )


def register_intent(intent_id: str, conversation_id: str,
                    target_tables: list, operation: str,
                    target_entity: str, target_column: str,
                    goal: str, allowed_schema: str = ""):
    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_intent_table(conn)
        conn.execute(
            "INSERT OR REPLACE INTO operation_intents "
            "(intent_id, conversation_id, process_id, target_tables, operation, "
            "target_entity, target_column, goal, registered_at, status, waiting_for) "
            "VALUES (?,?,?,?,?,?,?,?,?,'active',NULL)",
            (
                intent_id,
                conversation_id,
                os.getpid(),
                ",".join(target_tables),
                operation,
                target_entity,
                target_column,
                goal,
                datetime.now(timezone.utc).isoformat()
            )
        )
        conn.execute("COMMIT")
        print(f"[INTENT] Registered: {operation} on {target_tables} "
              f"entity='{target_entity}' column='{target_column}' "
              f"goal='{goal[:60]}'")
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        print(f"[INTENT] Register error: {e}")
    finally:
        conn.close()


def check_intent_conflict(conversation_id: str, target_tables: list,
                          operation: str, target_entity: str,
                          target_column: str,
                          allowed_schema: str = "") -> tuple:
    if operation == "SELECT" or not target_entity:
        return (False, None)

    incompatible = INCOMPATIBLE_OPERATIONS.get(operation, [])
    if not incompatible:
        return (False, None)

    normalized_entity = _normalize_entity(target_entity)

    conn = _get_conn()
    try:
        conn.execute("BEGIN IMMEDIATE")
        _ensure_intent_table(conn)
        rows = conn.execute(
            "SELECT intent_id, conversation_id, process_id, target_tables, "
            "operation, target_entity, target_column, goal, registered_at "
            "FROM operation_intents "
            "WHERE status='active' AND conversation_id != ?",
            (conversation_id,)
        ).fetchall()
        conn.execute("COMMIT")

        for row in rows:
            if _is_intent_stale(row[8]):
                continue

            existing_tables = [t.strip() for t in row[3].split(",")]
            existing_operation = row[4]
            existing_entity = row[5]
            existing_column = row[6] or ""

            if not any(t in existing_tables for t in target_tables):
                continue

            if _normalize_entity(existing_entity) != normalized_entity:
                continue

            if existing_operation not in incompatible:
                continue

            if operation == "UPDATE" and existing_operation == "UPDATE":
                if (existing_column and target_column
                        and existing_column != target_column):
                    continue

            return (True, {
                "intent_id": row[0],
                "conversation_id": row[1],
                "process_id": row[2],
                "operation": existing_operation,
                "target_entity": existing_entity,
                "goal": row[7]
            })

        return (False, None)
    except Exception as e:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        print(f"[INTENT] Conflict check error: {e}")
        return (False, None)
    finally:
        conn.close()


def wait_for_intent_clear(conflicting_conversation_id: str,
                          timeout: int = 60) -> bool:
    start = time.time()
    wait = 0.1
    while time.time() - start < timeout:
        conn = _get_conn()
        try:
            _ensure_intent_table(conn)
            row = conn.execute(
                "SELECT status, registered_at FROM operation_intents "
                "WHERE conversation_id=?",
                (conflicting_conversation_id,)
            ).fetchone()
            if not row:
                return True
            if row[0] == 'completed':
                return True
            if _is_intent_stale(row[1]):
                print(f"[INTENT] Conflicting intent declared stale — proceeding")
                return True
        finally:
            conn.close()

        jitter = wait * 0.2 * random.uniform(-1, 1)
        sleep_time = min(wait + jitter, 5.0)
        print(f"[INTENT] Waiting {sleep_time:.2f}s for conflicting intent "
              f"to complete...")
        time.sleep(sleep_time)
        wait = min(wait * 2, 5.0)
    return False


def clear_intent(conversation_id: str):
    conn = _get_conn()
    try:
        _ensure_intent_table(conn)
        conn.execute(
            "UPDATE operation_intents SET status='completed', finished_at=? "
            "WHERE conversation_id=?",
            (datetime.now(timezone.utc).isoformat(), conversation_id)
        )
        conn.commit()
        print(f"[INTENT] Cleared intent for conversation {conversation_id[:8]}")
    except Exception as e:
        print(f"[INTENT] Clear error: {e}")
    finally:
        conn.close()


def cleanup_old_intents():
    """
    Delete completed intent rows older than INTENT_CLEANUP_AGE_HOURS.
    Keeps the operation_intents table from growing indefinitely.
    Safe to call at the start of each session.
    """
    from datetime import timedelta
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=INTENT_CLEANUP_AGE_HOURS)).isoformat()
    conn = _get_conn()
    try:
        _ensure_intent_table(conn)
        result = conn.execute(
            "DELETE FROM operation_intents "
            "WHERE status='completed' AND finished_at < ?",
            (cutoff,)
        )
        conn.commit()
        deleted = result.rowcount
        if deleted > 0:
            print(f"[INTENT] Cleaned up {deleted} old completed intent(s)")
    except Exception as e:
        print(f"[INTENT] Cleanup error: {e}")
    finally:
        conn.close()


def write_escalation_flag(target_entity: str, target_tables: list,
                          flagged_by_conversation_id: str):
    """
    Write an escalation flag for a target entity.
    Called when repeated INSERT/DELETE loop pattern is detected.
    Inserts a sentinel row so the flag persists even if no other agent
    has an active intent at the moment this is called.
    """
    conn = _get_conn()
    try:
        _ensure_intent_table(conn)
        now = datetime.now(timezone.utc).isoformat()
        # Flag any still-active conflicting intents
        conn.execute(
            "UPDATE operation_intents SET escalated=1 "
            "WHERE target_entity=? AND status='active' "
            "AND conversation_id != ?",
            (target_entity, flagged_by_conversation_id)
        )
        # Insert a durable sentinel row — survives regardless of other agents' lifecycle
        conn.execute(
            "INSERT OR REPLACE INTO operation_intents "
            "(intent_id, conversation_id, process_id, target_tables, operation, "
            "target_entity, goal, registered_at, finished_at, status, escalated) "
            "VALUES (?,?,?,?,?,?,?,?,?,'completed',1)",
            (
                f"escalation_{flagged_by_conversation_id}",
                flagged_by_conversation_id,
                os.getpid(),
                ",".join(target_tables),
                "INSERT",
                target_entity,
                "ESCALATION_SENTINEL",
                now,
                now
            )
        )
        conn.commit()
        print(f"[INTENT] Escalation flag written for entity '{target_entity}'")
    except Exception as e:
        print(f"[INTENT] Escalation flag error: {e}")
    finally:
        conn.close()


def check_escalation_flag(target_entity: str, conversation_id: str) -> bool:
    """
    Check if this entity has been escalated by another agent session.
    Returns True if escalation flag exists for this entity from another session.
    """
    def normalize(e):
        e = e.replace(" ", "").lower()
        e = re.sub(r'\w+_id', 'id', e)
        return e

    normalized = normalize(target_entity)
    conn = _get_conn()
    try:
        _ensure_intent_table(conn)
        rows = conn.execute(
            "SELECT target_entity FROM operation_intents "
            "WHERE escalated=1 AND conversation_id != ? "
            "AND status IN ('active', 'completed')",
            (conversation_id,)
        ).fetchall()
        for row in rows:
            if normalize(row[0]) == normalized:
                return True
        return False
    except Exception as e:
        print(f"[INTENT] Escalation check error: {e}")
        return False
    finally:
        conn.close()


def log_insert_delete_conflict(target_entity: str, conversation_id: str):
    """
    Record that this INSERT session detected an active DELETE conflict on this entity.
    Called once per session (deduplicated by conversation_id) so replans don't double-count.
    """
    conn = _get_conn()
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conflict_log (
                log_id    INTEGER PRIMARY KEY AUTOINCREMENT,
                target_entity TEXT,
                conversation_id TEXT,
                logged_at TEXT
            )
        """)
        already = conn.execute(
            "SELECT COUNT(*) FROM conflict_log WHERE conversation_id=? AND target_entity=?",
            (conversation_id, target_entity)
        ).fetchone()[0]
        if already == 0:
            conn.execute(
                "INSERT INTO conflict_log (target_entity, conversation_id, logged_at) "
                "VALUES (?,?,?)",
                (target_entity, conversation_id, datetime.now(timezone.utc).isoformat())
            )
            conn.commit()
            print(f"[INTENT] INSERT/DELETE conflict logged for entity '{target_entity}'")
    except Exception as e:
        print(f"[INTENT] Conflict log error: {e}")
    finally:
        conn.close()


def count_recent_insert_delete_conflicts(
    target_entity: str,
    db_path: str,
    minutes: int = CONFLICT_TIME_WINDOW_MINUTES,
    current_conversation_id: str = None
) -> int:
    """
    Count active+completed DELETE intents on this entity in the time window.
    Including active intents means round N's concurrent Agent 1 DELETE is counted
    even when Agent 1 hasn't finished yet — giving count=2 completed (prev rounds)
    + 1 active (current round) = 3 by round 4 without needing Agent 1 to complete first.
    current_conversation_id is accepted for API compatibility but unused (DELETE intents
    always belong to Agent 1, never to Agent 4's INSERT session).
    """
    _ = db_path
    _ = current_conversation_id
    def normalize(e):
        e = e.replace(" ", "").lower()
        e = re.sub(r'\w+_id', 'id', e)
        return e

    normalized = normalize(target_entity)

    from datetime import timedelta
    since = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()

    conn = _get_conn()
    try:
        _ensure_intent_table(conn)
        rows = conn.execute(
            "SELECT target_entity, operation FROM operation_intents "
            "WHERE status IN ('active', 'completed') AND registered_at > ?",
            (since,)
        ).fetchall()
        count = sum(
            1 for row in rows
            if normalize(row[0]) == normalized and row[1] == "DELETE"
        )
        return count
    except Exception as e:
        print(f"[INTENT] Conflict count error: {e}")
        return 0
    finally:
        conn.close()
