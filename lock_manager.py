import os
import json
import time
from datetime import datetime

LOCK_FILE = "db_write_lock.json"

def acquire(agent_name: str, tables: list) -> tuple:
    """
    Try to acquire the write lock for the given tables.
    Returns (success: bool, state: dict)
    """
    # Check if lock file exists and is held
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                state = json.load(f)
            if state.get("held"):
                locked_tables = set(state.get("locked_tables", []))
                conflict = locked_tables & set(tables)
                if conflict:
                    return False, state
        except Exception:
            pass  # corrupt lock file — treat as free

    # Acquire the lock
    import os as _os
    state = {
        "held": True,
        "held_by": agent_name,
        "locked_tables": tables,
        "acquired_at": datetime.utcnow().isoformat(),
        "pid": _os.getpid()
    }
    with open(LOCK_FILE, "w") as f:
        json.dump(state, f, indent=2)
    return True, state


def release():
    """Release the write lock by deleting the lock file."""
    if os.path.exists(LOCK_FILE):
        os.remove(LOCK_FILE)
    print(f"[LOCK MANAGER] Lock released")


def is_held(tables: list = None) -> tuple:
    """
    Check if the lock is held for any of the given tables.
    Returns (held: bool, state: dict)
    """
    if not os.path.exists(LOCK_FILE):
        return False, {}
    try:
        with open(LOCK_FILE) as f:
            state = json.load(f)
        if not state.get("held"):
            return False, state
        if tables:
            locked = set(state.get("locked_tables", []))
            if locked & set(tables):
                return True, state
            return False, state
        return True, state
    except Exception:
        return False, {}


def wait_for_release(tables: list, timeout: int = 30, poll_interval: float = 0.5) -> bool:
    """
    Wait until the lock is released or timeout expires.
    Returns True if lock was released, False if timed out.
    """
    start = time.time()
    while time.time() - start < timeout:
        held, state = is_held(tables)
        if not held:
            return True
        waited = time.time() - start
        print(f"[LOCK MANAGER] Tables {tables} locked by '{state.get('held_by', 'unknown')}' "
              f"— waiting... ({waited:.1f}s)")
        time.sleep(poll_interval)
    return False
