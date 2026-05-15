"""
access_control.py — Database-driven Role-Based Access Control

Manages permissions for the multi-agent database system.
Replaces the hardcoded ROLE_PERMISSIONS dict in query_agent.py.

Database: access_control.db (SQLite)
Tables:
    roles                   — role definitions
    users                   — user accounts with role assignment
    role_table_permissions  — table-level access per role per database
    role_column_permissions — column-level access per role per table
    role_row_filters        — row-level filter patterns per role per table
"""

import sqlite3
import os
from datetime import datetime

ACCESS_DB_PATH = "access_control.db"


# ─── Schema Initialization ────────────────────────────────────────────────────

def init_access_control_db():
    """
    Creates the access control database and tables if they do not exist.
    Also seeds default roles and permissions matching the original
    hardcoded ROLE_PERMISSIONS for backward compatibility.
    """
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS roles (
            role_id   INTEGER PRIMARY KEY AUTOINCREMENT,
            role_name TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY,
            username   TEXT UNIQUE NOT NULL,
            role_id    INTEGER NOT NULL,
            is_active  INTEGER DEFAULT 1,
            created_at TEXT DEFAULT (datetime('now')),
            FOREIGN KEY (role_id) REFERENCES roles(role_id)
        );

        CREATE TABLE IF NOT EXISTS role_table_permissions (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            role_id    INTEGER NOT NULL,
            db_name    TEXT NOT NULL,
            table_name TEXT NOT NULL,
            can_select INTEGER DEFAULT 1,
            can_insert INTEGER DEFAULT 0,
            can_update INTEGER DEFAULT 0,
            can_delete INTEGER DEFAULT 0,
            FOREIGN KEY (role_id) REFERENCES roles(role_id)
        );

        CREATE TABLE IF NOT EXISTS role_column_permissions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            role_id     INTEGER NOT NULL,
            db_name     TEXT NOT NULL,
            table_name  TEXT NOT NULL,
            column_name TEXT NOT NULL,
            FOREIGN KEY (role_id) REFERENCES roles(role_id)
        );

        CREATE TABLE IF NOT EXISTS role_row_filters (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            role_id          INTEGER NOT NULL,
            db_name          TEXT NOT NULL,
            table_name       TEXT NOT NULL,
            filter_condition TEXT NOT NULL,
            FOREIGN KEY (role_id) REFERENCES roles(role_id)
        );
    """)

    # Seed default roles if they do not exist
    cursor.execute("SELECT COUNT(*) FROM roles")
    if cursor.fetchone()[0] == 0:
        _seed_default_permissions(cursor)

    conn.commit()
    conn.close()
    print("[ACCESS CONTROL] Database initialized.")


def _seed_default_permissions(cursor):
    """Seeds default admin and user roles matching original ROLE_PERMISSIONS."""

    # Insert roles
    cursor.execute("INSERT INTO roles (role_name) VALUES ('admin')")
    cursor.execute("INSERT INTO roles (role_name) VALUES ('user')")

    admin_id = cursor.execute(
        "SELECT role_id FROM roles WHERE role_name='admin'"
    ).fetchone()[0]

    user_id = cursor.execute(
        "SELECT role_id FROM roles WHERE role_name='user'"
    ).fetchone()[0]

    # Admin permissions — all tables, all operations, no row filter
    admin_tables = ["students", "enrollments", "courses"]
    for table in admin_tables:
        cursor.execute("""
            INSERT INTO role_table_permissions
            (role_id, db_name, table_name, can_select, can_insert, can_update, can_delete)
            VALUES (?, '*', ?, 1, 1, 1, 1)
        """, (admin_id, table))
        # None in allowed_columns means all columns permitted — no column restrictions

    # User permissions — students table only, SELECT only, restricted columns
    cursor.execute("""
        INSERT INTO role_table_permissions
        (role_id, db_name, table_name, can_select, can_insert, can_update, can_delete)
        VALUES (?, '*', 'students', 1, 0, 0, 0)
    """, (user_id,))

    user_columns = ["student_id", "name", "city", "gpa"]
    for col in user_columns:
        cursor.execute("""
            INSERT INTO role_column_permissions
            (role_id, db_name, table_name, column_name)
            VALUES (?, '*', 'students', ?)
        """, (user_id, col))

    # User row filter
    cursor.execute("""
        INSERT INTO role_row_filters
        (role_id, db_name, table_name, filter_condition)
        VALUES (?, '*', 'students', 'student_id = {user_id}')
    """, (user_id,))

    # Seed default users
    cursor.execute("""
        INSERT INTO users (user_id, username, role_id)
        VALUES (1, 'admin1', ?)
    """, (admin_id,))

    cursor.execute("""
        INSERT INTO users (user_id, username, role_id)
        VALUES (2, 'admin2', ?)
    """, (admin_id,))

    cursor.execute("""
        INSERT INTO users (user_id, username, role_id)
        VALUES (5, 'user5', ?)
    """, (user_id,))

    cursor.execute("""
        INSERT INTO users (user_id, username, role_id)
        VALUES (6, 'user6', ?)
    """, (user_id,))

    print("[ACCESS CONTROL] Default roles and permissions seeded.")


# ─── Core Access Control Functions ───────────────────────────────────────────

def get_user_role(user_id: int) -> str | None:
    """
    Looks up a user's role name from the users table.
    Returns None if user does not exist or is inactive.
    """
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()
    result = cursor.execute("""
        SELECT r.role_name
        FROM users u
        JOIN roles r ON u.role_id = r.role_id
        WHERE u.user_id = ? AND u.is_active = 1
    """, (user_id,)).fetchone()
    conn.close()
    return result[0] if result else None


def get_allowed_operations(role: str) -> list[str]:
    """
    Returns the list of SQL operations permitted for a role.
    Derived from the table permissions — if any table allows insert/update/delete,
    the role is considered to have that operation permission.
    """
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()

    role_id = _get_role_id(cursor, role)
    if not role_id:
        conn.close()
        return ["SELECT"]

    result = cursor.execute("""
        SELECT MAX(can_select), MAX(can_insert),
               MAX(can_update), MAX(can_delete)
        FROM role_table_permissions
        WHERE role_id = ?
    """, (role_id,)).fetchone()

    conn.close()

    ops = []
    if result[0]: ops.append("SELECT")
    if result[1]: ops.append("INSERT")
    if result[2]: ops.append("UPDATE")
    if result[3]: ops.append("DELETE")
    return ops


def get_allowed_tables(role: str, db_name: str = "*") -> list[str]:
    """
    Returns list of table names the role can access for a given database.
    db_name='*' matches permissions defined for all databases.
    """
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()

    role_id = _get_role_id(cursor, role)
    if not role_id:
        conn.close()
        return []

    results = cursor.execute("""
        SELECT table_name FROM role_table_permissions
        WHERE role_id = ?
        AND (db_name = ? OR db_name = '*')
        AND can_select = 1
    """, (role_id, db_name)).fetchall()

    conn.close()
    return [r[0] for r in results]


def get_allowed_columns(role: str, table_name: str, db_name: str = "*") -> list[str] | None:
    """
    Returns list of allowed column names for a role on a specific table.
    Returns None if all columns are permitted (no column restrictions defined).
    """
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()

    role_id = _get_role_id(cursor, role)
    if not role_id:
        conn.close()
        return None

    results = cursor.execute("""
        SELECT column_name FROM role_column_permissions
        WHERE role_id = ?
        AND table_name = ?
        AND (db_name = ? OR db_name = '*')
    """, (role_id, table_name, db_name)).fetchall()

    conn.close()

    # No column restrictions defined — all columns permitted
    if not results:
        return None

    return [r[0] for r in results]


def get_row_filters(role: str, user_id: int | None, db_name: str = "*") -> dict:
    """
    Returns resolved row-level filter conditions for a role and user.
    Replaces {user_id} placeholder with actual user_id value.
    Returns empty dict if no row filters apply or user_id is None.
    """
    if user_id is None:
        return {}

    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()

    role_id = _get_role_id(cursor, role)
    if not role_id:
        conn.close()
        return {}

    results = cursor.execute("""
        SELECT table_name, filter_condition FROM role_row_filters
        WHERE role_id = ?
        AND (db_name = ? OR db_name = '*')
    """, (role_id, db_name)).fetchall()

    conn.close()

    return {
        table: condition.replace("{user_id}", str(user_id))
        for table, condition in results
    }


def build_role_permissions_dict(role: str, db_name: str = "*") -> dict:
    """
    Builds a permissions dict in the same format as the original
    hardcoded ROLE_PERMISSIONS for backward compatibility with
    existing code that expects that structure.
    """
    allowed_tables = get_allowed_tables(role, db_name)
    allowed_columns = {}
    for table in allowed_tables:
        cols = get_allowed_columns(role, table, db_name)
        allowed_columns[table] = cols  # None means all columns

    return {
        "allowed_tables":     allowed_tables,
        "allowed_columns":    allowed_columns,
        "allowed_operations": get_allowed_operations(role),
        "row_filter":         {}  # raw patterns without user_id substitution
    }


# ─── User Management ──────────────────────────────────────────────────────────

def add_user(user_id: int, username: str, role: str) -> bool:
    """Adds a new user with the specified role."""
    try:
        conn = sqlite3.connect(ACCESS_DB_PATH)
        cursor = conn.cursor()
        role_id = _get_role_id(cursor, role)
        if not role_id:
            print(f"[ACCESS CONTROL] Role '{role}' not found.")
            conn.close()
            return False
        cursor.execute("""
            INSERT INTO users (user_id, username, role_id)
            VALUES (?, ?, ?)
        """, (user_id, username, role_id))
        conn.commit()
        conn.close()
        print(f"[ACCESS CONTROL] User '{username}' (id={user_id}) added with role '{role}'.")
        return True
    except sqlite3.IntegrityError as e:
        print(f"[ACCESS CONTROL] Failed to add user: {e}")
        return False


def deactivate_user(user_id: int) -> bool:
    """Deactivates a user — they can no longer log in."""
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE users SET is_active = 0 WHERE user_id = ?",
        (user_id,)
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected:
        print(f"[ACCESS CONTROL] User {user_id} deactivated.")
    return affected > 0


def change_user_role(user_id: int, new_role: str) -> bool:
    """Changes a user's role."""
    conn = sqlite3.connect(ACCESS_DB_PATH)
    cursor = conn.cursor()
    role_id = _get_role_id(cursor, new_role)
    if not role_id:
        print(f"[ACCESS CONTROL] Role '{new_role}' not found.")
        conn.close()
        return False
    cursor.execute(
        "UPDATE users SET role_id = ? WHERE user_id = ?",
        (role_id, user_id)
    )
    conn.commit()
    affected = cursor.rowcount
    conn.close()
    if affected:
        print(f"[ACCESS CONTROL] User {user_id} role changed to '{new_role}'.")
    return affected > 0


# ─── Helper ───────────────────────────────────────────────────────────────────

def _get_role_id(cursor, role_name: str) -> int | None:
    """Returns role_id for a role name or None if not found."""
    result = cursor.execute(
        "SELECT role_id FROM roles WHERE role_name = ?",
        (role_name,)
    ).fetchone()
    return result[0] if result else None


# ─── Initialize on import ─────────────────────────────────────────────────────

init_access_control_db()
