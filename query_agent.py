import os
import re
import time
from dotenv import load_dotenv
load_dotenv()
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from llm_manager import invoke_llm, invoke_verifier_llm, invoke_coder_llm
from langchain_core.messages import HumanMessage, SystemMessage
from database import get_schema, run_sql, get_db_statistics
import uuid
from datetime import datetime, timezone
import json
import memory as mem
import access_control as ac
import lock_manager as _lm





def get_content(response) -> str:
    """Extracts plain text from an LLM response.
    Handles both string content (Groq) and list content with thinking blocks (Gemini)."""
    content = response.content
    if isinstance(content, str):
        return content
    for block in content:
        if isinstance(block, dict) and block.get("type") == "text":
            return block.get("text", "")
    return str(content)


# ─── Message Log ──────────────────────────────────────────────────────────────
message_log = []


def log_message(msg: dict):
    """Records every message in the communication log."""
    message_log.append(msg)
    print(f"\n{'='*60}")
    print(f"[MESSAGE LOG]")
    print(f"  CONV:     {msg['conversation_id']}")
    print(f"  ID:       {msg['message_id']}")
    print(f"  FROM:     {msg['sender']}")
    print(f"  TO:       {msg['receiver']}")
    print(f"  TYPE:     {msg['message_type']}")
    print(f"  TIME:     {msg['timestamp']}")
    print(f"  CONTENT:  {list(msg['content'].keys())}")
    print(f"  MEMORY:")
    msg_memory = msg.get("memory", {})
    short = msg_memory.get("short_term", "")
    print(
        f"    short_term:  {short[:80]}..."
        if len(short) > 80
        else f"    short_term:  {short}"
    )
    # FIX 3: updated episodic display to reflect new architecture
    episodic = msg_memory.get("episodic", {})
    print(f"    coordinator_retries: {episodic.get('coordinator_retries', 0)}")
    print(f"    current_task:        {episodic.get('current_task_id', 'none')}")
    print(f"    semantic tables: {msg['memory'].get('semantic_summary', [])}")  # summary only — schema not stored in memory
    print(f"{'='*60}")


# ─── Schema Linking ───────────────────────────────────────────────────────────

def link_schema(question: str, full_schema: str) -> str:
    """Filters the schema to only tables relevant to the question."""
    response = invoke_llm([
        SystemMessage(content="""You are a schema linking expert.

Given a natural language question and a full database schema:

1. Return all tables directly related to the question
2. ALSO return tables connected through foreign keys that may be needed for joins
3. When unsure, include the table rather than excluding it
4. Never omit a table that could contain filtering, aggregation, or date information
5. Prefer returning slightly more schema instead of too little

Return ONLY the CREATE TABLE statements.
No explanation."""),
        HumanMessage(content=f"Question: {question}\n\nFull Schema:\n{full_schema}\n\nRelevant tables only:")
    ])
    linked = get_content(response).strip()
    return linked if "CREATE TABLE" in linked else full_schema


def smart_link_schema(question: str, permission_schema: str) -> str:
    """
    Schema linking that skips the LLM call when only one table is visible.
    If the permission schema has only one table, schema linking cannot do anything
    useful — the result will always be that one table. Skip the LLM call entirely.
    If multiple tables are visible, call link_schema normally to pick relevant ones.
    """
    tables = extract_table_names(permission_schema)
    if len(tables) <= 1:
        print(f"[SCHEMA LINK] Single table schema — skipping LLM call")
        return permission_schema
    print(f"[SCHEMA LINK] Multiple tables {tables} — running LLM schema linking")
    return link_schema(question, permission_schema)


def extract_table_names(schema: str) -> list:
    """Extracts table names from schema string for semantic summary."""
    return [
        line.split("(")[0].replace("CREATE TABLE", "").strip()
        for line in schema.split("\n")
        if line.strip().upper().startswith("CREATE TABLE")
    ]


def get_operation_type(sql: str) -> str:
    sql_stripped = sql.strip().upper()
    if sql_stripped.startswith("SELECT"):
        return "SELECT"
    elif sql_stripped.startswith("INSERT"):
        return "INSERT"
    elif sql_stripped.startswith("UPDATE"):
        return "UPDATE"
    elif sql_stripped.startswith("DELETE"):
        return "DELETE"
    elif sql_stripped.startswith("BEGIN"):
        return "TRANSACTION_BLOCK"
    else:
        return "UNKNOWN"




def take_db_snapshot(db_path: str, schema: str) -> dict:
    """Returns {table_name: row_count} for every table in schema."""
    snapshot = {}
    for table in extract_table_names(schema):
        result = run_sql(db_path, f'SELECT COUNT(*) FROM "{table}";')
        try:
            row_dict = eval(result.strip())
            snapshot[table] = int(list(row_dict.values())[0])
        except Exception:
            snapshot[table] = 0
    return snapshot


import re

def parse_insert_row(sql: str) -> dict:
    # Normalize whitespace
    sql_clean = re.sub(r'\s+', ' ', sql).strip()
    
    match = re.search(
        r'INSERT\s+INTO\s+\w+\s*\(([^)]+)\)\s*VALUES\s*\((.+)\)\s*;?\s*$',
        sql_clean, re.IGNORECASE
    )
    if not match:
        print(f"[parse_insert_row] NO MATCH on: {sql_clean}")
        return {}

    cols = [c.strip().strip("'\"` ") for c in match.group(1).split(',')]
    vals_raw = match.group(2)

    vals = []
    for token in re.findall(r"'[^']*'|\"[^\"]*\"|[^,]+", vals_raw):
        token = token.strip()
        if len(token) >= 2 and token[0] in ("'", '"') and token[-1] == token[0]:
            token = token[1:-1]
        vals.append(token)

    if len(cols) != len(vals):
        print(f"[parse_insert_row] MISMATCH: cols={cols} vals={vals}")
        return {}

    print(f"[parse_insert_row] SUCCESS: {dict(zip(cols, vals))}")
    return dict(zip(cols, vals))


def pre_execution_check(sql: str, db_path: str, schema: str) -> tuple:
    schema_tables = set(extract_table_names(schema))
    tables_in_sql = list(
        set(re.findall(r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', sql, re.IGNORECASE))
        & schema_tables
    )

    # --- Circular FK detection ---
    fk_graph = {t: [] for t in tables_in_sql}
    for table in tables_in_sql:
        fk_result = run_sql(db_path, f'PRAGMA foreign_key_list("{table}");')
        if "No results" not in fk_result and "ERROR" not in fk_result:
            for row in fk_result.splitlines():
                try:
                    row_dict = eval(row)
                    parent = row_dict.get("table", "")
                    if parent in tables_in_sql:
                        fk_graph[table].append(parent)
                except Exception:
                    pass

    def has_cycle(graph):
        visited, rec_stack = set(), set()
        def dfs(node):
            visited.add(node); rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor): return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False
        return any(dfs(n) for n in graph if n not in visited)

    if has_cycle(fk_graph):
        involved = [t for t in tables_in_sql if fk_graph.get(t)]
        return ("CIRCULAR", f"Circular FK dependency among: {involved}")

    operation = get_operation_type(sql)
    sql_upper = sql.upper()
    statements = [s.strip() for s in sql.split(";") if s.strip()]

    if len(statements) > 1:
        if operation == "DELETE":
            return (
                "CONSTRAINT",
                f"Multi-statement DELETE requires explicit approval. "
                f"Tables: {[s.split()[2] for s in statements if s.upper().startswith('DELETE')]}"
            )
        return (None, None)

    # Pre-parse INSERT row once
    insert_row = parse_insert_row(sql) if operation == "INSERT" else {}

    print(f"[pre_execution_check] operation={operation}")
    print(f"[pre_execution_check] tables_in_sql={tables_in_sql}")
    print(f"[pre_execution_check] insert_row={insert_row}")

    def get_val_for_col(col_name: str):
        if insert_row and col_name in insert_row:
            return insert_row[col_name]
        m = re.search(
            rf'["\']?{re.escape(col_name)}["\']?\s*=\s*["\']?(\w+)["\']?',
            sql, re.IGNORECASE
        )
        return m.group(1) if m else None

    for table in tables_in_sql:
        table_info_result = run_sql(db_path, f'PRAGMA table_info("{table}");')
        print(f"[pre_execution_check] PRAGMA table_info({table}): {table_info_result}")

        columns = []
        if "ERROR" not in table_info_result and "No results" not in table_info_result:
            for row in table_info_result.splitlines():
                try:
                    columns.append(eval(row))
                except Exception:
                    pass

        print(f"[pre_execution_check] columns parsed for {table}: {columns}")

        # --- NOT NULL check ---
        if operation == "INSERT":
            for col in columns:
                col_name    = col.get("name", "")
                is_pk       = col.get("pk", 0)
                notnull     = col.get("notnull", 0)
                has_default = col.get("dflt_value") is not None

                if notnull and not has_default and not is_pk:
                    if col_name.upper() not in sql_upper:
                        return (
                            "CONSTRAINT",
                            f"NOT NULL violation: column '{col_name}' in '{table}' requires a value"
                        )

        # --- PRIMARY KEY duplicate check ---
        if operation == "INSERT":
            pk_cols = [col for col in columns if col.get("pk", 0)]
            print(f"[pre_execution_check] pk_cols for {table}: {pk_cols}")
            for col in pk_cols:
                col_name = col.get("name", "")
                val = get_val_for_col(col_name)
                print(f"[pre_execution_check] PK check: col={col_name}, val={val}")
                if val is not None:
                    chk = run_sql(
                        db_path,
                        f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" = \'{val}\';'
                    )
                    print(f"[pre_execution_check] PK COUNT query result: {chk}")
                    try:
                        count = int(list(eval(chk.strip()).values())[0])
                        print(f"[pre_execution_check] PK count={count}")
                        if count > 0:
                            return (
                                "CONSTRAINT",
                                f"PRIMARY KEY constraint violation: '{col_name}' = {val} "
                                f"already exists in '{table}'"
                            )
                    except Exception as e:
                        print(f"[pre_execution_check] PK count eval failed: {e}, raw={chk}")

        # --- UNIQUE index check ---
        if operation in ("INSERT", "UPDATE"):
            idx_result = run_sql(db_path, f'PRAGMA index_list("{table}");')
            print(f"[pre_execution_check] PRAGMA index_list({table}): {idx_result}")
            if "ERROR" not in idx_result and "No results" not in idx_result:
                for idx_row in idx_result.splitlines():
                    try:
                        idx = eval(idx_row)
                        if idx.get("unique", 0):
                            idx_name = idx.get("name", "")
                            info = run_sql(db_path, f'PRAGMA index_info("{idx_name}");')
                            if "ERROR" not in info and "No results" not in info:
                                for info_row in info.splitlines():
                                    try:
                                        col_name = eval(info_row).get("name", "")
                                        val = get_val_for_col(col_name)
                                        print(f"[pre_execution_check] UNIQUE check: col={col_name}, val={val}")
                                        if val is not None:
                                            chk = run_sql(
                                                db_path,
                                                f'SELECT COUNT(*) FROM "{table}" '
                                                f'WHERE "{col_name}" = \'{val}\';'
                                            )
                                            count = int(list(eval(chk.strip()).values())[0])
                                            print(f"[pre_execution_check] UNIQUE count={count}")
                                            if count > 0:
                                                return (
                                                    "CONSTRAINT",
                                                    f"UNIQUE constraint violation: '{col_name}' = {val} "
                                                    f"already exists in '{table}'"
                                                )
                                    except Exception as e:
                                        print(f"[pre_execution_check] UNIQUE inner eval failed: {e}")
                    except Exception as e:
                        print(f"[pre_execution_check] UNIQUE outer eval failed: {e}")

        # --- FK existence check ---
        fk_result = run_sql(db_path, f'PRAGMA foreign_key_list("{table}");')
        if "ERROR" not in fk_result and "No results" not in fk_result:
            for fk_row in fk_result.splitlines():
                try:
                    fk = eval(fk_row)
                    fk_col       = fk.get("from", "")
                    parent_table = fk.get("table", "")
                    parent_col   = fk.get("to", "")
                    val = get_val_for_col(fk_col)
                    print(f"[pre_execution_check] FK check: fk_col={fk_col}, val={val}, parent={parent_table}.{parent_col}")
                    if val is not None:
                        chk = run_sql(
                            db_path,
                            f'SELECT COUNT(*) FROM "{parent_table}" '
                            f'WHERE "{parent_col}" = \'{val}\';'
                        )
                        count = int(list(eval(chk.strip()).values())[0])
                        print(f"[pre_execution_check] FK count={count}")
                        if count == 0:
                            return (
                                "CONSTRAINT",
                                f"FK violation: no matching row in '{parent_table}' "
                                f"for {fk_col} = {val}"
                            )
                except Exception as e:
                    print(f"[pre_execution_check] FK eval failed: {e}")

    print(f"[pre_execution_check] all checks passed, returning (None, None)")
    return (None, None)


def post_execution_check(sql: str, db_path: str, schema: str, snapshot_before: dict) -> tuple:
    """
    Checks for TRANSACTION anomalies after a successful write.
    Returns (conflict_type, description) or (None, None) if clean.
    """
    snapshot_after = take_db_snapshot(db_path, schema)
    operation = get_operation_type(sql)
    tables_in_sql = list(set(re.findall(r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', sql, re.IGNORECASE)))
    deltas = {t: snapshot_after.get(t, 0) - snapshot_before.get(t, 0) for t in snapshot_after}

    if operation == "INSERT":
        affected = [t for t in tables_in_sql if t in deltas]
        if affected and all(deltas.get(t, 0) <= 0 for t in affected):
            return (
                "TRANSACTION",
                "INSERT executed without error but no rows were added to any affected table — possible silent transaction failure"
            )
    elif operation == "DELETE":
        for table in tables_in_sql:
            delta = deltas.get(table, 0)
            before = snapshot_before.get(table, 0)
            if delta < 0 and before > 0 and abs(delta) > before * 0.5:
                return (
                    "TRANSACTION",
                    f"DELETE removed more than 50% of rows in table {table} — possible overly broad WHERE condition or silent transaction anomaly"
                )

    return (None, None)


# ─── Message Protocol ─────────────────────────────────────────────────────────

def create_message(
    sender: str, receiver: str, msg_type: str,
    content: dict, memory: dict, conversation_id: str
) -> dict:
    """
    Creates an explicit structured message passed between agents.
    Every message carries:
      - Routing metadata: conversation_id, message_id, sender, receiver, timestamp, message_type
      - Content: the work data for the receiving agent
      - Memory:
          short_term         — accumulates context within one question session
          episodic           — coordinator_retries count and current task id
          semantic           — stable filtered schema, set once and never changed
          semantic_summary   — human-readable list of table names for inspection
    """
    msg = {
        "conversation_id": conversation_id,
        "message_id": str(uuid.uuid4()),
        "sender": sender,
        "receiver": receiver,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "message_type": msg_type,
        "content": content,
        "memory": {
            "short_term": memory.get("short_term", ""),
            "episodic": {
                "coordinator_retries": memory.get("episodic", {}).get("coordinator_retries", 0),
                "current_task_id":     memory.get("episodic", {}).get("current_task_id", "none")
            },
            "semantic_summary": extract_table_names(memory.get("semantic", ""))
        }
    }
    log_message(msg)
    return msg


# ─── State ────────────────────────────────────────────────────────────────────

class QueryState(TypedDict):
    message:             dict   # current message — LangGraph routing
    verdict:             str    # routing signal for conditional edges
    coordinator_retries: int    # retry counter for routing guard
    sql_candidates:      list   # fallback SQL queue
    benchmark_mode:      bool   # session flag — set once, never in messages
    plan:                dict   # Coordinator-only working memory
    task_list:           list   # Coordinator-only working memory
    no_plan:             bool   # skip planning LLM — build minimal default plan instead
    no_verify:           bool   # skip verifier — execute coder SQL directly
    no_schema_link:      bool   # skip schema linking — use full permission schema directly
    no_memory:           bool   # skip episodic and procedural memory retrieval
    replan_count:        int


# ─── Coordinator ──────────────────────────────────────────────────────────────

def coordinator(state: QueryState) -> QueryState:
    print("\n[COORDINATOR] Coordinating...")

    incoming = state["message"]
    conversation_id = incoming["conversation_id"]
    coordinator_retries = state["coordinator_retries"]
    task_list           = list(state["task_list"])

    # ── First call: no task list yet ──────────────────────────────────────────
    if not task_list:
        question      = incoming["content"]["question"]
        db_path       = incoming["content"]["db_path"]
        role          = incoming["content"]["role"]
        user_id       = incoming["content"].get("user_id", None)
        conflict_constraints = incoming["content"].get("conflict_constraints", [])
        benchmark_mode = state.get("benchmark_mode", False)
        no_schema_link = state.get("no_schema_link", False)
        allowed_operations = ac.get_allowed_operations(role)
        allowed_ops_str = ", ".join(allowed_operations)

        try:
            full_schema = mem.get_full_schema_cached(db_path, get_schema)
        except Exception as e:
            print(f"[COORDINATOR] Could not read schema: {e}")
            outgoing = create_message(
                sender="coordinator", receiver="end",
                msg_type="result",
                content={"question": question, "result": f"System error: database unavailable ({e})", "verdict": "fail", "role": role},
                memory={"short_term": "", "episodic": {"coordinator_retries": 0, "current_task_id": "none"}, "semantic": ""},
                conversation_id=conversation_id
            )
            try:
                _lm.clear_intent(conversation_id)
            except Exception:
                pass
            return {**state, "message": outgoing, "verdict": "fail"}

        if benchmark_mode:
            # Benchmark mode — skip all access control, use full schema
            print("[COORDINATOR] Benchmark mode — access control disabled")
            allowed_schema = full_schema if state.get("no_schema_link", False) else smart_link_schema(question, full_schema)
            row_filters = {}

        else:
            # ── Layer 1: table-level access check — fast keyword check, no LLM ──────
            print(f"[COORDINATOR] Checking table access for role '{role}'...")
            allowed_tables = ac.get_allowed_tables(role)
            allowed_tables_lower = [t.lower() for t in allowed_tables]
            all_table_names = [
                line.split("(")[0].replace("CREATE TABLE", "").strip().lower()
                for line in full_schema.split("\n")
                if line.strip().upper().startswith("CREATE TABLE")
            ]
            forbidden_mentioned = [
                t for t in all_table_names
                if t in question.lower() and t not in allowed_tables_lower
            ]
            access_allowed = not forbidden_mentioned
            access_reason = (
                f"Access denied: you do not have permission to access "
                f"table(s): {forbidden_mentioned}. "
                f"Permitted tables: {allowed_tables_lower}."
                if forbidden_mentioned else ""
            )
            if not access_allowed:
                print(f"[COORDINATOR] ACCESS DENIED: {access_reason}")
                outgoing = create_message(
                    sender="coordinator", receiver="end",
                    msg_type="result",
                    content={
                        "question": question,
                        "result":   access_reason,
                        "verdict":  "fail"
                    },
                    memory={
                        "short_term": f"ACCESS DENIED: {access_reason}",
                        "episodic":   {"coordinator_retries": 0, "current_task_id": "none"},
                        "semantic":   ""
                    },
                    conversation_id=conversation_id
                )
                try:
                    _lm.clear_intent(conversation_id)
                except Exception:
                    pass
                return {**state, "message": outgoing, "verdict": "fail"}

            # ── Layer 2: build permission-filtered schema — LLM never sees forbidden tables/columns
            print(f"[COORDINATOR] Building permission-filtered schema for role '{role}'...")
            role_permissions = ac.build_role_permissions_dict(role)
            permission_schema = mem.build_filtered_schema_from_string(full_schema, role, {role: role_permissions})
            if not permission_schema:
                access_reason = f"Access denied: role '{role}' has no permitted tables in this database."
                print(f"[COORDINATOR] ACCESS DENIED: {access_reason}")
                outgoing = create_message(
                    sender="coordinator", receiver="end",
                    msg_type="result",
                    content={"question": question, "result": access_reason, "verdict": "fail", "role": role},
                    memory={"short_term": f"ACCESS DENIED: {access_reason}", "episodic": {"coordinator_retries": 0, "current_task_id": "none"}, "semantic": ""},
                    conversation_id=conversation_id
                )
                try:
                    _lm.clear_intent(conversation_id)
                except Exception:
                    pass
                return {**state, "message": outgoing, "verdict": "fail"}
            allowed_schema = permission_schema if no_schema_link else smart_link_schema(question, permission_schema)
            print(f"[COORDINATOR] Permitted tables: {extract_table_names(allowed_schema)}")

            # ── Layer 3: row-level security — inject user-specific WHERE constraints
            row_filters = ac.get_row_filters(role, user_id)
            if row_filters:
                print(f"[COORDINATOR] Row-level filters applied: {row_filters}")

        # Build initial_state — logged before any LLM call
        initial_state_data = {
            "conversation_id": conversation_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "user_id": user_id,
            "allowed_ops": allowed_operations,
            "question": question,
            "db_path": db_path,
            "status": "initialised"
        }

        create_message(
            sender="coordinator", receiver="coordinator",
            msg_type="instruction",
            content={"initial_state": initial_state_data},
            memory={
                "short_term": "",
                "episodic": {"coordinator_retries": 0, "current_task_id": "none"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )

        if state.get("no_memory", False):
            episodic_hints = []
            procedural_hints = []
        else:
            # Retrieve episodic memory — past episodes for similar questions
            #episodic_hints = []
            episodic_hints = mem.episodic_retrieve(
                 question=question,
                 db_path=db_path,
                 schema=allowed_schema,
                 user_id=user_id if role == "user" else None
             )
            #procedural_hints = []
            # Retrieve procedural memory — verified SQL hints for similar questions
            procedural_hints = mem.procedural_retrieve(
                 question=question,
                 db_path=db_path,
                 schema=allowed_schema,
                 role=role,
                 user_id=user_id if role == "user" else None
             )

        coder_agent = "user_coder" if role == "user" else "admin_coder"

        if state.get("no_plan", False):
            # No-plan mode — skip planning LLM, build minimal default plan directly
            print("[COORDINATOR] No-plan mode — skipping planning LLM call")
            if benchmark_mode:
                allowed_schema = full_schema if state.get("no_schema_link", False) else smart_link_schema(question, full_schema)
            else:
                allowed_schema = full_schema if state.get("no_schema_link", False) else smart_link_schema(question, permission_schema)
            procedural_hints = [] if state.get("no_schema_link", False) else procedural_hints
            episodic_hints = [] if state.get("no_schema_link", False) else episodic_hints
            plan = {
                "initial_state": initial_state_data,
                "goal": question,
                "description": {
                    "operation":     "SELECT",
                    "complexity":    "simple",
                    "logical_steps": "",
                    "values":        None
                },
                "constraints": [],
                "agents": [coder_agent, "verifier"],
                "allowed_schema": allowed_schema,
                "row_filters": {}
            }
        else:
            # Generate Plan tuple via LLM
            print("[COORDINATOR] Generating plan...")
            response = invoke_llm([
                SystemMessage(content="""You are a SQL planning expert for a database management system.
Given a natural language question and schema, generate a structured plan.
The plan should describe WHAT information is needed and the reasoning strategy, not HOW to write the SQL query.

Do NOT describe:
- SQL syntax, JOIN clauses, GROUP BY clauses, HAVING clauses, subqueries, or exact query structure

Focus only on:
- entities involved and their relationships
- filtering and comparison logic
- business reasoning steps
- security or operational constraints
- exact literal values that must be matched as stored in the database

Complexity levels:
- simple: single entity straightforward lookup or filter
- comparative: requires comparing entities against a threshold or each other
- analytical: requires multi-step reasoning or aggregation over groups
- relational: requires combining information from multiple entities
- set_operation: requires overlap, exclusion, or comparison between two result sets — use INTERSECT or EXCEPT

Reply in this exact format with no extra text:

GOAL: <one sentence describing what the user wants returned>
OPERATION: <SELECT|INSERT|UPDATE|DELETE>
COMPLEXITY: <simple|comparative|analytical|relational|set_operation>
LOGICAL_STEPS:
1. <high-level reasoning step>
2. <high-level reasoning step>
3. <high-level reasoning step>
VALUES: <values to insert or update, or None>
TARGET_ENTITY: <for INSERT, UPDATE, DELETE only — the primary key of the targeted row in format column=value, e.g. student_id=3. For bulk operations write: multiple. For SELECT write: None. Never write just a table name.>
CONSTRAINTS:
- <mandatory rules the SQL MUST satisfy — security requirements, hard exclusions, specific output requirements, or exact literal values that must be matched as stored in the database. If nothing mandatory applies, write: None>"""),
                HumanMessage(content=(
                    f"Question: {question}\n"
                    f"Role: {role}\n"
                    f"Allowed operations: {allowed_ops_str}\n"
                    f"Schema: {allowed_schema}"
                    + (
                        "\n\nSimilar past questions on this database (use to guide your plan):\n"
                        + "".join(
                            (
                                f"  WORKED (similarity={ep['similarity']}): {ep['question']}\n"
                                f"  Plan that succeeded — consider reusing this structure:\n"
                                f"    Operation: {ep['plan_description'].get('operation', 'none')}\n"
                                f"    Complexity: {ep['plan_description'].get('complexity', 'none')}\n"
                                f"    Logical Steps: {ep['plan_description'].get('logical_steps', 'none')}\n\n"
                                if ep['verdict'] == 'pass'
                                else
                                f"  FAILED after {ep['retries_used']} retries (similarity={ep['similarity']}): {ep['question']}\n"
                                f"  Plan that failed — do NOT repeat this, try a different approach:\n"
                                f"    Operation: {ep['plan_description'].get('operation', 'none')}\n"
                                f"    Complexity: {ep['plan_description'].get('complexity', 'none')}\n"
                                f"    Logical Steps: {ep['plan_description'].get('logical_steps', 'none')}\n\n"
                            )
                            for ep in episodic_hints
                        )
                        if episodic_hints else ""
                    )
                    + (
                        "\n\nCONFLICT RESOLUTION CONSTRAINTS — you MUST incorporate "
                        "these into your plan:\n"
                        + "\n".join(f"  - {c}" for c in conflict_constraints)
                        if conflict_constraints else ""
                    )
                ))
            ])

            raw = get_content(response).strip()

            goal = ""
            operation = ""
            complexity = "simple"
            intent_lines = []
            values = "None"
            target_entity_from_plan = ""
            constraints = []
            in_constraints = False
            in_logical_steps = False

            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("GOAL:"):
                    goal = line[5:].strip()
                    in_constraints = False
                    in_logical_steps = False
                elif line.startswith("OPERATION:"):
                    operation = line[10:].strip()
                    in_constraints = False
                    in_logical_steps = False
                elif line.startswith("COMPLEXITY:"):
                    complexity = line[11:].strip().lower()
                    in_constraints = False
                    in_logical_steps = False
                elif line.startswith("LOGICAL_STEPS:"):
                    in_logical_steps = True
                    in_constraints = False
                elif in_logical_steps and line and line[0].isdigit() and "." in line:
                    step = line.split(".", 1)[1].strip()
                    intent_lines.append(step)
                elif line.startswith("VALUES:"):
                    values = line[7:].strip()
                    in_constraints = False
                    in_logical_steps = False
                elif line.startswith("TARGET_ENTITY:"):
                    target_entity_from_plan = line[14:].strip()
                    in_constraints = False
                    in_logical_steps = False
                elif line.startswith("CONSTRAINTS:"):
                    in_constraints = True
                    in_logical_steps = False
                elif in_constraints and line.startswith("-"):
                    constraints.append(line[1:].strip())
                elif in_logical_steps and line and not any(line.startswith(k) for k in ["VALUES:", "TARGET_ENTITY:", "CONSTRAINTS:"]):
                    intent_lines.append(line)

            intent = " ".join(intent_lines)

            # Reject ambiguous or unexecutable operations early
            if not goal or not operation or operation == "UNKNOWN":
                print("[COORDINATOR] Could not determine a valid operation from the question.")
                outgoing = create_message(
                    sender="coordinator", receiver="end",
                    msg_type="result",
                    content={
                        "question": question,
                        "result": "Could not determine a valid database operation from your request.",
                        "verdict": "fail",
                        "role":     role
                    },
                    memory={
                        "short_term": "Plan generation failed: unknown operation",
                        "episodic": {"coordinator_retries": 0, "current_task_id": "none"},
                        "semantic": allowed_schema
                    },
                    conversation_id=conversation_id
                )
                try:
                    _lm.clear_intent(conversation_id)
                except Exception:
                    pass
                return {**state, "message": outgoing, "verdict": "fail"}

            # Rule-based operation permission check — no LLM needed
            # This runs before dispatching to the Coder, saving one wasted LLM call
            if operation not in allowed_operations:
                print(f"[COORDINATOR] Operation '{operation}' not permitted for role '{role}'")
                access_msg = (
                    f"Access denied: operation '{operation}' is not permitted for your role '{role}'. "
                    f"Permitted operations: {allowed_operations}."
                )
                outgoing = create_message(
                    sender="coordinator", receiver="end",
                    msg_type="result",
                    content={
                        "question": question,
                        "result":   access_msg,
                        "verdict":  "fail",
                        "role":     role
                    },
                    memory={
                        "short_term": f"ACCESS DENIED: operation {operation} not permitted for role {role}",
                        "episodic":   {"coordinator_retries": 0, "current_task_id": "none"},
                        "semantic":   allowed_schema
                    },
                    conversation_id=conversation_id
                )
                try:
                    _lm.clear_intent(conversation_id)
                except Exception:
                    pass
                return {**state, "message": outgoing, "verdict": "fail"}

            plan = {
                "initial_state": initial_state_data,
                "goal": goal,
                "description": {
                    "operation":     operation,
                    "complexity":    complexity,
                    "logical_steps": intent,
                    "values":        values if operation in ("INSERT", "UPDATE") else None,
                    "target_entity": target_entity_from_plan
                },
                "constraints": constraints,
                "agents": [coder_agent, "verifier"],
                "allowed_schema": allowed_schema
            }

        # Inject row-level security constraints into the plan
        if row_filters:
            for table, condition in row_filters.items():
                plan["constraints"].append(
                    f"SECURITY REQUIREMENT: for table '{table}' you MUST include "
                    f"WHERE {condition} in your SQL — "
                    f"the user can only access their own rows, never other users' data"
                )
            plan["row_filters"] = row_filters
        else:
            plan["row_filters"] = {}

        print(f"[COORDINATOR] Goal: {plan['goal']}")
        print(f"[COORDINATOR] Operation: {plan['description']['operation']}, Complexity: {plan['description']['complexity']}")
        # ── Semantic conflict detection and resolution ────────────────────────────
        _intent_operation = plan["description"]["operation"]
        _intent_tables = extract_table_names(plan["allowed_schema"])
        _replan_count = state.get("replan_count", 0)

        if _intent_operation in ("DELETE", "UPDATE", "INSERT"):
            _target_entity_raw = plan["description"].get("target_entity", "")
            if (not _target_entity_raw
                    or _target_entity_raw.lower() == "none"
                    or _target_entity_raw.lower().startswith("multiple")):
                _target_entity = ""
            else:
                _target_entity = _target_entity_raw.lower().replace(" ", "")

            # Fallback: LLM often omits target_entity for INSERT (no WHERE clause).
            # Use the original question — it always contains "student_id 4" etc.
            # plan["goal"] is LLM-reformulated and may not include the raw entity reference.
            if not _target_entity:
                _target_entity = _lm._extract_target_entity(
                    constraints=plan.get("constraints", []),
                    goal=question
                )

            _target_column = _lm._extract_target_column(
                constraints=plan["constraints"],
                goal=plan["goal"],
                operation=_intent_operation,
                allowed_schema=plan["allowed_schema"]
            )

            if _target_entity:
                # Check INSERT/DELETE loop BEFORE checking for active conflicts
                # This ensures escalation triggers even when no active conflict exists
                if _intent_operation == "INSERT":
                    _loop_count = _lm.count_recent_insert_delete_conflicts(
                        target_entity=_target_entity,
                        db_path=db_path,
                        minutes=_lm.CONFLICT_TIME_WINDOW_MINUTES,
                        current_conversation_id=conversation_id
                    )
                    print(f"[COORDINATOR] INSERT/DELETE conflict count for "
                          f"'{_target_entity}' in last "
                          f"{_lm.CONFLICT_TIME_WINDOW_MINUTES} minutes: {_loop_count}")

                    if _loop_count >= _lm.CONFLICT_ESCALATION_THRESHOLD:
                        print(f"[COORDINATOR] REPEATED INSERT/DELETE LOOP DETECTED — "
                              f"{_loop_count} DELETE operations on '{_target_entity}' "
                              f"in the last {_lm.CONFLICT_TIME_WINDOW_MINUTES} minutes")

                        _lm.write_escalation_flag(
                            target_entity=_target_entity,
                            target_tables=_intent_tables,
                            flagged_by_conversation_id=conversation_id
                        )

                        try:
                            _lm.clear_intent(conversation_id)
                        except Exception:
                            pass

                        outgoing = create_message(
                            sender="coordinator",
                            receiver="end",
                            msg_type="result",
                            content={
                                "question": question,
                                "result": (
                                    f"Operation rejected — repeated INSERT/DELETE "
                                    f"conflict pattern detected. {_loop_count} DELETE "
                                    f"operations on entity '{_target_entity}' in the "
                                    f"last {_lm.CONFLICT_TIME_WINDOW_MINUTES} minutes "
                                    f"suggest a possible agent loop. Escalation flag "
                                    f"written to notify the conflicting agent. "
                                    f"Human intervention may be required."
                                ),
                                "verdict": "fail",
                                "role": role
                            },
                            memory={
                                "short_term": (
                                    f"INSERT/DELETE loop escalated — {_loop_count} "
                                    f"conflicts on '{_target_entity}'"
                                ),
                                "episodic": {"coordinator_retries": 0,
                                             "current_task_id": "none"},
                                "semantic": allowed_schema
                            },
                            conversation_id=conversation_id
                        )
                        return {**state, "message": outgoing, "verdict": "fail",
                                "replan_count": state.get("replan_count", 0)}

                _conflict_found, _conflicting = _lm.check_intent_conflict(
                    conversation_id=conversation_id,
                    target_tables=_intent_tables,
                    operation=_intent_operation,
                    target_entity=_target_entity,
                    target_column=_target_column,
                    allowed_schema=plan["allowed_schema"]
                )

                if _conflict_found:
                    if (_intent_operation == "INSERT"
                            and _conflicting.get("operation") == "DELETE"):
                        _lm.log_insert_delete_conflict(
                            target_entity=_target_entity,
                            conversation_id=conversation_id
                        )

                    print(f"\n[COORDINATOR] SEMANTIC CONFLICT DETECTED — "
                          f"entering resolution phase (replan {_replan_count}/2)")
                    print(f"  This intent:        {_intent_operation} on "
                          f"'{_target_entity}' — {plan['goal']}")
                    print(f"  Conflicting intent: {_conflicting['operation']} on "
                          f"'{_conflicting['target_entity']}' — "
                          f"{_conflicting['goal']}")
                    print(f"  Conflicting PID:    {_conflicting['process_id']}")

                    if _replan_count >= 2:
                        print(f"[COORDINATOR] Re-plan limit reached (2/2) — aborting")
                        try:
                            _lm.clear_intent(conversation_id)
                        except Exception:
                            pass
                        outgoing = create_message(
                            sender="coordinator",
                            receiver="end",
                            msg_type="result",
                            content={
                                "question": question,
                                "result": (
                                    f"Operation aborted — semantic conflict could not "
                                    f"be resolved after 2 re-planning attempts. "
                                    f"Conflicting operation: "
                                    f"{_conflicting['operation']} on {_target_entity} "
                                    f"by another agent session."
                                ),
                                "verdict": "fail",
                                "role": role
                            },
                            memory={
                                "short_term": "Re-plan limit reached — semantic conflict unresolved",
                                "episodic": {"coordinator_retries": 0,
                                             "current_task_id": "none"},
                                "semantic": allowed_schema
                            },
                            conversation_id=conversation_id
                        )
                        return {**state, "message": outgoing, "verdict": "fail",
                                "replan_count": _replan_count}

                    mem.episodic_store(
                        question=question,
                        role=role,
                        plan=plan,
                        final_sql="",
                        verdict="conflict",
                        retries_used=0,
                        short_term=(
                            f"SEMANTIC_CONFLICT: {_intent_operation} on "
                            f"'{_target_entity}' conflicts with active "
                            f"{_conflicting['operation']} by "
                            f"PID {_conflicting['process_id']}"
                        ),
                        db_path=db_path,
                        schema=allowed_schema,
                        user_id=user_id if role == "user" else None
                    )

                    print(f"[COORDINATOR] Waiting for conflicting intent to complete "
                          f"(timeout=60s)...")
                    _cleared = _lm.wait_for_intent_clear(
                        _conflicting["conversation_id"], timeout=60
                    )

                    if not _cleared:
                        print(f"[COORDINATOR] Conflict resolution timed out — aborting")
                        try:
                            _lm.clear_intent(conversation_id)
                        except Exception:
                            pass
                        outgoing = create_message(
                            sender="coordinator",
                            receiver="end",
                            msg_type="result",
                            content={
                                "question": question,
                                "result": (
                                    f"Operation aborted — conflicting intent "
                                    f"({_conflicting['operation']} on "
                                    f"{_target_entity}) did not complete within "
                                    f"60 seconds."
                                ),
                                "verdict": "fail",
                                "role": role
                            },
                            memory={
                                "short_term": "Semantic conflict resolution timed out",
                                "episodic": {"coordinator_retries": 0,
                                             "current_task_id": "none"},
                                "semantic": allowed_schema
                            },
                            conversation_id=conversation_id
                        )
                        return {**state, "message": outgoing, "verdict": "fail",
                                "replan_count": _replan_count}

                    # conflict cleared — derive constraint from current DB state
                    print(f"[COORDINATOR] Conflicting intent cleared — "
                          f"reading current DB state to derive constraint...")
                    _conflict_constraint = _lm.derive_conflict_constraint(
                        db_path=db_path,
                        target_entity=_target_entity,
                        target_tables=_intent_tables,
                        operation=_intent_operation
                    )
                    print(f"[COORDINATOR] Derived constraint: "
                          f"{_conflict_constraint[:100]}")

                    try:
                        _lm.clear_intent(conversation_id)
                    except Exception:
                        pass

                    new_initial_message = create_message(
                        sender="user",
                        receiver="coordinator",
                        msg_type="instruction",
                        content={
                            "question":             question,
                            "db_path":              db_path,
                            "role":                 role,
                            "user_id":              user_id,
                            "conflict_constraints": [_conflict_constraint]
                        },
                        memory={
                            "short_term": (
                                f"RE-PLANNING after semantic conflict resolution "
                                f"(attempt {_replan_count + 1}/2) — "
                                f"original question: {question}"
                            ),
                            "episodic": {"coordinator_retries": 0,
                                         "current_task_id": "none"},
                            "semantic": ""
                        },
                        conversation_id=conversation_id
                    )
                    return {
                        **state,
                        "message":             new_initial_message,
                        "verdict":             "",
                        "coordinator_retries": 0,
                        "sql_candidates":      [],
                        "plan":                {},
                        "task_list":           [],
                        "replan_count":        0
                    }

            # no conflict or no entity extracted — check escalation flag then register intent
            if _target_entity:
                _escalated = _lm.check_escalation_flag(_target_entity, conversation_id)
                if _escalated:
                    print(f"[COORDINATOR] ESCALATION FLAG DETECTED — entity '{_target_entity}' "
                          f"has been flagged due to repeated conflict loops by another agent session.")
                    mem.episodic_store(
                        question=question,
                        role=role,
                        plan=plan,
                        final_sql="",
                        verdict="conflict",
                        retries_used=0,
                        short_term=(
                            f"ESCALATION_WARNING: operation {_intent_operation} on "
                            f"'{_target_entity}' blocked — repeated conflict pattern "
                            f"flagged by another agent session"
                        ),
                        db_path=db_path,
                        schema=allowed_schema,
                        user_id=user_id if role == "user" else None
                    )
                    try:
                        _lm.clear_intent(conversation_id)
                    except Exception:
                        pass
                    outgoing = create_message(
                        sender="coordinator",
                        receiver="end",
                        msg_type="result",
                        content={
                            "question": question,
                            "result": (
                                f"Operation blocked — entity '{_target_entity}' has been "
                                f"flagged due to a repeated INSERT/DELETE conflict loop "
                                f"detected by another agent session. Human intervention "
                                f"may be required to resolve the contention on this entity."
                            ),
                            "verdict": "fail",
                            "role": role
                        },
                        memory={
                            "short_term": f"Escalation flag blocked operation on {_target_entity}",
                            "episodic": {"coordinator_retries": 0,
                                         "current_task_id": "none"},
                            "semantic": allowed_schema
                        },
                        conversation_id=conversation_id
                    )
                    return {**state, "message": outgoing, "verdict": "fail",
                            "replan_count": state.get("replan_count", 0)}

                _lm.register_intent(
                    intent_id=conversation_id,
                    conversation_id=conversation_id,
                    target_tables=_intent_tables,
                    operation=_intent_operation,
                    target_entity=_target_entity,
                    target_column=_target_column,
                    goal=plan["goal"],
                    allowed_schema=plan["allowed_schema"]
                )

                _demo_sleep = int(os.environ.get("DEMO_CONFLICT_SLEEP", "0"))
                if _demo_sleep > 0:
                    print(f"[COORDINATOR] Demo mode — holding intent for "
                          f"{_demo_sleep}s so concurrent agents can detect it...")
                    time.sleep(_demo_sleep)

        # Build Task List in code — not LLM
        task_list = [
            {
                "task_id": "task_001",
                "agent": coder_agent,
                "goal": plan["goal"],
                "input": {
                    "plan": plan,
                    "failure_reason": None,
                    "procedural_hints": procedural_hints
                },
                "constraints": plan["constraints"],
                "allowed_schema": plan["allowed_schema"],
                "status": "in_progress",
                "result": None
            },
            {
                "task_id": "task_002",
                "agent": "verifier",
                "goal": "Verify the generated SQL passes all three checks",
                "input": {
                    "sql": None,
                    "question": question
                },
                "constraints": [
                    "Check 1 — Structural: all tables and columns exist in allowed_schema",
                    "Check 2 — Validity: result is plausible against DB statistics",
                    "Check 3 — Consistency: SQL correctly answers the original question"
                ],
                "allowed_schema": plan["allowed_schema"],
                "status": "pending",
                "result": None
            }
        ]

        outgoing = create_message(
            sender="coordinator",
            receiver=coder_agent,
            msg_type="instruction",
            content={
                "question": question,
                "db_path":  db_path,
                "role":     role,
                "plan":     plan,
                "task":     task_list[0]
            },
            memory={
                "short_term": (
                    f"Question: {question}\n"
                    f"Role: {role}\n"
                    f"Goal: {plan['goal']}\n"
                    f"Operation: {plan['description']['operation']}\n"
                    f"Complexity: {plan['description'].get('complexity', 'simple')}\n"
                    f"task_001 dispatched to {coder_agent}"
                ),
                "episodic": {"coordinator_retries": 0, "current_task_id": "task_001"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )

        return {
            **state,
            "message":        outgoing,
            "plan":           plan,
            "task_list":      task_list,
            "sql_candidates": [],
            "replan_count":   state.get("replan_count", 0)
        }

    # ── Receiving from coder: forward SQL to verifier ────────────────────────
    if incoming["sender"] in ("user_coder_agent", "admin_coder_agent"):
        # OPERATION_NOT_PERMITTED — coder rejected the request outright
        if incoming["content"].get("verdict") == "fail":
            reason         = incoming["content"].get("reason", "Operation not permitted")
            role           = incoming["content"].get("role", "user")
            question       = incoming["content"].get("question", state["plan"]["initial_state"]["question"])
            allowed_schema = state["plan"]["allowed_schema"]
            print(f"[COORDINATOR] Coder rejected operation: {reason}")
            outgoing = create_message(
                sender="coordinator",
                receiver="end",
                msg_type="result",
                content={"question": question, "result": reason, "verdict": "fail", "role": role},
                memory={
                    "short_term": "",
                    "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_001"},
                    "semantic": allowed_schema
                },
                conversation_id=conversation_id
            )
            return {**state, "message": outgoing, "verdict": "fail"}

        # Normal case — dispatch task_002 to verifier
        question      = incoming["content"]["question"]
        db_path       = incoming["content"]["db_path"]
        role          = incoming["content"].get("role", "user")
        chosen_sql          = incoming["content"]["sql"]
        chosen_reasoning    = incoming["content"].get("reasoning", "")
        fallback_sqls       = incoming["content"].get("fallbacks", [])
        fallback_reasonings = incoming["content"].get("fallback_reasonings", [])
        plan          = state["plan"]
        allowed_schema = plan["allowed_schema"]
        task_list     = list(state["task_list"])

        # Coordinator owns task list — update it here, not in the Coder
        task_list[0]["result"] = chosen_sql
        task_list[0]["status"] = "completed"
        task_list[1]["input"]["sql"]       = chosen_sql
        task_list[1]["input"]["reasoning"] = chosen_reasoning
        task_list[1]["status"] = "in_progress"

        if state.get("no_verify", False):
            print("[COORDINATOR] no_verify mode — skipping verifier, executing SQL directly")
            execution_result = run_sql(db_path, chosen_sql)
            final_verdict = "pass" if not execution_result.startswith("ERROR") else "fail"
            outgoing = create_message(
                sender="coordinator", receiver="end",
                msg_type="result",
                content={"question": question, "result": execution_result, "verdict": final_verdict, "role": role},
                memory={"short_term": "", "episodic": {"coordinator_retries": 0, "current_task_id": "task_001"}, "semantic": allowed_schema},
                conversation_id=conversation_id
            )
            return {**state, "message": outgoing, "verdict": final_verdict}

        print(f"[COORDINATOR] SQL received from coder — dispatching to verifier...")
        outgoing = create_message(
            sender="coordinator",
            receiver="verifier",
            msg_type="instruction",
            content={
                "question":    question,
                "db_path":     db_path,
                "role":        role,
                "task":        task_list[1],
                "row_filters": plan.get("row_filters", {})
            },
            memory={
                "short_term": incoming["memory"].get("short_term", ""),
                "episodic":   {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic":   allowed_schema
            },
            conversation_id=conversation_id
        )
        # Pair each fallback SQL with its own reasoning as (sql, reasoning) tuples
        fallback_candidates = list(zip(fallback_sqls, fallback_reasonings))

        return {
            **state,
            "message":        outgoing,
            "task_list":      task_list,
            "sql_candidates": fallback_candidates   # list of (sql, reasoning) tuples
        }

    # ── Receiving from verifier: execute or retry ─────────────────────────────
    role          = incoming["content"].get("role", "user")
    plan          = state["plan"]
    allowed_schema = plan["allowed_schema"]
    task_list     = list(state["task_list"])
    verdict  = incoming["content"].get("verdict", "fail")
    sql      = incoming["content"].get("sql", "")
    reason   = incoming["content"].get("reason", "")
    question = incoming["content"].get("question", plan["initial_state"]["question"])
    db_path  = incoming["content"].get("db_path",  plan["initial_state"]["db_path"])

    task_list[1]["result"] = {"verdict": verdict, "reason": reason, "sql": sql}
    task_list[1]["status"] = "completed" if verdict == "pass" else "failed"

    if verdict == "pass":
        print("[COORDINATOR] Verifier passed — executing SQL...")
        operation = get_operation_type(sql)

        # Take DB snapshot before write for post-execution comparison
        snapshot_before = (
            take_db_snapshot(db_path, allowed_schema)
            if operation in ("INSERT", "UPDATE", "DELETE")
            else {}
        )

        # ── Acquire table-level lock, execute, then release ───────────────────
        import re as _re
        tables_in_sql = set(_re.findall(
            r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', sql, _re.IGNORECASE
        ))

        if operation in ("INSERT", "UPDATE", "DELETE"):
            _row_key = _lm.extract_row_key(sql) if operation in ("UPDATE", "DELETE") else ""
            if _row_key:
                print(f"[COORDINATOR] Row-level lock key: '{_row_key}'")
            else:
                print(f"[COORDINATOR] Table-level lock (no specific row identified)")

            held, held_state = _lm.is_held(list(tables_in_sql), row_key=_row_key)
            if held:
                conflicting = held_state.get("locked_tables", [])
                print(f"[COORDINATOR] CONFLICT DETECTED — table(s) {conflicting} locked by "
                      f"'{held_state.get('held_by')}' (PID {held_state.get('pid')})")

                mem.episodic_store(
                    question=question,
                    role=role,
                    plan=plan,
                    final_sql=sql,
                    verdict="conflict",
                    retries_used=coordinator_retries,
                    short_term=f"WRITE_CONFLICT: tables {conflicting} locked by another agent session",
                    db_path=db_path,
                    schema=allowed_schema,
                    user_id=plan.get("initial_state", {}).get("user_id", None) if role == "user" else None
                )

                for t in conflicting:
                    queue = _lm.get_queue_state(t)
                    print(f"[LOCK] Queue for table '{t}':")
                    for entry in queue:
                        print(f"  status={entry['status']}  position={entry['queue_position']}  "
                              f"pid={entry['process_id']}  row_key={entry.get('row_key', '')}  "
                              f"held_by={entry['held_by']}")

                released = _lm.wait_for_release(
                    list(tables_in_sql), timeout=30, row_key=_row_key
                )
                if not released:
                    print(f"[COORDINATOR] Lock wait timed out after 30s — aborting")
                    try:
                        _lm.clear_intent(conversation_id)
                    except Exception:
                        pass
                    outgoing = create_message(
                        sender="coordinator",
                        receiver="end",
                        msg_type="result",
                        content={
                            "question": question,
                            "result": f"Operation aborted — lock on {conflicting} "
                                      f"not released within 30 seconds.",
                            "verdict": "fail",
                            "role": role
                        },
                        memory={
                            "short_term": "Lock timeout — task aborted",
                            "episodic": {"coordinator_retries": coordinator_retries,
                                         "current_task_id": "task_002"},
                            "semantic": allowed_schema
                        },
                        conversation_id=conversation_id
                    )
                    return {**state, "message": outgoing, "verdict": "fail",
                            "replan_count": state.get("replan_count", 0)}
                print(f"[COORDINATOR] Lock released and it is my turn — proceeding")

            acquired = _lm.acquire(
                list(tables_in_sql),
                held_by=f"coordinator_pid_{os.getpid()}",
                row_key=_row_key
            )
            if not acquired:
                print(f"[COORDINATOR] Could not acquire lock — retrying once")
                time.sleep(0.5)
                acquired = _lm.acquire(
                    list(tables_in_sql),
                    held_by=f"coordinator_pid_{os.getpid()}",
                    row_key=_row_key
                )
            if not acquired:
                try:
                    _lm.clear_intent(conversation_id)
                except Exception:
                    pass
                outgoing = create_message(
                    sender="coordinator",
                    receiver="end",
                    msg_type="result",
                    content={
                        "question": question,
                        "result": "Operation aborted — could not acquire write lock.",
                        "verdict": "fail",
                        "role": role
                    },
                    memory={
                        "short_term": "Lock acquisition failed",
                        "episodic": {"coordinator_retries": coordinator_retries,
                                     "current_task_id": "task_002"},
                        "semantic": allowed_schema
                    },
                    conversation_id=conversation_id
                )
                return {**state, "message": outgoing, "verdict": "fail",
                        "replan_count": state.get("replan_count", 0)}

            heartbeat_stop = _lm.start_heartbeat(list(tables_in_sql))
            print(f"[COORDINATOR] Lock ACQUIRED on {set(tables_in_sql)} row_key='{_row_key}'")

            _lock_demo_sleep = int(os.environ.get("DEMO_LOCK_SLEEP", "0"))
            if _lock_demo_sleep > 0:
                print(f"[COORDINATOR] Demo mode — holding LOCK for "
                      f"{_lock_demo_sleep}s so concurrent agents can detect it...")
                time.sleep(_lock_demo_sleep)

        execution_result = run_sql(db_path, sql)
        print(f"[COORDINATOR] Result: {str(execution_result)[:200]}")

        # Post-execution anomaly check for write operations only
        if operation in ("INSERT", "UPDATE", "DELETE") and not execution_result.startswith("ERROR"):
            post_type, post_desc = post_execution_check(sql, db_path, allowed_schema, snapshot_before)
            if post_type:
                print(f"[COORDINATOR] Post-execution FAILED: {post_type} — {post_desc}")
                execution_result = f"ERROR: Post-execution anomaly — {post_desc}"

        if operation in ("INSERT", "UPDATE", "DELETE"):
            _lm.stop_heartbeat(heartbeat_stop)
            _lm.release(list(tables_in_sql), row_key=_row_key)

        # Determine final verdict from actual execution result
        final_verdict = "pass" if not execution_result.startswith("ERROR") else "fail"

        final_short_term = incoming["memory"].get("short_term", "")
        final_short_term += f"\nSQL executed: {execution_result[:100]}"

        # Store episode with actual verdict — not hardcoded pass
        mem.episodic_store(
            question=question,
            role=role,
            plan=plan,
            final_sql=sql,
            verdict=final_verdict,
            retries_used=coordinator_retries,
            short_term=final_short_term,
            db_path=db_path,
            schema=allowed_schema,
            user_id=plan.get("initial_state", {}).get("user_id", None) if role == "user" else None
        )

        # Only store procedural memory on genuine success
        if final_verdict == "pass" and not state.get("benchmark_mode", False):
            mem.procedural_store(
                question=question,
                sql=sql,
                db_path=db_path,
                schema=allowed_schema,
                role=role,
                user_id=plan.get("initial_state", {}).get("user_id", None) if role == "user" else None
            )

        outgoing = create_message(
            sender="coordinator", receiver="end",
            msg_type="result",
            content={
                "question": question,
                "result":   execution_result,
                "verdict":  final_verdict,
                "role":     role
            },
            memory={
                "short_term": final_short_term,
                "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        try:
            _lm.clear_intent(conversation_id)
        except Exception:
            pass
        return {**state, "message": outgoing, "verdict": final_verdict, "task_list": task_list}

    # Integrity constraint violations are not retryable — abort immediately
    if reason.startswith("CONSTRAINT:") or reason.startswith("CIRCULAR:"):
        print(f"[COORDINATOR] Integrity constraint violation — aborting immediately: {reason}")
        mem.episodic_store(
            question=question,
            role=role,
            plan=plan,
            final_sql=sql,
            verdict="fail",
            retries_used=0,
            short_term=f"Integrity constraint violation: {reason}",
            db_path=db_path,
            schema=allowed_schema,
            user_id=plan.get("initial_state", {}).get("user_id", None) if role == "user" else None
        )
        outgoing = create_message(
            sender="coordinator",
            receiver="end",
            msg_type="result",
            content={
                "question": question,
                "result": f"Operation rejected: {reason}",
                "verdict": "fail",
                "role": role
            },
            memory={
                "short_term": f"Integrity constraint violation: {reason}",
                "episodic": {"coordinator_retries": 0, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        try:
            _lm.clear_intent(conversation_id)
        except Exception:
            pass
        return {**state, "message": outgoing, "verdict": "fail"}

    # try remaining fallback candidates before triggering a full retry
    sql_candidates = list(state.get("sql_candidates", []))
    if sql_candidates:
        next_sql, next_reasoning = sql_candidates[0]   # unpack (sql, reasoning) tuple
        remaining = sql_candidates[1:]
        print(f"[COORDINATOR] Verifier failed — trying next fallback candidate ({len(remaining)} remaining after this)...")

        task_list[1]["input"]["sql"]       = next_sql
        task_list[1]["input"]["reasoning"] = next_reasoning   # fallback's own reasoning
        task_list[1]["status"] = "in_progress"
        task_list[1]["result"] = None

        outgoing = create_message(
            sender="coordinator",
            receiver="verifier",
            msg_type="instruction",
            content={
                "question":    question,
                "db_path":     db_path,
                "role":        role,
                "task":        task_list[1],
                "row_filters": plan.get("row_filters", {})
            },
            memory={
                "short_term": f"Question: {question}\nTrying fallback candidate",
                "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        return {
            **state,
            "message":        outgoing,
            "verdict":        "fail",
            "task_list":      task_list,
            "sql_candidates": remaining
        }

    # No fallbacks left — check for deadlock before retrying
    if "database is locked" in reason.lower() or "deadlock" in reason.lower():
        print(f"[COORDINATOR] DEADLOCK detected — lock_manager will handle retry via wait_for_release")
        outgoing = create_message(
            sender="coordinator",
            receiver="end",
            msg_type="result",
            content={
                "question": question,
                "result": "Operation queued — database is locked by another agent. Will retry when lock is released.",
                "verdict": "queued",
                "role": role
            },
            memory={
                "short_term": "DEADLOCK detected — task queued",
                "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        return {**state, "message": outgoing, "verdict": "queued"}

    # No fallbacks left — retry from Coder with failure reason
    if coordinator_retries < 3:
        print(f"[COORDINATOR] Verifier failed (attempt {coordinator_retries + 1}/3): {reason}")

        task_list[0]["status"] = "in_progress"
        task_list[0]["input"]["failure_reason"] = reason[:300]
        task_list[0]["input"]["failed_sql"] = sql
        task_list[1]["input"]["sql"] = None
        task_list[1]["status"] = "pending"
        task_list[1]["result"] = None

        coder_agent = "user_coder" if role == "user" else "admin_coder"
        short_term = incoming["memory"].get("short_term", "")
        short_term += (
            f"\nCoordinator retry {coordinator_retries + 1}/3 — "
            f"reason: {reason}"
        )
        outgoing = create_message(
            sender="coordinator",
            receiver=coder_agent,
            msg_type="instruction",
            content={
                "question": question,
                "db_path":  db_path,
                "role":     role,
                "plan":     plan,
                "task":     task_list[0]
            },
            memory={
                "short_term": short_term,
                "episodic": {"coordinator_retries": coordinator_retries + 1, "current_task_id": "task_001"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        return {
            **state,
            "message":             outgoing,
            "verdict":             "fail",
            "coordinator_retries": coordinator_retries + 1,
            "task_list":           task_list,
            "sql_candidates":      []
        }

    else:
        print("[COORDINATOR] Max retries reached — attempting fallback execution of last SQL...")
        final_short_term = incoming["memory"].get("short_term", "")

        # Attempt to execute the last SQL anyway — but only if:
        # 1. It is a SELECT operation (never execute writes as fallback)
        # 2. The failure came from Check 3 (LLM judge) not Check 1 or Check 2
        fallback_result = None
        final_verdict = "fail"
        if (sql
                and not sql.startswith("ERROR")
                and get_operation_type(sql) == "SELECT"
                and reason.startswith("Consistency:")):
            try:
                fallback_result = run_sql(db_path, sql)
                if fallback_result and not fallback_result.startswith("ERROR"):
                    print(f"[COORDINATOR] Fallback execution succeeded: {fallback_result[:100]}")
                    final_verdict = "pass"
                    final_short_term += f"\nFallback execution succeeded after max retries: {fallback_result[:100]}"
                else:
                    print(f"[COORDINATOR] Fallback execution failed: {fallback_result}")
                    final_short_term += f"\nSession ended: max retries exhausted — {reason}"
            except Exception as e:
                print(f"[COORDINATOR] Fallback execution error: {e}")
                final_short_term += f"\nSession ended: max retries exhausted — {reason}"
        else:
            final_short_term += f"\nSession ended: max retries exhausted — {reason}"

        result_to_return = (
            fallback_result
            if final_verdict == "pass"
            else f"Could not generate valid SQL after {coordinator_retries} attempts: {reason}"
        )

        mem.episodic_store(
            question=question,
            role=role,
            plan=plan,
            final_sql=sql,
            verdict=final_verdict,
            retries_used=coordinator_retries,
            short_term=final_short_term,
            db_path=db_path,
            schema=allowed_schema,
            user_id=plan.get("initial_state", {}).get("user_id", None) if role == "user" else None
        )

        outgoing = create_message(
            sender="coordinator", receiver="end",
            msg_type="result",
            content={
                "question": question,
                "result":   result_to_return,
                "verdict":  final_verdict,
                "role":     role
            },
            memory={
                "short_term": final_short_term,
                "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        try:
            _lm.clear_intent(conversation_id)
        except Exception:
            pass
        return {**state, "message": outgoing, "verdict": final_verdict, "task_list": task_list}


# ─── Shared Coder Logic ───────────────────────────────────────────────────────

def _coder_node(state: QueryState, system_prompt: str, agent_name: str) -> QueryState:
    """Shared coder logic used by both user_coder and admin_coder."""
    print(f"\n[{agent_name.upper()}] Writing SQL...")

    incoming = state["message"]
    conversation_id = incoming["conversation_id"]
    question = incoming["content"]["question"]
    db_path = incoming["content"]["db_path"]
    plan = incoming["content"]["plan"]
    task = incoming["content"]["task"]
    task_goal = task["goal"]
    failure_reason = task["input"].get("failure_reason")
    allowed_schema = task["allowed_schema"]
    coordinator_retries = state["coordinator_retries"]

    failure_context = ""
    failed_sql = task["input"].get("failed_sql", "")
    if failure_reason:
        failure_context = (
            f"\n\nPrevious attempt failed."
            f"\nFailed SQL: {failed_sql}\n"
            f"Failure reason: {failure_reason}\n"
            f"Identify specifically what was wrong in the failed SQL and correct only that part.\n"
            f"If the failure mentions a wrong table or column, check the schema carefully.\n"
            f"If the failure mentions wrong results, reconsider the query logic — "
            f"especially whether a subquery, JOIN, or aggregation is needed."
        )
    

    procedural_hints = task["input"].get("procedural_hints", [])
    hints_context = ""
    if procedural_hints:
        hints_context = (
            "\n\nSimilar questions your system answered correctly before "
            "(use as reference only — do not copy blindly):\n"
        )
        for h in procedural_hints:
            hints_context += (
                f"  Past question (similarity={h['similarity']}): {h['question']}\n"
                f"  SQL that worked: {h['sql']}\n\n"
            )

    # Separate security constraints from regular constraints
    security_constraints = [c for c in plan["constraints"] if "SECURITY REQUIREMENT" in c]
    regular_constraints = [c for c in plan["constraints"] if "SECURITY REQUIREMENT" not in c]

    security_block = ""
    if security_constraints:
        security_block = (
            "CRITICAL SECURITY REQUIREMENTS — YOU MUST FOLLOW THESE EXACTLY:\n" +
            "\n".join(f"  - {c}" for c in security_constraints) +
            "\n\nFailure to include these conditions will cause the query to be rejected.\n\n"
        )

    no_schema_link = state.get("no_schema_link", False)

    if no_schema_link:
        prompt_content = (
            f"\nQuestion: {question}\n"
            f"Schema: {allowed_schema}\n"
            f"{failure_context}"
            f"\n\nWrite the SQL:"
        )
    else:
        prompt_content = (
            f"{security_block}"
            f"\nQuestion: {question}\n"
            f"{failure_context}\n"
            f"Task goal: {task_goal}\n"
            f"Schema: {allowed_schema}\n"
            f"Plan:\n"
            f"  Operation: {plan['description']['operation']}\n"
            f"  Complexity: {plan['description'].get('complexity', 'simple')}\n"
            f"  Logical Steps: {plan['description'].get('logical_steps', '')}\n"
            + (f"  Values: {plan['description']['values']}\n" if plan['description'].get('values') else "")
            + f"Constraints: {'; '.join(regular_constraints)}"
            f"{hints_context}"
            f"\n\nWrite the SQL:"
        )

    coder_prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt_content)
    ]

    # Check for CONFLICT RESOLUTION constraint requiring immediate abort
    # This fires when the entity was deleted by a concurrent agent session
    _conflict_abort_constraints = [
        c for c in plan["constraints"]
        if c.startswith("CONFLICT RESOLUTION:")
        and "OPERATION_NOT_PERMITTED" in c
        and plan["description"].get("operation", "") != "INSERT"
    ]
    if _conflict_abort_constraints:
        print(f"[{agent_name.upper()}] CONFLICT RESOLUTION constraint — "
              f"entity no longer exists, aborting without LLM call")
        outgoing = create_message(
            sender=f"{agent_name}_agent",
            receiver="coordinator",
            msg_type="result",
            content={
                "question": question,
                "db_path":  db_path,
                "role":     incoming["content"].get("role", "user"),
                "verdict":  "fail",
                "reason":   _conflict_abort_constraints[0],
                "sql":      "OPERATION_NOT_PERMITTED"
            },
            memory={
                "short_term": f"Question: {question}",
                "episodic": {"coordinator_retries": coordinator_retries,
                             "current_task_id": "task_001"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        return {**state, "message": outgoing, "verdict": "fail"}

    is_first_attempt = coordinator_retries == 0
    no_verify = state.get("no_verify", False)
    num_candidates = 1 if no_verify else (3 if is_first_attempt else 1)

    candidates = []
    errors = []

    for i in range(num_candidates):
        response = invoke_coder_llm(coder_prompt)

        if response is None:
            print(f"[{agent_name.upper()}] Candidate {i+1}/{num_candidates}: rate limit exhausted — skipping")
            errors.append(("", "ERROR: rate limit exhausted after 5 retries", ""))
            continue

        raw = get_content(response).strip()
        if "<think>" in raw:
            parts = raw.split("</think>")
            raw = parts[-1].strip() if len(parts) > 1 else ""
        raw_cleaned = raw.replace("```sql", "").replace("```", "").strip()

        # Guard: if nothing left after stripping think block, treat as error
        if not raw_cleaned or len(raw_cleaned) < 10:
            errors.append(("", "ERROR: model produced only thinking block with no SQL", ""))
            continue

        # Parse sql and reasoning — each candidate has its own reasoning
        reasoning = ""
        if "REASONING:" in raw_cleaned:
            parts     = raw_cleaned.split("REASONING:", 1)
            sql       = parts[0].strip()
            reasoning = parts[1].strip().split("\n")[0].strip()
        else:
            sql       = raw_cleaned
            reasoning = ""
    # Reject trivial or degenerate SQL outputs
        if len(sql.split()) < 4 or sql.strip().upper() in ("SELECT 1", "SELECT 1;"):
            print(f"[{agent_name.upper()}] Candidate {i+1}: trivial SQL rejected — {sql[:40]}")
            errors.append((sql, "ERROR: trivial SQL output rejected", ""))
            continue
        # Check for operation permission before anything else
        if sql.strip().upper().startswith("OPERATION_NOT_PERMITTED"):
            print(f"[{agent_name.upper()}] Operation not permitted for this role.")
            outgoing = create_message(
                sender=f"{agent_name}_agent",
                receiver="coordinator",
                msg_type="result",
                content={
                    "question": question,
                    "db_path":  db_path,
                    "role":     incoming["content"].get("role", "user"),
                    "verdict":  "fail",
                    "reason":   "OPERATION_NOT_PERMITTED: This operation is not allowed for your role.",
                    "sql":      sql
                },
                memory={
                    "short_term": f"Question: {question}",
                    "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_001"},
                    "semantic": allowed_schema
                },
                conversation_id=conversation_id
            )
            return {**state, "message": outgoing, "verdict": "fail"}

        candidate_operation = get_operation_type(sql)
        if candidate_operation in ("INSERT", "UPDATE", "DELETE", "TRANSACTION_BLOCK"):
            print(f"[{agent_name.upper()}] Candidate {i+1}/{num_candidates}: {sql[:80]}...")
            print(f"[{agent_name.upper()}] Candidate {i+1}: write operation — deferred to verifier")
            candidates.append((sql, "PENDING_VERIFICATION", reasoning))
        else:
            result = run_sql(db_path, sql)
            print(f"[{agent_name.upper()}] Candidate {i+1}/{num_candidates}: {sql[:80]}...")
            print(f"[{agent_name.upper()}] Candidate {i+1} result: {result[:80]}")
            if result.startswith("ERROR:"):
                errors.append((sql, result, reasoning))
            else:
                candidates.append((sql, result, reasoning))

    # Self-consistency: majority vote on execution results only — reasoning does not affect the vote
    if len(candidates) >= 2:
        from collections import Counter
        result_counts = Counter(r for _, r, _ in candidates)
        majority_result, majority_count = result_counts.most_common(1)[0]
        if majority_count >= 2:
            majority = [(s, r, rsn) for s, r, rsn in candidates if r == majority_result]
            minority = [(s, r, rsn) for s, r, rsn in candidates if r != majority_result]
            print(f"[{agent_name.upper()}] Majority agreement ({majority_count}/3): {majority_result[:80]}")
            all_ordered = majority + minority + errors
        else:
            all_ordered = candidates + errors
    else:
        all_ordered = candidates + errors

    
    chosen_sql       = all_ordered[0][0] if all_ordered else ""
    chosen_reasoning = all_ordered[0][2] if all_ordered else ""

    if not chosen_sql or chosen_sql.strip().upper() == "SELECT 1":
        if len(all_ordered) > 1:
            chosen_sql       = all_ordered[1][0]
            chosen_reasoning = all_ordered[1][2]
            remaining_candidates = [(s, rsn) for s, _, rsn in all_ordered[2:]]
        else:
            remaining_candidates = []
    else:
        remaining_candidates = [(s, rsn) for s, _, rsn in all_ordered[1:]]
    remaining_sqls       = [s for s, _ in remaining_candidates]

    print(f"[{agent_name.upper()}] Chosen SQL: {chosen_sql}")
    print(f"[{agent_name.upper()}] Chosen reasoning: {chosen_reasoning}")
    print(f"[{agent_name.upper()}] {len(remaining_candidates)} fallback candidate(s) queued")

    coder_short_term = (
        incoming["memory"].get("short_term", "") +
        f"\nCoder generated SQL: {chosen_sql[:120]}"
        + (f"\n{len(remaining_sqls)} fallback candidate(s) queued" if remaining_sqls else "")
    )
    outgoing = create_message(
        sender=f"{agent_name}_agent",
        receiver="coordinator",
        msg_type="result",
        content={
            "question":            question,
            "db_path":             db_path,
            "role":                incoming["content"].get("role", "user"),
            "sql":                 chosen_sql,
            "reasoning":           chosen_reasoning,
            "fallbacks":           remaining_sqls,
            "fallback_reasonings": [rsn for _, rsn in remaining_candidates]
        },
        memory={
            "short_term": coder_short_term,
            "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_001"},
            "semantic": allowed_schema
        },
        conversation_id=conversation_id
    )

    return {
        **state,
        "message":        outgoing,
        "sql_candidates": remaining_candidates   # list of (sql, reasoning) tuples
    }

def user_coder(state: QueryState) -> QueryState:
    system_prompt = (
        "You are an expert SQLite SQL generator.\n"
        "Generate ONLY valid SQLite SQL.\n"
        "Return ONLY the SQL query followed by one REASONING line.\n"
        "Do not use markdown, comments, or explanations.\n"
        "Table names and column names MUST exactly match the schema.\n"

        "SQLITE RULES:\n"
        "- Use SQLite syntax only\n"
        "- NEVER use EXTRACT, ILIKE, DATE_TRUNC, RIGHT JOIN, or FULL OUTER JOIN\n"
        "- Use LIKE instead of ILIKE\n"
        "- Use substr(date_column,1,4) for year extraction\n"
        "- Use CAST(... AS REAL) for decimal division\n"

        "SET OPERATION RULES:\n"
        "- When a question asks for items satisfying two separate conditions on different rows of the same column, consider using INTERSECT\n"
        "- When a question asks for items not in another set, consider using EXCEPT\n"

        "SEMANTIC RULES:\n"
        "- Match literal values exactly as stored in the database\n"
        "- When uncertain about string case, use LOWER() for comparison\n"

        "OUTPUT FORMAT:\n"
        "SQL query ending with semicolon\n"
        "REASONING: brief explanation of the query logic\n"
    )
    return _coder_node(state, system_prompt, "user_coder")


def admin_coder(state: QueryState) -> QueryState:
    system_prompt = (
        "You are an expert SQLite SQL generator.\n"
        "Generate ONLY valid SQLite SQL.\n"
        "Return ONLY the SQL query followed by one REASONING line.\n"
        "Do not use markdown, comments, or explanations.\n"
        "Table names and column names MUST exactly match the schema.\n"

        "SQLITE RULES:\n"
        "- Use SQLite syntax only\n"
        "- NEVER use EXTRACT, ILIKE, DATE_TRUNC, RIGHT JOIN, or FULL OUTER JOIN\n"
        "- Use LIKE instead of ILIKE\n"
        "- Use substr(date_column,1,4) for year extraction\n"
        "- Use CAST(... AS REAL) for decimal division\n"

        "SET OPERATION RULES:\n"
        "- When a question asks for items satisfying two separate conditions on different rows of the same column, consider using INTERSECT\n"
        "- When a question asks for items not in another set, consider using EXCEPT\n"

        "SELECT RULES:\n"
        "- Return exactly the information requested by the question\n"
        "- Use subqueries only when needed\n"

        "WRITE RULES:\n"
        "- Always include a WHERE clause for UPDATE and DELETE unless explicitly updating or deleting all rows\n"

        "SEMANTIC RULES:\n"
        "- Match literal values exactly as stored in the database\n"
        "- When uncertain about string case, use LOWER() for comparison\n"

        "OUTPUT FORMAT:\n"
        "SQL query ending with semicolon\n"
        "REASONING: brief explanation of the query logic\n"
    )
    return _coder_node(state, system_prompt, "admin_coder")

# ─── Verifier ─────────────────────────────────────────────────────────────────

def verifier(state: QueryState) -> QueryState:
    print("\n[VERIFIER] Running verification checks...")

    incoming = state["message"]
    conversation_id = incoming["conversation_id"]
    db_path        = incoming["content"]["db_path"]
    task           = incoming["content"]["task"]
    sql       = task["input"]["sql"]
    question  = task["input"]["question"]
    reasoning = task["input"].get("reasoning", "")
    allowed_schema = task["allowed_schema"]
    coordinator_retries = state["coordinator_retries"]
    row_filters    = incoming["content"].get("row_filters", {})
    benchmark_mode = state.get("benchmark_mode", False)

    def send_fail(reason: str):
        task["result"] = {"verdict": "fail", "reason": reason, "sql": sql}
        short_term_updated = (
            incoming["memory"].get("short_term", "") +
            f"\nVerifier FAIL: {reason}"
        )
        outgoing = create_message(
            sender="verifier_agent",
            receiver="coordinator",
            msg_type="result",
            content={
                "question": question,
                "db_path":  db_path,
                "role":     incoming["content"].get("role", "user"),
                "verdict":  "fail",
                "reason":   reason,
                "sql":      sql
            },
            memory={
                "short_term": short_term_updated,
                "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        return {**state, "message": outgoing, "verdict": "fail"}

    def send_pass():
        task["result"] = {"verdict": "pass", "reason": "All checks passed", "sql": sql}
        short_term_updated = (
            incoming["memory"].get("short_term", "") +
            f"\nVerifier PASS: all three checks passed — SQL: {sql[:80]}"
        )
        outgoing = create_message(
            sender="verifier_agent",
            receiver="coordinator",
            msg_type="result",
            content={
                "question": question,
                "db_path":  db_path,
                "role":     incoming["content"].get("role", "user"),
                "verdict":  "pass",
                "reason":   "All checks passed",
                "sql":      sql
            },
            memory={
                "short_term": short_term_updated,
                "episodic": {"coordinator_retries": coordinator_retries, "current_task_id": "task_002"},
                "semantic": allowed_schema
            },
            conversation_id=conversation_id
        )
        return {**state, "message": outgoing, "verdict": "pass"}

   # ── Check 1: Structural (rule-based, no LLM) ──────────────────────────────
    print("[VERIFIER] Running Check 1 — Structural...")
    tables_in_sql = re.findall(r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', sql, re.IGNORECASE)
    for table in tables_in_sql:
        if table.upper() not in allowed_schema.upper():
            print(f"[VERIFIER] Check 1 FAILED: table {table} not in schema")
            return send_fail(f"Structural: table '{table}' does not exist in allowed schema")

    select_match = re.search(r'SELECT\s+(.*?)\s+FROM', sql, re.IGNORECASE | re.DOTALL)
    if select_match:
        select_clause = select_match.group(1).strip()
        if select_clause.strip() != '*':
            for col_part in select_clause.split(','):
                col_part = col_part.strip()
                func_match = re.search(r'\w+\s*\(\s*([^)]*)\s*\)', col_part)
                if func_match:
                    # Skip entire column check for complex function calls
                    # with multiple arguments or string literals (e.g. strftime)
                    inner = func_match.group(1).strip()
                    if not inner or inner == '*':
                        continue
                    # If function has multiple arguments, skip validation entirely
                    if ',' in inner:
                        continue
                    # If argument is a string literal, skip
                    if inner.startswith("'") or inner.startswith('"'):
                        continue
                    col_name = inner
                    if '.' in col_name:
                        col_name = col_name.split('.')[-1]
                else:
                    # Strip table alias prefix first e.g. s.Age -> Age, T1.name -> name
                    raw = col_part.strip()
                    if '.' in raw:
                        raw = raw.split('.')[-1]
                    # Then strip AS alias e.g. name AS student_name -> name
                    col_name = re.split(r'\s+AS\s+|\s+', raw, flags=re.IGNORECASE)[0]
                col_name = col_name.strip().strip("'\"")
                if (col_name
                        and col_name != '*'
                        and not col_name.isdigit()
                        and col_name.upper() not in ('DISTINCT', 'NULL', 'COUNT', 'SUM', 'AVG', 'MIN', 'MAX')):
                    if col_name.upper() not in allowed_schema.upper():
                        print(f"[VERIFIER] Check 1 FAILED: column {col_name} not in schema")
                        return send_fail(f"Structural: column '{col_name}' does not exist in allowed schema")

    print("[VERIFIER] Check 1 (Structural) passed.")
    # ── Check 2: Validity (rule-based, uses DB statistics) ───────────────────
    # FIX 2: implemented real validity checks using DB statistics
    print("[VERIFIER] Running Check 2 — Validity...")
    stats = get_db_statistics(db_path, allowed_schema)

    # Parse total row counts from statistics string
    # Statistics format expected: "table_name: N rows" or similar
    # We use the tables in SQL and check plausibility
    try:
        # Check 2a: if SQL has a LIMIT clause, the limit should be positive
        limit_match = re.search(r'\bLIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            limit_val = int(limit_match.group(1))
            if limit_val <= 0:
                print("[VERIFIER] Check 2 FAILED: LIMIT value is not positive")
                return send_fail("Validity: LIMIT value must be greater than zero")

        # Check 2b: if SQL filters on a specific integer value in WHERE,
        # verify it is a positive integer (basic sanity check)
        where_match = re.search(r'\bWHERE\b(.*?)(?:\bGROUP\b|\bORDER\b|\bLIMIT\b|\bHAVING\b|$)', sql, re.IGNORECASE | re.DOTALL)
        if where_match:
            where_clause = where_match.group(1)
            # Check for obviously invalid negative IDs
            negative_id = re.search(r'\bid\s*=\s*(-\d+)', where_clause, re.IGNORECASE)
            if negative_id:
                print(f"[VERIFIER] Check 2 FAILED: negative ID in WHERE clause")
                return send_fail(f"Validity: negative ID value {negative_id.group(1)} is not plausible")

        # Check 2c: for each table in the SQL, verify it appears in statistics
        # (meaning it actually has data context — not a hard fail, just a warning logged)
        for table in tables_in_sql:
            if table.lower() not in stats.lower():
                print(f"[VERIFIER] Check 2 note: table {table} not found in statistics (may be empty)")

        # Check 2d: row-level security enforcement
        if row_filters and not benchmark_mode:
            for table, condition in row_filters.items():
                table_lower = table.lower()
                tables_lower = [t.lower() for t in tables_in_sql]
                if table_lower in tables_lower:
                    parts = condition.split("=")
                    if len(parts) == 2:
                        filter_col = parts[0].strip()
                        filter_val = parts[1].strip()
                        pattern = rf'{filter_col}\s*=\s*{filter_val}'
                        if not re.search(pattern, sql, re.IGNORECASE):
                            print(f"[VERIFIER] Check 2 FAILED: row-level filter missing — {condition} not found in SQL")
                            return send_fail(
                                f"Validity: row-level security violation — SQL accesses table "
                                f"'{table}' but is missing required filter: WHERE {condition}"
                            )

    except Exception as e:
        # Validity check errors are non-fatal — log and continue
        print(f"[VERIFIER] Check 2 warning: {e}")

    print("[VERIFIER] Check 2 (Validity) passed.")

    # ── Check 2e: Pre-execution constraint check (write operations only) ──────
    operation_type = get_operation_type(sql)
    if operation_type in ("INSERT", "UPDATE", "DELETE"):
        print("[VERIFIER] Running Check 2e — Pre-execution constraint check...")
        conflict_type, conflict_desc = pre_execution_check(sql, db_path, allowed_schema)
        if conflict_type:
            print(f"[VERIFIER] Check 2e FAILED: {conflict_type} — {conflict_desc}")
            return send_fail(f"{conflict_type}: {conflict_desc}")
        print("[VERIFIER] Check 2e (Pre-execution) passed.")

    # ── Check 3: Consistency ──────────────────────────────────────────────────
    print("[VERIFIER] Running Check 3 — Consistency...")
    try:
        operation_for_check3 = get_operation_type(sql)
        fail_type = ""
        fail_reason = ""

        if operation_for_check3 == "SELECT":
            # For read operations: execute the SQL and verify the actual result
            # The Verifier may execute SELECT safely — it is non-destructive
            print("[VERIFIER] Check 3 — executing SELECT to verify result...")
            execution_result = run_sql(db_path, sql)

            if execution_result.startswith("ERROR"):
                print(f"[VERIFIER] Check 3 FAILED: SQL execution error — {execution_result}")
                return send_fail(f"Consistency: SQL produced an execution error — {execution_result}")

            # Build row filter context so LLM knows security restrictions are intentional
            row_filter_context = ""
            if row_filters and not benchmark_mode:
                row_filter_context = (
                    f"\nRow-level security is active for this query. "
                    f"The following WHERE conditions are mandatory security constraints "
                    f"automatically enforced by the system — do NOT flag them as errors:\n"
                    + "\n".join(f"  - {condition}" for condition in row_filters.values())
                    + "\nThe result may therefore contain fewer rows than expected "
                    f"because it is intentionally restricted to the current user's data.\n"
                )

            reasoning_context = f"Coder reasoning: {reasoning}\n" if reasoning else ""

            complexity = state["plan"].get("description", {}).get("complexity", "simple")
            complexity_guidance = {
                "simple":        "This is a simple query — verify the correct table and filter.",
                "comparative":   "This is a comparative query — verify the threshold or ranking logic is correct.",
                "analytical":    "This is an analytical query — verify aggregation, grouping, and HAVING conditions carefully.",
                "relational":    "This is a relational query — verify the join condition and that all required tables are combined correctly.",
                "set_operation": "This is a set operation query — verify that INTERSECT or EXCEPT is used correctly and that both subqueries reference the correct columns."
            }.get(complexity, "")

            response = invoke_verifier_llm([
                SystemMessage(content=f"""You are a SQL verification expert.
Your task is to determine whether the SQL query and its execution result correctly answer the user's question.
Your final verdict must be based on the question, SQL query, schema, and execution result, the coder reasoning is provided only as supporting context.

Verify whether:
1. Correct tables and relationships are used.
2. Filtering conditions match the question.
3. Aggregation, grouping, comparison, or ranking logic is correct if required.
4. Result shape matches the request (single value, list, grouped table, etc.).
5. Final result correctly answers the question.


Rules:
- Do not discuss schema design quality.
- Do not invent missing conditions.
- Do not explain step-by-step reasoning.
- Do not reconsider your conclusion after deciding.
- If an empty result is plausible given the database contents, PASS it.
- Row-level security filters enforced by the system are intentional and valid.
- Focus only on whether the final result answers the question correctly.
{complexity_guidance}                              

Reply with ONLY one of the following:
PASS
or
FAIL_TYPE: <wrong_table|wrong_column|wrong_join|missing_filter|extra_filter|wrong_aggregation|wrong_grouping|wrong_ordering|wrong_limit|wrong_subquery|empty_result|syntax_error|wrong_operation|wrong_result>
FAIL_REASON: <one short sentence>"""),
               HumanMessage(content=(
    f"Question: {question}\n"
    f"SQL: {sql}\n"
    f"Schema: {allowed_schema}\n"
    f"Actual result from database: {execution_result[:500]}\n"
    f"{row_filter_context}"
    f"{reasoning_context}"
))
            ])

        else:
            print("[VERIFIER] Check 3 — verifying write operation logic...")
            reasoning_context = f"Coder reasoning: {reasoning}\n" if reasoning else ""
            response = invoke_verifier_llm([
                SystemMessage(content="""You are a SQL verification expert.
Determine whether the SQL query correctly performs the requested write operation.
The coder reasoning is provided only as supporting context — your final verdict must be based on the question, SQL query, schema, and database statistics.

Verify whether:
1. Correct tables are referenced.
2. Correct columns are set or modified.
3. WHERE conditions correctly target what was requested.
4. Operation type (INSERT/UPDATE/DELETE) matches the intent.
5. Values being inserted or updated are appropriate given the schema and statistics.

Rules:
- Do not discuss schema design quality.
- Do not reconsider your conclusion after deciding.
- Do not explain step-by-step reasoning.
- Do not assume a query is wrong simply because it is simple.
- Focus on correctness of the operation, not stylistic SQL preferences.
- Do not invent conditions that were not requested.

Reply with ONLY one of the following:
PASS
or
FAIL_TYPE: <wrong_table|wrong_column|wrong_filter|wrong_values|wrong_operation|missing_where|extra_where|wrong_result>
FAIL_REASON: <one short sentence>"""),
                HumanMessage(content=(
                    f"Question: {question}\n"
                    f"SQL: {sql}\n"
                    f"Schema: {allowed_schema}\n"
                    f"Database Statistics: {stats}\n"
                    f"{reasoning_context}"
                ))
            ])

        response_text = get_content(response).strip()

        if len(response_text) > 500:
            return send_fail("Consistency: malformed verifier response — response too long")

        if response_text.upper().startswith("PASS"):
            return send_pass()

        if "FAIL_TYPE:" not in response_text or "FAIL_REASON:" not in response_text:
            return send_fail("Consistency: malformed verifier response — missing FAIL_TYPE or FAIL_REASON")

        verdict_lines = response_text.splitlines()
        verdict_raw = verdict_lines[0].strip()

        fail_type = ""
        fail_reason = ""
        for line in verdict_lines:
            if line.startswith("FAIL_TYPE:"):
                fail_type = line[10:].strip()
            elif line.startswith("FAIL_REASON:"):
                fail_reason = line[12:].strip()

        print(f"[VERIFIER] Check 3 (Consistency) LLM verdict: {verdict_raw}")
        if fail_type:
            print(f"[VERIFIER] Fail type: {fail_type}")
        if fail_reason:
            print(f"[VERIFIER] Fail reason: {fail_reason}")

    except Exception as e:
        print(f"[VERIFIER] Check 3 LLM error: {e} — passing through to coordinator")
        return send_fail(f"Consistency check failed due to LLM error: {e}")

    if fail_type and fail_reason:
        return send_fail(f"Consistency: {fail_type} — {fail_reason}")
    elif fail_type:
        return send_fail(f"Consistency: {fail_type}")
    else:
        return send_fail("Consistency: " + verdict_raw)


# ─── Routing ──────────────────────────────────────────────────────────────────

def route_from_coordinator(state: QueryState) -> str:
    # FIX 1: receiver strings must match graph node names exactly
    receiver = state["message"]["receiver"]
    if receiver == "user_coder":
        return "user_coder"
    elif receiver == "admin_coder":
        return "admin_coder"
    elif receiver == "verifier":
        return "verifier"
    return "end"


def route_from_verifier(state: QueryState) -> str:
    # Verifier always reports back to coordinator
    return "coordinator"


# ─── Graph ────────────────────────────────────────────────────────────────────

def build_query_graph():
    graph = StateGraph(QueryState)
    graph.add_node("coordinator", coordinator)
    graph.add_node("user_coder", user_coder)
    graph.add_node("admin_coder", admin_coder)
    graph.add_node("verifier", verifier)

    graph.set_entry_point("coordinator")

    # FIX 1: routing strings match node names exactly
    graph.add_conditional_edges(
        "coordinator", route_from_coordinator,
        {
            "user_coder": "user_coder",
            "admin_coder": "admin_coder",
            "verifier": "verifier",
            "end": END
        }
    )

    # Coders always report back to coordinator
    graph.add_edge("user_coder", "coordinator")
    graph.add_edge("admin_coder", "coordinator")

    # Verifier always goes back to coordinator
    graph.add_conditional_edges(
        "verifier", route_from_verifier,
        {"coordinator": "coordinator"}
    )

    return graph.compile()


# ─── Entry point ──────────────────────────────────────────────────────────────

def run_query_agent(
    question: str,
    db_path: str,
    role: str = "user",
    user_id: int = None,
    benchmark_mode: bool = False,
    no_plan: bool = False,
    no_verify: bool = False,
    no_schema_link: bool = False,
    no_memory: bool = False
) -> str:
    message_log.clear()
    conversation_id = str(uuid.uuid4())

    # Clean up old completed intents to prevent table growing indefinitely
    try:
        import lock_manager as _lm_cleanup
        _lm_cleanup.cleanup_old_intents()
    except Exception:
        pass

    initial_message = create_message(
        sender="user",
        receiver="coordinator",
        msg_type="instruction",
        content={
            "question": question,
            "db_path":  db_path,
            "role":     role,
            "user_id":  user_id
        },
        memory={
            "short_term": "",
            "episodic": {"coordinator_retries": 0, "current_task_id": "none"},
            "semantic": ""
        },
        conversation_id=conversation_id
    )

    initial_state = QueryState(
        message=initial_message,
        verdict="",
        coordinator_retries=0,
        sql_candidates=[],
        benchmark_mode=benchmark_mode,
        plan={},
        task_list=[],
        no_plan=no_plan,
        no_verify=no_verify,
        no_schema_link=no_schema_link,
        no_memory=no_memory,
        replan_count=0
    )

    graph = build_query_graph()
    try:
        final_state = graph.invoke(initial_state, config={"recursion_limit": 50})
    except Exception as e:
        print(f"[RUN_QUERY_AGENT] Unhandled graph error: {e}")
        # Create a synthetic end message so message_log is never empty
        error_message = create_message(
            sender="coordinator",
            receiver="end",
            msg_type="result",
            content={
                "question": question,
                "result":   f"System error: {e}",
                "verdict":  "fail",
                "role":     role
            },
            memory={
                "short_term": f"System error: {e}",
                "episodic": {"coordinator_retries": 0, "current_task_id": "none"},
                "semantic": ""
            },
            conversation_id=conversation_id
        )
        return f"System error during query execution: {e}", []

    print(f"\n{'='*60}")
    print(f"[COMMUNICATION SUMMARY] Total messages exchanged: {len(message_log)}")
    for i, msg in enumerate(message_log, 1):
        print(
            f"  {i}. [{msg['message_type'].upper()}] "
            f"{msg['sender']} -> {msg['receiver']} @ {msg['timestamp']}"
        )
    print(f"{'='*60}\n")
    print(f"[MEMORY] Episodic entries stored: {mem.episodic_count()}")
    print(f"[MEMORY] Procedural entries stored: {mem.procedural_count()}")
    print(f"[MEMORY] Databases with cached schema: {mem.schema_store_count()}")

    # Save message log to JSON for inspection
    try:
        with open("message_log.json", "w") as f:
            json.dump(message_log, f, indent=2)
    except Exception:
        pass

    return final_state["message"]["content"].get("result", "No result."), list(message_log)