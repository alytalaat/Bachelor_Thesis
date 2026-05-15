import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
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
import threading
import memory as mem
import access_control as ac

_GLOBAL_WRITE_LOCK = {
    "locked_tables": set(),
    "held_by": "",
    "queue": []
}
_LOCK_MUTEX = threading.Lock()




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
Given a natural language question and a full database schema, return ONLY the CREATE TABLE statements for tables needed to answer the question.
Include tables needed for JOINs even if not directly mentioned.
Return ONLY the CREATE TABLE statements, no explanation."""),
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


def pre_execution_check(sql: str, db_path: str, schema: str) -> tuple:
    """
    Checks for CIRCULAR or CONSTRAINT conflicts before a write executes.
    Returns (conflict_type, description) or (None, None) if clean.
    """
    import re
    schema_tables = set(extract_table_names(schema))
    tables_in_sql = list(set(re.findall(
        r'(?:FROM|INTO|UPDATE|JOIN)\s+(\w+)', sql, re.IGNORECASE
    )) & schema_tables)

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
            visited.add(node)
            rec_stack.add(node)
            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False
        return any(dfs(n) for n in graph if n not in visited)

    if has_cycle(fk_graph):
        involved = [t for t in tables_in_sql if fk_graph.get(t)]
        return ("CIRCULAR", f"Circular foreign key dependency detected among tables: {involved}")

    operation = get_operation_type(sql)
    sql_upper = sql.upper()
    statements = [s.strip() for s in sql.split(";") if s.strip()]
    if len(statements) > 1:
        return (None, None)

    for table in tables_in_sql:
        if operation == "INSERT":
            pragma = run_sql(db_path, f'PRAGMA table_info("{table}");')
            if "ERROR" not in pragma and "No results" not in pragma:
                for row in pragma.splitlines():
                    try:
                        col = eval(row)
                        col_name = col.get("name", "")
                        if (col.get("notnull", 0)
                                and col.get("dflt_value") is None
                                and not col.get("pk", 0)
                                and col_name.upper() not in sql_upper):
                            return (
                                "CONSTRAINT",
                                f"NOT NULL constraint would be violated: column "
                                f"{col_name} in table {table} requires a value"
                            )
                    except Exception:
                        pass

        if operation in ("INSERT", "UPDATE"):
            idx_result = run_sql(db_path, f'PRAGMA index_list("{table}");')
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
                                        val_match = re.search(
                                            rf"['\"]?{re.escape(col_name)}['\"]?\s*[=,)]\s*['\"]?(\w+)['\"]?",
                                            sql, re.IGNORECASE
                                        )
                                        if val_match:
                                            val = val_match.group(1)
                                            chk = run_sql(db_path, f'SELECT COUNT(*) FROM "{table}" WHERE "{col_name}" = \'{val}\';')
                                            count = int(list(eval(chk.strip()).values())[0])
                                            if count > 0:
                                                return (
                                                    "CONSTRAINT",
                                                    f"UNIQUE constraint would be violated: value already exists in column {col_name} in table {table}"
                                                )
                                    except Exception:
                                        pass
                    except Exception:
                        pass

        fk_result = run_sql(db_path, f'PRAGMA foreign_key_list("{table}");')
        if "ERROR" not in fk_result and "No results" not in fk_result:
            for fk_row in fk_result.splitlines():
                try:
                    fk = eval(fk_row)
                    fk_col = fk.get("from", "")
                    parent_table = fk.get("table", "")
                    parent_col = fk.get("to", "")
                    val_match = re.search(
                        rf"['\"]?{re.escape(fk_col)}['\"]?\s*[=,)]\s*['\"]?(\w+)['\"]?",
                        sql, re.IGNORECASE
                    )
                    if val_match:
                        val = val_match.group(1)
                        chk = run_sql(db_path, f'SELECT COUNT(*) FROM "{parent_table}" WHERE "{parent_col}" = \'{val}\';')
                        count = int(list(eval(chk.strip()).values())[0])
                        if count == 0:
                            return (
                                "CONSTRAINT",
                                f"Foreign key constraint would be violated: no matching row in {parent_table} for value {val}"
                            )
                except Exception:
                    pass

    return (None, None)


def post_execution_check(sql: str, db_path: str, schema: str, snapshot_before: dict) -> tuple:
    """
    Checks for TRANSACTION anomalies after a successful write.
    Returns (conflict_type, description) or (None, None) if clean.
    """
    import re
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
        benchmark_mode = state.get("benchmark_mode", False)
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
            return {**state, "message": outgoing, "verdict": "fail"}

        if benchmark_mode:
            # Benchmark mode — skip all access control, use full schema
            print("[COORDINATOR] Benchmark mode — access control disabled")
            allowed_schema = smart_link_schema(question, full_schema)
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
                return {**state, "message": outgoing, "verdict": "fail"}
            allowed_schema = smart_link_schema(question, permission_schema)
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

        # Retrieve episodic memory — past episodes for similar questions
        episodic_hints = mem.episodic_retrieve(
            question=question,
            db_path=db_path,
            schema=allowed_schema,
            user_id=user_id if role == "user" else None
        )

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

Complexity levels:
- simple: single entity straightforward lookup or filter
- comparative: requires comparing entities against a threshold or each other
- analytical: requires multi-step reasoning or aggregation over groups
- relational: requires combining information from multiple entities

Reply in this exact format with no extra text:

GOAL: <one sentence describing what the user wants returned>
OPERATION: <SELECT|INSERT|UPDATE|DELETE>
COMPLEXITY: <simple|comparative|analytical|relational>
LOGICAL_STEPS: <a single paragraph describing the logical relationships and data transformations needed in plain English — focus on entities, relationships, and business logic. Do NOT mention SQL clauses, join types, aggregation functions, or any database implementation details. Write as if explaining to a business analyst, not a programmer.>
VALUES: <values to insert or update, or None>
CONSTRAINTS:
- <mandatory rules the SQL MUST satisfy — security requirements, hard exclusions, or specific output requirements that cannot be inferred from the schema or question alone. If nothing mandatory applies, write: None>"""),
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
                ))
            ])

            raw = get_content(response).strip()

            goal = ""
            operation = ""
            complexity = "simple"
            intent = ""
            values = "None"
            constraints = []
            in_constraints = False

            for line in raw.splitlines():
                line = line.strip()
                if line.startswith("GOAL:"):
                    goal = line[5:].strip()
                    in_constraints = False
                elif line.startswith("OPERATION:"):
                    operation = line[10:].strip()
                    in_constraints = False
                elif line.startswith("COMPLEXITY:"):
                    complexity = line[11:].strip().lower()
                    in_constraints = False
                elif line.startswith("LOGICAL_STEPS:"):
                    intent = line[14:].strip()
                    in_constraints = False
                elif line.startswith("VALUES:"):
                    values = line[7:].strip()
                    in_constraints = False
                elif line.startswith("CONSTRAINTS:"):
                    in_constraints = True
                elif in_constraints and line.startswith("-"):
                    constraints.append(line[1:].strip())

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
                return {**state, "message": outgoing, "verdict": "fail"}

            plan = {
                "initial_state": initial_state_data,
                "goal": goal,
                "description": {
                    "operation":     operation,
                    "complexity":    complexity,
                    "logical_steps": intent,
                    "values":        values if operation in ("INSERT", "UPDATE") else None
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
            "sql_candidates": []
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

        with _LOCK_MUTEX:
            import lock_manager as _lm
            import os as _os
            file_held, file_state = _lm.is_held(list(tables_in_sql))
            file_pid = file_state.get("pid")
            current_pid = _os.getpid()

            if file_held and file_pid != current_pid:
                conflicting = [t for t in tables_in_sql
                               if t in file_state.get("locked_tables", [])]
                if conflicting:
                    print(f"[COORDINATOR] CONFLICT DETECTED — table(s) {conflicting} "
                          f"locked by '{file_state.get('held_by')}' (PID {file_pid})")
                    print(f"[COORDINATOR] Enforcing write lock constraint — "
                          f"queuing task and waiting for lock release...")

                    # Wait for lock release inside coordinator
                    released = _lm.wait_for_release(conflicting, timeout=30)

                    if released:
                        print(f"[COORDINATOR] Lock released — re-dispatching task: "
                              f"{question[:60]}")
                        # Re-acquire global lock and proceed with execution
                        _GLOBAL_WRITE_LOCK["locked_tables"].update(tables_in_sql)
                        _GLOBAL_WRITE_LOCK["held_by"] = "admin_coder"
                        print(f"[COORDINATOR] Table-level lock ACQUIRED on "
                              f"{set(tables_in_sql)}")
                    else:
                        print(f"[COORDINATOR] Lock wait timed out after 30s — aborting task")
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
                                             "current_task_id": "task_001"},
                                "semantic": allowed_schema
                            },
                            conversation_id=conversation_id
                        )
                        return {**state, "message": outgoing, "verdict": "fail"}

            elif operation in ("INSERT", "UPDATE", "DELETE"):
                conflicting_global = [t for t in tables_in_sql
                                      if t in _GLOBAL_WRITE_LOCK["locked_tables"]]
                if conflicting_global:
                    print(f"[COORDINATOR] CONFLICT — tables {set(conflicting_global)} "
                          f"locked by '{_GLOBAL_WRITE_LOCK['held_by']}' — queuing this task")
                    _GLOBAL_WRITE_LOCK["queue"].append({
                        "sql": sql,
                        "question": question,
                        "db_path": db_path,
                        "role": role,
                        "queued_at": datetime.now(timezone.utc).isoformat(),
                        "max_wait_seconds": 30,
                        "operation": operation
                    })
                    outgoing = create_message(
                        sender="coordinator",
                        receiver="end",
                        msg_type="result",
                        content={
                            "question": question,
                            "result": f"Operation queued — tables {set(conflicting_global)} "
                                      f"are locked by another agent.",
                            "verdict": "queued",
                            "role": role
                        },
                        memory={
                            "short_term": f"CONFLICT: tables {set(conflicting_global)} "
                                          f"locked — task queued",
                            "episodic": {"coordinator_retries": coordinator_retries,
                                         "current_task_id": "task_001"},
                            "semantic": allowed_schema
                        },
                        conversation_id=conversation_id
                    )
                    return {**state, "message": outgoing, "verdict": "queued"}
                else:
                    _GLOBAL_WRITE_LOCK["locked_tables"].update(tables_in_sql)
                    _GLOBAL_WRITE_LOCK["held_by"] = "admin_coder"
                    print(f"[COORDINATOR] Table-level lock ACQUIRED on "
                          f"{set(tables_in_sql)}")

        execution_result = run_sql(db_path, sql)
        print(f"[COORDINATOR] Result: {str(execution_result)[:200]}")

        # Post-execution anomaly check for write operations only
        if operation in ("INSERT", "UPDATE", "DELETE") and not execution_result.startswith("ERROR"):
            post_type, post_desc = post_execution_check(sql, db_path, allowed_schema, snapshot_before)
            if post_type:
                print(f"[COORDINATOR] Post-execution FAILED: {post_type} — {post_desc}")
                execution_result = f"ERROR: Post-execution anomaly — {post_desc}"

        if operation in ("INSERT", "UPDATE", "DELETE"):
            with _LOCK_MUTEX:
                released = list(_GLOBAL_WRITE_LOCK["locked_tables"])
                _GLOBAL_WRITE_LOCK["locked_tables"].clear()
                _GLOBAL_WRITE_LOCK["held_by"] = ""
                print(f"[COORDINATOR] Table-level lock released on {released}")
                if _GLOBAL_WRITE_LOCK["queue"]:
                    queued = _GLOBAL_WRITE_LOCK["queue"][0]
                    queued_at = datetime.fromisoformat(queued["queued_at"])
                    waited = (datetime.now(timezone.utc) - queued_at.replace(tzinfo=timezone.utc)).total_seconds()
                    max_wait = queued.get("max_wait_seconds", 30)
                    if waited > max_wait:
                        _GLOBAL_WRITE_LOCK["queue"].pop(0)
                        print(f"[COORDINATOR] Queued task timed out after {waited:.1f}s — aborting")
                    else:
                        _GLOBAL_WRITE_LOCK["queue"].pop(0)
                        print(f"[COORDINATOR] Re-dispatching queued task after {waited:.1f}s: {queued['question'][:60]}")

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
        return {**state, "message": outgoing, "verdict": final_verdict, "task_list": task_list}

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
        print(f"[COORDINATOR] DEADLOCK detected — queuing task instead of rewriting SQL")
        with _LOCK_MUTEX:
            _GLOBAL_WRITE_LOCK["queue"].append({
                "sql": sql,
                "question": question,
                "db_path": db_path,
                "role": role,
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "max_wait_seconds": 30,
                "operation": "SELECT"
            })
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
        task_list[0]["input"]["failure_reason"] = reason
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
            user_id=user_id if role == "user" else None
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
    if failure_reason:
        failure_context = (
            f"\n\nPrevious attempt failed with this reason: {failure_reason}\n"
            f"Identify specifically what was wrong and correct only that part.\n"
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

    coder_prompt = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"{security_block}"
            f"\nQuestion: {question}\n"
            f"Task goal: {task_goal}\n"
            f"Schema: {allowed_schema}\n"
            f"Plan:\n"
            f"  Operation: {plan['description']['operation']}\n"
            f"  Complexity: {plan['description'].get('complexity', 'simple')}\n"
            f"  Logical Steps: {plan['description'].get('logical_steps', '')}\n"
            + (f"  Values: {plan['description']['values']}\n" if plan['description'].get('values') else "")
            + f"Constraints: {'; '.join(regular_constraints)}"
            f"{hints_context}"
            f"{failure_context}\n\nWrite the SQL:"
        ))
    ]

    is_first_attempt = coordinator_retries == 0
    num_candidates = 3 if is_first_attempt else 1

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
    # Each fallback keeps its own reasoning as a (sql, reasoning) tuple
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
    "You are a SQL expert specializing in complex queries.\n"
    "Write ONLY the SQL query based on the plan and schema.\n"
    "You are only permitted to generate SELECT statements. "
    "If the plan requires INSERT, UPDATE, or DELETE, respond with exactly: "
    "OPERATION_NOT_PERMITTED\n"
    "General rules:\n"
    "- No explanations. No markdown. No backticks. Raw SQL only, ending with a semicolon.\n"
    "- Column and table names must exactly match the schema.\n"
    "SELECT rules:\n"
    "- Use correct JOIN syntax with ON conditions matching foreign keys in the schema\n"
    "- Use subqueries or CTEs (WITH ...) for multi-step logic\n"
    "- Use subqueries in WHERE when filtering by an aggregate (e.g. WHERE x > (SELECT AVG(x) FROM ...))\n"
    "- Use subqueries in FROM when you need to compute intermediate results before filtering\n"
    "- If the Logical Steps describe a comparative or threshold pattern, consider whether a subquery is more appropriate than a JOIN\n"
    "- Use HAVING for conditions on aggregated values in the outer query, not WHERE\n"
    "- Use DISTINCT when duplicates must be eliminated\n"
    "- Use INTERSECT / EXCEPT / UNION when combining result sets\n"
    "REASONING REQUIREMENT:\n"
    "After writing the SQL, on a new line write:\n"
    "REASONING: <one sentence explaining your key choices — which tables, columns, conditions, and why>\n"
    "The REASONING line is mandatory. Always include it after the SQL."
)
    return _coder_node(state, system_prompt, "user_coder")


def admin_coder(state: QueryState) -> QueryState:
    system_prompt = (
    "You are a SQL expert specializing in complex queries.\n"
    "Write ONLY the SQL query based on the plan and schema.\n"
    "You are permitted to generate SELECT, INSERT, UPDATE, and DELETE statements.\n"
    "General rules:\n"
    "- No explanations. No markdown. No backticks. Raw SQL only, ending with a semicolon.\n"
    "- Column and table names must exactly match the schema.\n"
    "SELECT rules:\n"
    "- Use correct JOIN syntax with ON conditions matching foreign keys in the schema\n"
    "- Use subqueries or CTEs (WITH ...) for multi-step logic\n"
    "- Use subqueries in WHERE when filtering by an aggregate (e.g. WHERE x > (SELECT AVG(x) FROM ...))\n"
    "- Use subqueries in FROM when you need to compute intermediate results before filtering\n"
    "- If the Logical Steps describe a comparative or threshold pattern, consider whether a subquery is more appropriate than a JOIN\n"
    "- Use HAVING for conditions on aggregated values in the outer query, not WHERE\n"
    "- Use DISTINCT when duplicates must be eliminated\n"
    "- Use INTERSECT / EXCEPT / UNION when combining result sets\n"
    "INSERT rules:\n"
    "- Use INSERT INTO table (col1, col2, ...) VALUES (val1, val2, ...)\n"
    "- Only include columns that exist in the schema\n"
    "UPDATE rules:\n"
    "- Use UPDATE table SET col = val WHERE condition\n"
    "- Always include a WHERE clause to avoid updating all rows\n"
    "DELETE rules:\n"
    "- Use DELETE FROM table WHERE condition\n"
    "- Always include a WHERE clause to avoid deleting all rows\n"
    "REASONING REQUIREMENT:\n"
    "After writing the SQL, on a new line write:\n"
    "REASONING: <one sentence explaining your key choices — which tables, columns, conditions, and why>\n"
    "The REASONING line is mandatory. Always include it after the SQL."
)
    return _coder_node(state, system_prompt, "admin_coder")


# ─── Verifier ─────────────────────────────────────────────────────────────────

def verifier(state: QueryState) -> QueryState:
    print("\n[VERIFIER] Running verification checks...")

    import re

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
                    col_name = func_match.group(1).strip()
                    if col_name == '*' or not col_name:
                        continue
                    # Strip table alias prefix inside function e.g. COUNT(s.Age) -> Age
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
            import re as _re
            for table, condition in row_filters.items():
                table_lower = table.lower()
                tables_lower = [t.lower() for t in tables_in_sql]
                if table_lower in tables_lower:
                    parts = condition.split("=")
                    if len(parts) == 2:
                        filter_col = parts[0].strip()
                        filter_val = parts[1].strip()
                        pattern = rf'{filter_col}\s*=\s*{filter_val}'
                        if not _re.search(pattern, sql, _re.IGNORECASE):
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

            response = invoke_verifier_llm([
                SystemMessage(content="""You are a SQL result verification expert.
Given the original question, the SQL query, the schema, and the actual result, decide if the result correctly answers the question.

The coder has provided their reasoning. Use it to understand their intent before evaluating — if the reasoning is sound but the result looks unexpected, consider whether the database data explains it rather than assuming the SQL is wrong.

Check:
- Does the result contain the information the question asked for?
- Is the result shape correct — single value, list, or table as appropriate?
- Are the values plausible given the question and schema?
- If the question asked for a maximum, minimum, count, or specific filter, does the result reflect that?
- If the result is empty, is that genuinely correct given the data, or does it indicate a wrong query?
- Do not fail a result just because it is shorter or simpler than expected — if it correctly answers the question, pass it.

Reply with ONLY:
  PASS — if the result correctly answers the question
  FAIL: <reason> — if the result is wrong, empty when it should not be, or does not match what was asked"""),
                HumanMessage(content=(
                    f"Question: {question}\n"
                    f"SQL: {sql}\n"
                    f"{reasoning_context}"
                    f"Schema: {allowed_schema}\n"
                    f"Actual result from database: {execution_result[:500]}\n"
                    f"{row_filter_context}"
                ))
            ])

        else:
            print("[VERIFIER] Check 3 — verifying write operation logic...")
            reasoning_context = f"Coder reasoning: {reasoning}\n" if reasoning else ""
            response = invoke_verifier_llm([
                SystemMessage(content="""You are a SQL verification expert.
Given the original question, the SQL query, the schema, and database statistics, decide if the SQL correctly performs the requested operation.

The coder has provided their reasoning. Use it to understand their intent before evaluating.

Check:
- The correct tables are referenced
- The correct columns are set or modified
- The WHERE conditions correctly target what was asked
- The operation type (INSERT/UPDATE/DELETE) matches the intent
- Values being inserted or updated are appropriate given the schema and statistics

Reply with ONLY:
  PASS — if the SQL correctly performs the requested operation
  FAIL: <reason> — if the SQL is wrong, targets wrong rows, or uses wrong values"""),
                HumanMessage(content=(
                    f"Question: {question}\n"
                    f"SQL: {sql}\n"
                    f"{reasoning_context}"
                    f"Schema: {allowed_schema}\n"
                    f"Database Statistics: {stats}\n"
                ))
            ])

        verdict_raw = get_content(response).strip().splitlines()[0].strip()
        print(f"[VERIFIER] Check 3 (Consistency) LLM verdict: {verdict_raw}")

    except Exception as e:
        print(f"[VERIFIER] Check 3 LLM error: {e} — passing through to coordinator")
        return send_fail(f"Consistency check failed due to LLM error: {e}")

    if verdict_raw.upper().startswith("PASS"):
        return send_pass()
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
    no_verify: bool = False
) -> str:
    message_log.clear()
    conversation_id = str(uuid.uuid4())

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
        no_verify=no_verify
    )

    graph = build_query_graph()
    try:
        final_state = graph.invoke(initial_state)
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
        return f"System error during query execution: {e}"

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

    return final_state["message"]["content"].get("result", "No result.")