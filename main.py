from dotenv import load_dotenv
load_dotenv()
import json
import sqlite3
import threading
import time
from database import create_test_db
from query_agent import run_query_agent, message_log

DB_PATH = "test.db"


def hold_db_lock(db_path: str, duration: float):
    """Holds an EXCLUSIVE write lock on the DB for `duration` seconds."""
    conn = sqlite3.connect(db_path, timeout=0)
    conn.execute("BEGIN EXCLUSIVE")
    time.sleep(duration)
    conn.rollback()
    conn.close()


def run_test(label, question, role, user_id=None, setup_fn=None, benchmark_mode=False):
    print(f"\n{'=' * 60}")
    print(f"TEST:     {label}")
    print(f"ROLE:     {role}")
    print(f"USER ID:  {user_id if user_id is not None else 'N/A'}")
    print(f"QUESTION: {question}")
    print("-" * 60)
    if setup_fn:
        setup_fn()
        time.sleep(0.3)
    response = run_query_agent(
        question,
        db_path=DB_PATH,
        role=role,
        user_id=user_id,
        benchmark_mode=benchmark_mode
    )
    print(f"\nRESPONSE: {response}")
    return response


def main():
    print("=" * 60)
    print(" Multi-Agent Database Management System — Full Test Suite")
    print("=" * 60)

    create_test_db(DB_PATH)

    all_messages = []
    results = []

    tests = [

        # ══════════════════════════════════════════════════════════════
        # SECTION 1 — OPERATION LEVEL: role-based SQL operation access
        # ══════════════════════════════════════════════════════════════

        ("OPERATION | USER — SELECT allowed",
         "Show me all students with GPA above 3.5",
         "user", None, None, "pass"),

        ("OPERATION | USER — INSERT blocked",
         "Insert a new student named Layla Ahmed age 20 from Cairo GPA 3.7",
         "user", None, None, "fail"),

        ("OPERATION | USER — UPDATE blocked",
         "Update the GPA of student with id 1 to 4.0",
         "user", None, None, "fail"),

        ("OPERATION | USER — DELETE blocked",
         "Delete the student with id 2 from the database",
         "user", None, None, "fail"),

        ("OPERATION | ADMIN — SELECT allowed",
         "Show me all students with GPA above 3.5",
         "admin", None, None, "pass"),

        ("OPERATION | ADMIN — INSERT allowed",
         "Insert a new student with id 30 name Rania Samir age 22 from Giza GPA 3.4",
         "admin", None, None, "pass"),

        ("OPERATION | ADMIN — UPDATE allowed",
         "Update the GPA of student with id 1 to 3.9",
         "admin", None, None, "pass"),

        ("OPERATION | ADMIN — DELETE allowed",
         "Delete the student with id 30 from the database",
         "admin", None, None, "pass"),

        # ══════════════════════════════════════════════════════════════
        # SECTION 2 — TABLE LEVEL: role-based table access
        # ══════════════════════════════════════════════════════════════

       
        ("TABLE | USER — allowed table (students)",
         "Show me all students from Cairo",
         "user", None, None, "pass"),

        ("TABLE | USER — forbidden table (enrollments) blocked",
         "Show me all enrollments for student id 1",
         "user", None, None, "fail"),

        ("TABLE | USER — forbidden table (courses) blocked",
         "List all available courses",
         "user", None, None, "fail"),

        ("TABLE | ADMIN — all tables allowed (enrollments)",
         "Show me all enrollments for student id 1",
         "admin", None, None, "pass"),

        ("TABLE | ADMIN — all tables allowed (courses)",
         "List all available courses",
         "admin", None, None, "pass"),

        ("TABLE | USER — JOIN across forbidden tables blocked",
         "Show me student names and the courses they are enrolled in",
         "user", None, None, "fail"),

        ("TABLE | ADMIN — JOIN across all tables allowed",
         "Show me student names and the courses they are enrolled in",
         "admin", None, None, "pass"),

        # ══════════════════════════════════════════════════════════════
        # SECTION 3 — COLUMN LEVEL: role-based column access
        # ══════════════════════════════════════════════════════════════

        ("COLUMN | USER — allowed columns (name gpa)",
         "Show me the names and GPAs of all students",
         "user", None, None, "pass"),

        ("COLUMN | USER — forbidden column (age) blocked",
         "Show me the age of all students",
         "user", None, None, "fail"),

        ("COLUMN | USER — mixed allowed and forbidden columns blocked",
         "Show me the name and age of all students",
         "user", None, None, "fail"),

        ("COLUMN | ADMIN — all columns allowed including age",
         "Show me the name age and GPA of all students",
         "admin", None, None, "pass"),

        # ══════════════════════════════════════════════════════════════
        # SECTION 4 — ROW LEVEL: user-specific row access
        # ══════════════════════════════════════════════════════════════

        ("ROW | USER id=1 — SELECT own row",
         "Show me my GPA",
         "user", 1, None, "pass"),

        ("ROW | USER id=2 — SELECT own row",
         "What is my GPA and city?",
         "user", 2, None, "pass"),

        ("ROW | USER id=1 — all students restricted to own row",
         "Show me all students GPAs",
         "user", 1, None, "pass"),

        ("ROW | USER id=1 — SELECT another users row blocked",
         "Show me the GPA of student with id 2",
         "user", 1, None, "fail"),

        ("ROW | ADMIN — no row restriction sees all rows",
         "Show me all students GPAs",
         "admin", None, None, "pass"),

        # ══════════════════════════════════════════════════════════════
        # SECTION 5 — CONFLICT DETECTION (admin role, write operations)
        # ══════════════════════════════════════════════════════════════

        ("CONFLICT | BASELINE — clean SELECT",
         "Which students have a GPA above 3.5 and are from Cairo?",
         "admin", None, None, "pass"),

        ("CONFLICT | SYNTAX — wrong column name",
         "Show me the student_name and gpa from students",
         "admin", None, None, "pass"),

        ("CONFLICT | CONSTRAINT — NOT NULL violation",
         "Insert a new student with id 50 age 21 from Alexandria GPA 3.2 but do not include a name",
         "admin", None, None, "fail"),

        ("CONFLICT | CONSTRAINT — FK violation",
         "Enroll student with id 999 in course_id 1 with grade A",
         "admin", None, None, "fail"),

        ("CONFLICT | CONSTRAINT — duplicate primary key",
         "Insert a new student with id 1 name Layla Ahmed age 20 from Cairo GPA 3.7",
         "admin", None, None, "fail"),

        ("CONFLICT | CIRCULAR — FK ordering conflict",
         "Write two separate INSERT statements: first insert a new student named Rania Samir "
         "with id 10 age 22 from Giza GPA 3.4 then insert an enrollment for that student "
         "in course_id 2 with grade B. Both must be in the same SQL.",
         "admin", None, None, "fail"),

        ("CONFLICT | TRANSACTION — commit with no active transaction",
         "Commit any pending database transaction that is currently open",
         "admin", None, None, "fail"),

        ("CONFLICT | TRANSACTION — overly broad DELETE",
         "Delete all students from the database",
         "admin", None, None, "fail"),

        ("CONFLICT | DEADLOCK — database locked by another connection",
         "Add a new student named Deadlock Test age 25 from Cairo GPA 3.0",
         "admin", None,
         lambda: threading.Thread(
             target=hold_db_lock, args=(DB_PATH, 12), daemon=True
         ).start(),
         "fail"),
         
         

    ]

    for label, question, role, user_id, setup_fn, expected in tests:
        run_test(label, question, role, user_id, setup_fn)

        actual_verdict = (
            message_log[-1]["content"].get("verdict", "unknown")
            if message_log else "unknown"
        )
        passed = actual_verdict == expected
        results.append({
            "label":    label,
            "role":     role,
            "user_id":  user_id,
            "expected": expected,
            "actual":   actual_verdict,
            "passed":   passed
        })

        for msg in message_log:
            msg["test_label"] = label
            msg["role"] = role
        all_messages.extend(message_log)
        print(f"[LOG] {len(message_log)} messages from this test ({len(all_messages)} total so far)")

    # ── Full summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(" FULL TEST SUMMARY")
    print(f"{'=' * 60}")
    print(f"{'Label':<55} {'Role':<8} {'UID':<5} {'Exp':<6} {'Act':<6} Result")
    print("-" * 92)
    for r in results:
        ok = "✓" if r["passed"] else "✗ FAIL"
        uid = str(r["user_id"]) if r["user_id"] is not None else "-"
        print(f"{r['label']:<55} {r['role']:<8} {uid:<5} {r['expected']:<6} {r['actual']:<6} {ok}")

    total = len(results)
    passed_count = sum(1 for r in results if r["passed"])
    print(f"\nOverall: {passed_count}/{total} passed")

    # Section breakdown
    print(f"\nBy section:")
    for section in ["OPERATION", "TABLE", "COLUMN", "ROW", "CONFLICT"]:
        section_tests = [r for r in results if r["label"].startswith(section)]
        if section_tests:
            section_passed = sum(1 for r in section_tests if r["passed"])
            print(f"  {section:<12} {section_passed}/{len(section_tests)} passed")

    # Role breakdown
    print(f"\nBy role:")
    for role in ["user", "admin"]:
        role_tests = [r for r in results if r["role"] == role]
        if role_tests:
            role_passed = sum(1 for r in role_tests if r["passed"])
            print(f"  {role:<8} {role_passed}/{len(role_tests)} passed")

    with open("message_log.json", "w") as f:
        json.dump(all_messages, f, indent=2)
    print(f"\n[LOG] All {len(all_messages)} messages saved to message_log.json")


if __name__ == "__main__":
    main()