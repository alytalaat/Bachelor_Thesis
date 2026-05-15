"""
SPIDER benchmark evaluation script with official evaluation integration.

Usage:
    python spider_runner.py --spider_dir path/to/spider --difficulty hard
    python spider_runner.py --spider_dir path/to/spider --difficulty hard --no_plan
    python spider_runner.py --spider_dir path/to/spider --difficulty extra --start 51
"""

import json
import os
import sys
import subprocess
import argparse
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

import llm_manager
from query_agent import run_query_agent, message_log

import memory as mem
from database import run_sql, get_schema
from memory import build_filtered_schema_from_string
import access_control as ac


def clean_sql(sql: str) -> str:
    """Collapses multiline SQL into a single line and strips trailing semicolon."""
    return " ".join(sql.split()).rstrip(";").strip()


def evaluate(
    spider_dir: str,
    split:      str  = "dev",
    start:      int  = 0,
    limit:      int  = 9999,
    difficulty: str  = None,
    no_plan:    bool = False,
    no_verify:  bool = False,
):
    data_file  = os.path.join(spider_dir, "dev_with_difficulty.json")
    db_dir     = os.path.join(spider_dir, "database")
    table_file = os.path.join(spider_dir, "tables.json")

    with open(data_file) as f:
        data = json.load(f)

    if difficulty:
        data = [d for d in data if d.get("difficulty") == difficulty]
        print(f"[RUNNER] Filtered to {len(data)} '{difficulty}' questions")
        if not data:
            print(f"[RUNNER] No questions found for difficulty '{difficulty}'.")
            return

    data = data[start:]
    data = data[:limit]
    print(f"[RUNNER] Running questions {start} to {start + len(data) - 1} ({len(data)} total)")
    if no_plan:
        print("[RUNNER] No-plan baseline mode")

    ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
    diff_tag  = difficulty or "all"
    if no_plan and no_verify:
        mode_tag = "noplan_noverify"
    elif no_plan:
        mode_tag = "noplan"
    elif no_verify:
        mode_tag = "noverify"
    else:
        mode_tag = "plan"
    base_name = f"{diff_tag}_{mode_tag}_{ts}"
    pred_file = f"predicted_{base_name}.sql"
    gold_file = f"gold_{base_name}.sql"

    print("[RUNNER] Clearing procedural and episodic memory...")
    mem.clear_procedural_memory()
    mem.clear_episodic_memory()

    correct        = 0
    errors         = 0
    total          = 0
    cached_results = []
    pred_lines     = []
    gold_lines     = []

    for item in data:
        question = item["question"]
        gold_sql = item["query"]
        db_id    = item["db_id"]
        db_path  = os.path.join(db_dir, db_id, f"{db_id}.sqlite")

        if not os.path.exists(db_path):
            print(f"[SKIP] DB not found: {db_path}")
            cached_results.append({
                "item": item, "predicted": None, "gold": gold_sql,
                "passed": False, "error": "DB not found", "reason": "DB not found"
            })
            continue

        total += 1

        try:
            predicted_result = run_query_agent(
                question, db_path, role="admin",
                benchmark_mode=True, no_plan=no_plan, no_verify=no_verify
            )
            if not isinstance(predicted_result, str):
                predicted_result = str(predicted_result)

            predicted_sql = ""
            for msg in reversed(message_log):
                if (msg.get("sender") == "verifier_agent"
                        and msg["content"].get("verdict") == "pass"):
                    predicted_sql = msg["content"].get("sql", "")
                    break
            if not predicted_sql:
                for msg in reversed(message_log):
                    if msg.get("sender") == "verifier_agent":
                        predicted_sql = msg["content"].get("sql", "")
                        break
            if not predicted_sql:
                # no_verify mode — get SQL from coder directly
                for msg in reversed(message_log):
                    if msg.get("sender") in ("admin_coder_agent", "user_coder_agent"):
                        predicted_sql = msg["content"].get("sql", "")
                        break
            if not predicted_sql:
                predicted_sql = "SELECT 1"

        except Exception as e:
            print(f"[ERROR] [{start + total}] {question}\n  {e}")
            errors += 1
            predicted_sql = "SELECT 1"
            cached_results.append({
                "item": item, "predicted": None, "gold": gold_sql,
                "passed": False, "error": str(e), "reason": str(e)[:60]
            })
            pred_lines.append(f"{clean_sql(predicted_sql)}\t{db_id}")
            gold_lines.append(f"{clean_sql(gold_sql)}\t{db_id}")
            continue

        gold_result = run_sql(db_path, gold_sql)
        passed = _quick_match(predicted_result, gold_result)
        if passed:
            correct += 1

        reason = "—"
        if not passed:
            for msg in reversed(message_log):
                if msg.get("sender") == "verifier_agent":
                    reason = msg["content"].get("reason", "verifier failed")[:60]
                    break

        cached_results.append({
            "item": item, "predicted": predicted_sql, "gold": gold_sql,
            "passed": passed, "error": None, "reason": reason
        })

        pred_lines.append(f"{clean_sql(predicted_sql)}\t{db_id}")
        gold_lines.append(f"{clean_sql(gold_sql)}\t{db_id}")

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] [{start + total}] {question[:80]}")
        if not passed:
            print(f"  Reason: {reason}")

        if passed:
            try:
                full_schema    = mem.get_full_schema_cached(db_path, get_schema)
                role_perms     = ac.build_role_permissions_dict("admin")
                allowed_schema = build_filtered_schema_from_string(
                    full_schema, "admin", {"admin": role_perms}
                )
                mem.procedural_store(
                    question=question, sql=predicted_sql,
                    db_path=db_path, schema=allowed_schema
                )
            except Exception as e:
                print(f"[RUNNER] Procedural store error: {e}")

    with open(pred_file, "w", encoding="utf-8") as f:
        f.write("\n".join(pred_lines))
    with open(gold_file, "w", encoding="utf-8") as f:
        f.write("\n".join(gold_lines))

    print(f"\n[RUNNER] Predicted SQLs: {pred_file}")
    print(f"[RUNNER] Gold SQLs:      {gold_file}")

    print(f"\n{'='*50}")
    print(f"In-runner EA: {correct}/{total} = {correct/total:.1%}" if total else "No questions.")
    print(f"Errors: {errors}/{total}")

    print(f"\n{'='*60}")
    print(" EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"{'#':<5} {'Status':<8} {'Question':<55} {'Reason'}")
    print(f"{'-'*5} {'-'*8} {'-'*55} {'-'*30}")
    for i, entry in enumerate(cached_results, start + 1):
        q      = entry["item"]["question"]
        passed = entry["passed"]
        error  = entry["error"]
        reason = entry["reason"]
        if error == "DB not found":
            print(f"{i:<5} {'SKIP':<8} {q[:55]:<55} DB not found")
        elif error:
            print(f"{i:<5} {'ERROR':<8} {q[:55]:<55} {error[:40]}")
        elif passed:
            print(f"{i:<5} {'PASS':<8} {q[:55]:<55} —")
        else:
            print(f"{i:<5} {'FAIL':<8} {q[:55]:<55} {reason}")

    print(f"\n{'='*60}")
    print(" OFFICIAL SPIDER EVALUATION")
    print(f"{'='*60}")

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-suite-sql-eval", "evaluation.py")
    if not os.path.exists(eval_script):
        print("[RUNNER] test-suite-sql-eval not found — skipping official eval.")
        print(f"  Run manually: python test-suite-sql-eval/evaluation.py --gold {gold_file} --pred {pred_file} --db {db_dir} --etype exec")
        return

    cmd = [
        sys.executable, eval_script,
        "--gold",  gold_file,
        "--pred",  pred_file,
        "--db",    db_dir,
        "--etype", "exec"
        ]
    print("[RUNNER] Running official eval...\n")
    subprocess.run(cmd)


def _quick_match(predicted: str, gold: str) -> bool:
    if not predicted or not gold:
        return False

    def normalize(s):
        if not s or s.strip() in ("No results.", "No result."):
            return []
        rows = []
        for line in s.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                import ast
                d = ast.literal_eval(line)
                def norm_val(v):
                    try:
                        f = float(v)
                        return str(int(f)) if f == int(f) else str(f)
                    except:
                        return str(v).strip().lower()
                rows.append("|".join(sorted(norm_val(v) for v in d.values())))
            except:
                rows.append(line.lower().strip())
        return sorted(rows)

    return normalize(predicted) == normalize(gold)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spider_dir", required=True)
    parser.add_argument("--split",      default="dev")
    parser.add_argument("--start",      type=int, default=0)
    parser.add_argument("--limit",      type=int, default=9999)
    parser.add_argument("--difficulty", default=None, choices=["easy", "medium", "hard", "extra"])
    parser.add_argument("--no_plan",    action="store_true", default=False)
    parser.add_argument("--no_verify",  action="store_true", default=False)
    args = parser.parse_args()

    evaluate(args.spider_dir, args.split, args.start, args.limit, args.difficulty, args.no_plan, args.no_verify)