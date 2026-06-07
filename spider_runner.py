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
from openai import models
load_dotenv()

import llm_manager
from query_agent import run_query_agent, message_log

import memory as mem
from database import run_sql, get_schema
from memory import build_filtered_schema_from_string
import access_control as ac


llm_manager.set_models(
    planner="openai/gpt-oss-120b",
    coder="gpt-5.4-mini-2026-03-17",
    verifier="gpt-5.4-nano-2026-03-17"
)

def clean_sql(sql: str) -> str:
    """Collapses multiline SQL into a single line and strips trailing semicolon."""
    return " ".join(sql.split()).rstrip(";").strip()


def evaluate(
    spider_dir: str,
    split:      str  = "dev",
    start:      int  = 0,
    limit:      int  = 9999,
    difficulty: str  = None,
    no_plan:       bool = False,
    no_verify:     bool = False,
    no_schema_link: bool = False,
    no_memory:     bool = False,
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
    if no_schema_link:
        mode_tag += "_noschema"
    if no_memory:
        mode_tag += "_nomemory"
    models = llm_manager.get_models()
    planner_tag = models["planner"].split("/")[-1].replace("-", "_")[:15]
    coder_tag = models["coder"].split("/")[-1].replace("-", "_")[:15]
    verifier_tag = models["verifier"].split("/")[-1].replace("-", "_")[:15]
    base_name = f"{diff_tag}_{mode_tag}_{planner_tag}_c{coder_tag}_{verifier_tag}_{ts}"
    pred_file    = os.path.abspath(f"predicted_{base_name}.sql")
    gold_file    = os.path.abspath(f"gold_{base_name}.sql")
    results_file = os.path.abspath(f"results_{base_name}.txt")

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
            predicted_result, _ = run_query_agent(
                question, db_path, role="admin",
                benchmark_mode=True, no_plan=no_plan, no_verify=no_verify,
                no_schema_link=no_schema_link, no_memory=no_memory
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
            pred_lines.append(clean_sql(predicted_sql))
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

        retries_used = 0
        for msg in reversed(message_log):
            if msg.get("sender") == "coordinator" and msg.get("receiver") == "end":
                retries_used = msg.get("memory", {}).get("episodic", {}).get("coordinator_retries", 0)
                break

        cached_results.append({
            "item": item, "predicted": predicted_sql, "gold": gold_sql,
            "passed": passed, "error": None, "reason": reason, "retries_used": retries_used,
            "predicted_result": predicted_result
        })

        pred_lines.append(clean_sql(predicted_sql))
        gold_lines.append(f"{clean_sql(gold_sql)}\t{db_id}")

        status = "PASS" if passed else "FAIL"
        print(f"[{status}] [{start + total}] {question[:80]}")
        if not passed:
            print(f"  Reason: {reason}")

        if passed and not no_schema_link and not no_verify:
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

    # Termination reason distribution
    from collections import Counter
    termination_reasons = []
    for entry in cached_results:
        if entry.get("error") == "DB not found":
            termination_reasons.append("DB not found")
        elif entry.get("error"):
            termination_reasons.append("system error")
        elif entry["passed"]:
            if entry.get("retries_used", 0) > 0:
                termination_reasons.append("success after retry")
            else:
                termination_reasons.append("success — first attempt")
        else:
            reason = entry["reason"]
            predicted = entry.get("predicted", "")
            predicted_result = entry.get("predicted_result", "")
            if predicted == "SELECT 1":
                termination_reasons.append("coder failed — SELECT 1 fallback")
            elif predicted_result.startswith("ERROR"):
                termination_reasons.append("execution error — syntax or runtime failure")
            elif reason == "—" or not reason or reason.strip() == "":
                termination_reasons.append("wrong result — no verification")
            elif reason.startswith("Could not generate"):
                termination_reasons.append("max retries exhausted")
            elif "wrong_table" in reason or "wrong_column" in reason or "wrong_join" in reason:
                termination_reasons.append("verifier failed — wrong schema usage")
            elif "missing_filter" in reason or "extra_filter" in reason:
                termination_reasons.append("verifier failed — wrong filter")
            elif "wrong_aggregation" in reason or "wrong_grouping" in reason or "wrong_ordering" in reason:
                termination_reasons.append("verifier failed — wrong aggregation or ordering")
            elif "empty_result" in reason:
                termination_reasons.append("verifier failed — empty result")
            elif "wrong_result" in reason or reason.startswith("Consistency:"):
                termination_reasons.append("verifier failed — wrong result")
            elif reason.startswith("Structural:"):
                termination_reasons.append("verifier failed — structural")
            elif reason.startswith("Validity:"):
                termination_reasons.append("verifier failed — validity")
            elif "max retries" in reason.lower():
                termination_reasons.append("max retries exhausted")
            elif "all checks passed" in reason.lower():
                termination_reasons.append("result mismatch — verifier approved wrong SQL")
            else:
                termination_reasons.append("verifier failed — other")

    reason_counts = Counter(termination_reasons)
    print(f"\n{'='*60}")
    print(" TERMINATION REASONS")
    print(f"{'='*60}")
    for reason, count in reason_counts.most_common():
        print(f"  {reason:<45} {count} ({count/total:.1%})")

    # Progress rounds
    print(f"\n{'='*60}")
    print(" PROGRESS ROUNDS")
    print(f"{'='*60}")
    print(f"  Runs that produced a procedural memory entry: {mem.procedural_count()} ({mem.procedural_count()/total:.1%})")
    print(f"  Total questions attempted: {total}")
    print(f"  Progress rate: {mem.procedural_count()/total:.1%}" if total else "")

    with open(results_file, "w", encoding="utf-8") as rf:
        rf.write(f"RUN: {base_name}\n")
        rf.write(f"Difficulty: {difficulty or 'all'}\n")
        rf.write(f"Questions: {start} to {start + total - 1} ({total} total)\n\n")
        rf.write(f"In-runner EA: {correct}/{total} = {correct/total:.1%}\n" if total else "No questions.\n")
        rf.write(f"Errors: {errors}/{total}\n\n")
        rf.write("EVALUATION SUMMARY\n")
        rf.write(f"{'#':<5} {'Status':<8} {'Question':<55} {'Reason'}\n")
        rf.write(f"{'-'*5} {'-'*8} {'-'*55} {'-'*30}\n")
        for i, entry in enumerate(cached_results, start + 1):
            q      = entry["item"]["question"]
            passed = entry["passed"]
            error  = entry["error"]
            reason = entry["reason"]
            if error == "DB not found":
                rf.write(f"{i:<5} {'SKIP':<8} {q[:55]:<55} DB not found\n")
            elif error:
                rf.write(f"{i:<5} {'ERROR':<8} {q[:55]:<55} {error[:40]}\n")
            elif passed:
                rf.write(f"{i:<5} {'PASS':<8} {q[:55]:<55} —\n")
            else:
                rf.write(f"{i:<5} {'FAIL':<8} {q[:55]:<55} {reason}\n")
        rf.write("\nTERMINATION REASONS\n")
        for reason, count in reason_counts.most_common():
            rf.write(f"  {reason:<45} {count} ({count/total:.1%})\n")
        rf.write(f"\nPROGRESS ROUNDS\n")
        rf.write(f"  Runs that produced a procedural memory entry: {mem.procedural_count()} ({mem.procedural_count()/total:.1%})\n")
        rf.write(f"  Total questions attempted: {total}\n")
    print(f"\n[RUNNER] In-runner results saved to: {results_file}")

    print(f"\n{'='*60}")
    print(" OFFICIAL SPIDER EVALUATION")
    print(f"{'='*60}")

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test-suite-sql-eval", "evaluation.py")
    if not os.path.exists(eval_script):
        print("[RUNNER] test-suite-sql-eval not found — skipping official eval.")
        print(f"  Run manually: python test-suite-sql-eval/evaluation.py --gold {gold_file} --pred {pred_file} --db {db_dir} --etype exec")
        return

    eval_suite_db_dir = os.path.join(os.path.dirname(eval_script), "database")

    cmd = [
        sys.executable, eval_script,
        "--gold",  gold_file,
        "--pred",  pred_file,
        "--db",    eval_suite_db_dir,
        "--etype", "exec"
    ]
    eval_dir = os.path.dirname(eval_script)
    print("[RUNNER] Running official eval...\n")
    with open(results_file, "a", encoding="utf-8") as rf:
        rf.write("\nOFFICIAL SPIDER EVALUATION\n")
        result = subprocess.run(
            cmd,
            cwd=eval_dir,
            capture_output=True,
            text=True
        )
        rf.write(result.stdout)
        if result.stderr:
            rf.write("\nSTDERR:\n")
            rf.write(result.stderr)
    print(result.stdout)
    print(f"[RUNNER] Official eval results appended to: {results_file}")


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
    parser.add_argument("--no_verify",     action="store_true", default=False)
    parser.add_argument("--no_schema_link", action="store_true", default=False)
    parser.add_argument("--no_memory",      action="store_true", default=False)
    args = parser.parse_args()

    evaluate(args.spider_dir, args.split, args.start, args.limit, args.difficulty, args.no_plan, args.no_verify, args.no_schema_link, args.no_memory)