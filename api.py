import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional
import json
from query_agent import run_query_agent

app = FastAPI(
    title="LLM-Coordinated Multi-Agent Database System",
    description="Natural language database queries using a multi-agent LLM pipeline",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Base directory for databases — works both locally and on Railway
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPIDER_DIR = os.path.join(BASE_DIR, "spider_data")
DEMO_DB = os.path.join(BASE_DIR, "demo_conflict.db")


class QueryRequest(BaseModel):
    question: str
    database: Optional[str] = None
    role: Optional[str] = "admin"
    user_id: Optional[int] = None


class QueryResponse(BaseModel):
    question: str
    database: str
    role: str
    result: str
    verdict: str


def resolve_db_path(database: Optional[str]) -> str:
    """
    Resolve database name to full path.
    If no database specified, use demo_conflict.db.
    If database name given, look for it in spider_data directory.
    """
    if not database:
        return DEMO_DB

    # Check if it is a full path
    if os.path.exists(database):
        return database

    # Look in spider_data directory
    spider_path = os.path.join(
        SPIDER_DIR, database, f"{database}.sqlite"
    )
    if os.path.exists(spider_path):
        return spider_path

    # Try demo_conflict.db as fallback
    if database == "demo":
        return DEMO_DB

    raise FileNotFoundError(
        f"Database '{database}' not found. "
        f"Available: demo, or Spider database names like 'concert_singer', 'college_2'"
    )


@app.get("/")
def root():
    return {
        "system": "LLM-Coordinated Multi-Agent Database System",
        "thesis": "LLM-Coordinated Multi-Agent Systems for Database Management Tasks",
        "author": "Aly Talaat",
        "supervisor": "Dr. Mayar Osama",
        "university": "German University in Cairo",
        "endpoints": {
            "POST /query": "Submit a natural language query",
            "GET /databases": "List available databases",
            "GET /health": "Health check"
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/databases")
def list_databases():
    """List all available databases."""
    available = ["demo"]

    if os.path.exists(SPIDER_DIR):
        for entry in os.scandir(SPIDER_DIR):
            if entry.is_dir():
                db_file = os.path.join(
                    entry.path, f"{entry.name}.sqlite"
                )
                if os.path.exists(db_file):
                    available.append(entry.name)

    return {
        "databases": sorted(available),
        "demo_description": "demo_conflict.db — students, courses, enrollments with FK constraints",
        "spider_description": "Spider benchmark databases for text-to-SQL evaluation"
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    """
    Submit a natural language query to the multi-agent pipeline.

    The system will:
    1. Generate an execution plan (Coordinator)
    2. Generate SQL (Coder agent)
    3. Verify the SQL (Verifier agent)
    4. Execute and return the result
    """
    try:
        db_path = resolve_db_path(request.database)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        # Use benchmark_mode for Spider databases — bypasses role-based access control
        # Use normal mode for demo database — enforces access control
        is_spider_db = request.database and request.database != "demo"

        result = run_query_agent(
            question=request.question,
            db_path=db_path,
            role=request.role,
            user_id=request.user_id,
            benchmark_mode=is_spider_db,
            no_plan=False,
            no_verify=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    db_name = request.database or "demo"
    verdict = "pass" if not str(result).startswith("ERROR") else "fail"

    return QueryResponse(
        question=request.question,
        database=db_name,
        role=request.role,
        result=str(result),
        verdict=verdict
    )


@app.post("/query-with-log")
def query_with_log(request: QueryRequest):
    """
    Submit a natural language query and get back both the result
    and the full agent communication log for UI display.
    """
    from query_agent import message_log

    try:
        db_path = resolve_db_path(request.database)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        is_spider_db = request.database and request.database != "demo"
        result = run_query_agent(
            question=request.question,
            db_path=db_path,
            role=request.role,
            user_id=request.user_id,
            benchmark_mode=is_spider_db,
            no_plan=False,
            no_verify=False
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline error: {str(e)}"
        )

    db_name = request.database or "demo"
    verdict = "pass" if not str(result).startswith("ERROR") else "fail"

    # Build clean message log for UI
    clean_log = []
    for msg in message_log:
        clean_log.append({
            "sender": msg["sender"],
            "receiver": msg["receiver"],
            "message_type": msg["message_type"],
            "timestamp": msg["timestamp"],
            "content_keys": list(msg["content"].keys()),
            "short_term": msg["memory"].get("short_term", "")[:200],
            "semantic_tables": msg["memory"].get("semantic_summary", [])
        })

    return {
        "question": request.question,
        "database": db_name,
        "role": request.role,
        "result": str(result),
        "verdict": verdict,
        "message_log": clean_log,
        "total_messages": len(clean_log)
    }


import os as _os
_static_dir = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "static")
if _os.path.exists(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="static")