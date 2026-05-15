import sqlite3
import time


def get_schema(db_path: str) -> str:
    """Returns CREATE TABLE statements with primary keys and foreign keys for all tables."""
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        schema_parts = []
        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()
            col_defs = []
            for col in columns:
                col_def = f"{col[1]} {col[2]}"
                if col[5]:  # pk flag
                    col_def += " PRIMARY KEY"
                col_defs.append(col_def)

            cursor.execute(f"PRAGMA foreign_key_list({table})")
            fks = cursor.fetchall()
            for fk in fks:
                col_defs.append(f"FOREIGN KEY ({fk[3]}) REFERENCES {fk[2]}({fk[4]})")

            schema_parts.append(f"CREATE TABLE {table} ({', '.join(col_defs)})")
        return "\n".join(schema_parts)
    except sqlite3.OperationalError as e:
        raise RuntimeError(f"Could not read schema: {e}") from e
    finally:
        if conn:
            conn.close()


def run_sql(db_path: str, sql: str) -> str:
    """Executes a SQL query and returns results as a string."""
    conn = None
    try:
        conn = sqlite3.connect(db_path, isolation_level=None)
        conn.execute("PRAGMA foreign_keys = ON")
        cursor = conn.cursor()

        # # Confirmation prompt for destructive operations
        # if sql.strip().upper().startswith(("UPDATE", "DELETE")):
        #     print(f"\n⚠️  WARNING: About to execute:\n{sql}")
        #     confirm = input("Confirm? (yes/no): ")
        #     if confirm.lower() != "yes":
        #         return "Operation cancelled by user."

        # Multiple statements — use executescript
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        if len(statements) > 1:
            conn.executescript(sql)
            return f"Success. {len(statements)} statements executed."

        cursor.execute(sql)

        # Commit and report rows affected for write operations
        if sql.strip().upper().startswith(("INSERT", "UPDATE", "DELETE")):
            conn.commit()
            affected = cursor.rowcount
            return f"Success. {affected} row(s) affected."

        # Fetch results for SELECT
        rows = cursor.fetchall()
        cols = [d[0] for d in cursor.description] if cursor.description else []
        if not rows:
            return "No results."
        return "\n".join([str(dict(zip(cols, row))) for row in rows[:100]])

    except Exception as e:
        return f"ERROR: {str(e)}"
    finally:
        if conn:
            conn.close()
        


def get_db_statistics(db_path: str, schema: str) -> str:
    """Returns per-table row counts and per-column distinct counts and min/max ranges."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    tables = [
        line.split("(")[0].replace("CREATE TABLE", "").strip()
        for line in schema.split("\n")
        if line.strip().upper().startswith("CREATE TABLE")
    ]

    stats = []
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) FROM \"{table}\"")
            row_count = cursor.fetchone()[0]
            stats.append(f"Table '{table}': {row_count} rows")

            cursor.execute(f"PRAGMA table_info(\"{table}\")")
            columns = cursor.fetchall()
            for col in columns:
                col_name, col_type = col[1], col[2]
                try:
                    cursor.execute(f"SELECT COUNT(DISTINCT \"{col_name}\") FROM \"{table}\"")
                    distinct = cursor.fetchone()[0]
                    if any(t in col_type.upper() for t in ("INT", "REAL", "NUMERIC", "FLOAT", "DOUBLE")):
                        cursor.execute(f"SELECT MIN(\"{col_name}\"), MAX(\"{col_name}\") FROM \"{table}\"")
                        mn, mx = cursor.fetchone()
                        stats.append(f"  {col_name}: {distinct} distinct values, range [{mn}, {mx}]")
                    else:
                        cursor.execute(f"SELECT \"{col_name}\" FROM \"{table}\" GROUP BY \"{col_name}\" ORDER BY COUNT(*) DESC LIMIT 5")
                        samples = [str(r[0]) for r in cursor.fetchall()]
                        stats.append(f"  {col_name}: {distinct} distinct values, samples: {samples}")
                except Exception:
                    pass
        except Exception:
            pass

    conn.close()
    return "\n".join(stats)


def get_query_plan(db_path: str, query: str) -> str:
    """Returns the SQLite EXPLAIN QUERY PLAN output for a given query."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute(f"EXPLAIN QUERY PLAN {query}")
        plan = cursor.fetchall()
        plan_str = "\n".join([str(row) for row in plan])
    except Exception as e:
        plan_str = f"Could not retrieve query plan: {e}"
    conn.close()
    return plan_str


def get_query_execution_time(db_path: str, query: str) -> float:
    """Runs a query and returns execution time in milliseconds."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    start = time.time()
    try:
        cursor.execute(query)
        cursor.fetchall()
    except Exception:
        pass
    elapsed = (time.time() - start) * 1000
    conn.close()
    return round(elapsed, 3)


def build_metrics(db_path: str, slow_query: str) -> dict:
    """
    Combines simulated CPU/memory metrics with real SQLite query data.
    In production, CPU and memory would come from system monitoring tools.
    """
    return {
        "cpu_usage_percent": 89,        # simulated
        "memory_usage_percent": 78,     # simulated
        "active_connections": 42,       # simulated
        "query_time_ms": get_query_execution_time(db_path, slow_query),
        "query_plan": get_query_plan(db_path, slow_query),
        "db_size_mb": 1.2               # simulated
    }


def create_test_db(db_path: str = "test.db"):
    """Creates the test SQLite database with students, courses, enrollments."""
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        DROP TABLE IF EXISTS students;
        DROP TABLE IF EXISTS courses;
        DROP TABLE IF EXISTS enrollments;

       CREATE TABLE students (
    student_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    age INTEGER,
    city TEXT,
    gpa REAL
);

CREATE TABLE enrollments (
    enrollment_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course_id INTEGER,
    grade TEXT,
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

        CREATE TABLE courses (
            course_id INTEGER PRIMARY KEY,
            course_name TEXT,
            department TEXT,
            credits INTEGER
        );

        

        INSERT INTO students VALUES
        (1,'Ahmed Hassan',20,'Cairo',3.8),
        (2,'Sara Ali',22,'Alexandria',3.5),
        (3,'Mohamed Khaled',21,'Cairo',3.2),
        (4,'Nour Ibrahim',23,'Giza',3.9),
        (5,'Youssef Mahmoud',20,'Cairo',2.8);

        INSERT INTO courses VALUES
        (1,'Database Systems','CS',3),
        (2,'Machine Learning','CS',3),
        (3,'Data Structures','CS',3),
        (4,'Calculus','Math',4);

        INSERT INTO enrollments VALUES
        (1,1,1,'A'),(2,1,2,'B'),
        (3,2,1,'A'),(4,3,3,'C'),
        (5,4,2,'A'),(6,4,1,'A'),
        (7,5,4,'B');
    """)
    conn.commit()
    conn.close()
    print(f"Test database created at {db_path}")