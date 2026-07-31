import os
import re
import time
import datetime
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from config import GROQ_API_KEY, MODEL_NAME

# ── LangSmith setup ───────────────────────────────────────────────────────────
os.environ.setdefault("LANGCHAIN_TRACING_V2", os.getenv("LANGCHAIN_TRACING_V2", "false"))
os.environ.setdefault("LANGCHAIN_API_KEY",     os.getenv("LANGCHAIN_API_KEY", ""))
os.environ.setdefault("LANGCHAIN_PROJECT",     os.getenv("LANGCHAIN_PROJECT", "sql-talk-bot"))
os.environ.setdefault("LANGCHAIN_ENDPOINT",    os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"))

_LS_ENABLED = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

def _get_ls_client():
    if not _LS_ENABLED:
        return None
    try:
        from langsmith import Client
        return Client()
    except Exception:
        return None


# ── Constants ─────────────────────────────────────────────────────────────────
MAX_CORRECTION_ATTEMPTS = 3

FALLBACK_MODELS = [
    MODEL_NAME,
    "llama-3.1-8b-instant",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
]

# ── Prompt templates ──────────────────────────────────────────────────────────
_CHAT_SYSTEM = """You are a helpful data analyst assistant.
You have access to a SQLite database with the following schema:

{schema}

When the user asks a question:
1. Write the SQL query that answers it inside a ```sql ... ``` block.
2. Format your final answer as:

```sql
SELECT ...
```
**Answer:** <one-sentence plain English description of what the query returns>

Rules:
- Use only tables and columns from the schema above.
- Keep queries simple and efficient.
- If the question cannot be answered from the schema, say so clearly.
"""

_CORRECTION_SYSTEM = """You are an expert SQLite developer and debugger.
A SQL query failed with an error. Your job is to fix it.

Database schema:
{schema}

Original question: {question}

Failed SQL:
```sql
{failed_sql}
```

Error message:
{error}

Write a corrected SQL query inside a ```sql ... ``` block.
Only output the fixed SQL — no explanation needed.
"""

_SQL_ONLY_SYSTEM = """You are an expert SQLite developer.
Given a database schema and a question, write the correct SQL query.

Schema:
{schema}

Rules:
1. Always wrap SQL in a ```sql ... ``` code block.
2. After the block write a one-sentence plain-English explanation.
3. Use only tables/columns from the schema.
4. Do NOT execute the query — just write it.
"""

_SUMMARY_SYSTEM = """You are a helpful data analyst.
The user asked a question and we ran SQL queries against their database.
Summarise the results clearly in plain English.
Do NOT include SQL code or markdown tables in your reply.
If the result is tabular data, briefly describe what the rows represent,
then say the full data is shown in the table below.
"""


# ── LangSmith side-car tracer ─────────────────────────────────────────────────
class _Tracer:
    """
    Logs the complete reasoning trace to LangSmith as a single Run with
    structured child events. Gracefully no-ops when tracing is disabled
    or the langsmith SDK is not installed — the bot never crashes because
    of observability code.
    """

    def __init__(self, question: str, db_name: str):
        self._client   = _get_ls_client()
        self._run_id   = None
        self._question = question
        self._db_name  = db_name
        self._events   = []
        self._start    = datetime.datetime.utcnow()

    def _ts(self) -> str:
        return datetime.datetime.utcnow().isoformat()

    def start(self):
        """Open the parent Run on LangSmith."""
        if not self._client:
            return
        try:
            import uuid
            self._run_id = str(uuid.uuid4())
            self._client.create_run(
                id=self._run_id,
                name="sql-talk-bot: query",
                run_type="chain",
                project_name=os.getenv("LANGCHAIN_PROJECT", "sql-talk-bot"),
                inputs={"question": self._question, "db": self._db_name},
                start_time=self._start,
                tags=["sql-talk-bot", "query_agent"],
            )
        except Exception:
            self._run_id = None

    def log(self, event: str, data: dict = None):
        entry = {"ts": self._ts(), "event": event, **(data or {})}
        self._events.append(entry)

    def log_sql_attempt(self, attempt: int, sql: str, result: str, error: str = None):
        self.log(f"sql_attempt_{attempt}",
                 {"sql": sql, "result_preview": str(result)[:300], "error": error})

    def log_correction(self, attempt: int, failed_sql: str, error: str, corrected_sql: str):
        self.log(f"self_correction_{attempt}",
                 {"failed_sql": failed_sql, "error": error, "corrected_sql": corrected_sql})

    def log_model_fallback(self, from_model: str, to_model: str, reason: str):
        self.log("model_fallback", {"from": from_model, "to": to_model, "reason": reason})

    def finish(self, answer: str, error: str = None):
        """Close the Run on LangSmith with final outputs."""
        if not self._client or not self._run_id:
            return
        try:
            self._client.update_run(
                self._run_id,
                outputs={"answer": answer, "error": error, "trace_events": self._events},
                end_time=datetime.datetime.utcnow(),
                error=error,
            )
        except Exception:
            pass

    @property
    def events(self) -> list:
        return self._events


# ── Core helpers ──────────────────────────────────────────────────────────────
def _build_llm(model_name: str) -> ChatGroq:
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=model_name,
        temperature=0,
        max_tokens=1024,
        request_timeout=25,
    )


def _extract_sql(text: str) -> list:
    """
    Pass 1: fenced ```sql``` blocks (preferred).
    Pass 2: bare-regex fallback ONLY if pass 1 found nothing.
    Final:  normalise + deduplicate so the same query never appears twice.
    """
    found, seen = [], set()
    SQL_KW = ("SELECT","INSERT","UPDATE","DELETE","WITH","CREATE","DROP","ALTER","PRAGMA")

    for m in re.finditer(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL | re.IGNORECASE):
        s = m.group(1).strip()
        if any(s.upper().startswith(k) for k in SQL_KW):
            key = re.sub(r"\s+", " ", s).lower()
            if key not in seen:
                seen.add(key)
                found.append(s)

    if not found:
        for m in re.finditer(
            r'((?:SELECT|INSERT|UPDATE|DELETE|WITH)\b[^;`]{8,})',
            text, re.IGNORECASE | re.DOTALL
        ):
            s = m.group(1).strip()
            key = re.sub(r"\s+", " ", s).lower()
            if key not in seen and len(s) > 10:
                seen.add(key)
                found.append(s)

    final, final_seen = [], set()
    for s in found:
        key = re.sub(r"\s+", " ", s).lower().rstrip(";").strip()
        if key not in final_seen:
            final_seen.add(key)
            final.append(s)

    return final


def _clean_answer(text: str) -> str:
    text = re.sub(r'```[\w]*.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'(?m)^\|.*\|\s*$', '', text)
    text = re.sub(r'(?im)^here is the (full )?data:?\s*$', '', text)
    m = re.search(r'\*\*Answer:\*\*\s*(.*)', text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def _is_rate_limit(e: Exception) -> bool:
    return "429" in str(e) or "rate_limit_exceeded" in str(e)


def _try_llm(model: str, messages: list) -> str:
    llm = _build_llm(model)
    response = llm.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def _is_exec_error(result: str) -> bool:
    lowered = result.lower().strip()
    return any(kw in lowered for kw in (
        "operationalerror", "no such table", "no such column",
        "syntax error", "ambiguous column", "error:"
    ))


# ── Self-correction engine ────────────────────────────────────────────────────
def _execute_with_correction(db, sql, schema, question, model, tracer, offset=0):
    """
    Execute `sql`. On failure, ask the LLM to fix it and retry up to
    MAX_CORRECTION_ATTEMPTS times.  Returns (final_sql, result_str, correction_log).
    """
    correction_log = []
    current_sql    = sql

    for attempt in range(1, MAX_CORRECTION_ATTEMPTS + 1):
        try:
            result = db.run(current_sql)
        except Exception as db_exc:
            result = f"Error: {db_exc}"

        if not _is_exec_error(str(result)):
            tracer.log_sql_attempt(offset + attempt, current_sql, result)
            return current_sql, result, correction_log

        # ── Execution failed — self-correct ───────────────────────────────
        error_msg = str(result)
        tracer.log(f"exec_error_{offset + attempt}", {"sql": current_sql, "error": error_msg})

        if attempt == MAX_CORRECTION_ATTEMPTS:
            correction_log.append({
                "attempt": attempt, "failed_sql": current_sql,
                "error": error_msg, "corrected_sql": None,
                "note": "Max correction attempts reached",
            })
            return current_sql, error_msg, correction_log

        # Ask LLM to fix it
        fix_messages = [
            {
                "role": "system",
                "content": _CORRECTION_SYSTEM.format(
                    schema=schema, question=question,
                    failed_sql=current_sql, error=error_msg,
                ),
            },
            {"role": "user", "content": "Please provide the corrected SQL."},
        ]
        try:
            fix_raw       = _try_llm(model, fix_messages)
            fix_sqls      = _extract_sql(fix_raw)
            corrected_sql = fix_sqls[0] if fix_sqls else current_sql
        except Exception:
            corrected_sql = current_sql

        tracer.log_correction(offset + attempt, current_sql, error_msg, corrected_sql)
        correction_log.append({
            "attempt": attempt, "failed_sql": current_sql,
            "error": error_msg, "corrected_sql": corrected_sql,
        })
        current_sql = corrected_sql

    return current_sql, result, correction_log


# ── Public API ────────────────────────────────────────────────────────────────
def build_agent(db_path: str):
    db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
    return db, db


def query_agent(agent, question: str) -> dict:
    """
    Full pipeline with self-correction + LangSmith side-car tracing:
      1. LLM generates SQL
      2. Execute with autonomous correction loop (≤ MAX_CORRECTION_ATTEMPTS retries)
      3. LLM summarises real DB results in plain English
      4. Every step logged to LangSmith
    """
    db = agent
    try:
        schema = db.get_table_info()
    except Exception as e:
        return {"answer": None, "steps": [], "error": f"Could not read schema: {e}",
                "corrections": [], "trace_events": []}

    db_name = str(getattr(getattr(db, "_engine", None), "url", "database"))

    tracer = _Tracer(question, db_name)
    tracer.start()
    tracer.log("query_start", {"question": question})

    messages = [
        {"role": "system", "content": _CHAT_SYSTEM.format(schema=schema)},
        {"role": "user",   "content": question},
    ]

    last_error      = None
    all_corrections = []

    for model in FALLBACK_MODELS:
        try:
            tracer.log("llm_call", {"model": model, "phase": "sql_generation"})
            raw  = _try_llm(model, messages)
            sqls = _extract_sql(raw)
            tracer.log("sql_extracted", {"count": len(sqls), "queries": sqls})

            if not sqls:
                answer = _clean_answer(raw)
                tracer.log("no_sql_found", {"raw_preview": raw[:200]})
                tracer.finish(answer)
                return {
                    "answer": answer or "I could not generate a SQL query for that question.",
                    "steps": [], "error": None, "corrections": [],
                    "trace_events": tracer.events,
                }

            # ── Execute each SQL with self-correction ─────────────────────
            steps             = []
            execution_results = []

            for i, sql in enumerate(sqls):
                final_sql, result_str, correction_log = _execute_with_correction(
                    db, sql, schema, question, model, tracer,
                    offset=i * MAX_CORRECTION_ATTEMPTS,
                )
                all_corrections.extend(correction_log)
                execution_results.append((final_sql, result_str))
                steps.append((
                    type("A", (), {"tool_input": final_sql, "log": ""})(),
                    result_str,
                ))

            tracer.log("execution_complete", {
                "queries_run": len(execution_results),
                "corrections": len(all_corrections),
            })

            # ── Summarise real results ────────────────────────────────────
            results_block = "\n\n".join(
                f"SQL:\n{sql}\n\nResult:\n{res}"
                for sql, res in execution_results
            )
            summary_messages = [
                {"role": "system", "content": _SUMMARY_SYSTEM},
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n\n{results_block}\n\n"
                        "Please give a clear plain-English answer."
                    ),
                },
            ]

            tracer.log("llm_call", {"model": model, "phase": "summarisation"})
            try:
                answer = _clean_answer(_try_llm(model, summary_messages))
            except Exception:
                answer = _clean_answer(raw)

            tracer.log("answer_ready", {"answer_preview": answer[:200]})
            tracer.finish(answer)

            return {
                "answer":       answer or "Query completed.",
                "steps":        steps,
                "error":        None,
                "corrections":  all_corrections,
                "trace_events": tracer.events,
            }

        except Exception as e:
            last_error = e
            tracer.log("exception", {"model": model, "error": str(e)})
            if _is_rate_limit(e):
                idx = FALLBACK_MODELS.index(model)
                if idx < len(FALLBACK_MODELS) - 1:
                    tracer.log_model_fallback(model, FALLBACK_MODELS[idx + 1], "rate_limit")
                time.sleep(3)
                continue
            tracer.finish(None, error=str(e))
            return {"answer": None, "steps": [], "error": str(e),
                    "corrections": all_corrections, "trace_events": tracer.events}

    err = str(last_error) or "All models failed. Please try again."
    tracer.finish(None, error=err)
    return {"answer": None, "steps": [], "error": err,
            "corrections": all_corrections, "trace_events": tracer.events}


def generate_sql_for_prompt(db, prompt: str) -> dict:
    """SQL Log tab: generate SQL only (no execution)."""
    try:
        schema = db.get_table_info()
    except Exception as e:
        return {"sql": [], "explanation": "", "error": f"Could not read schema: {e}"}

    messages = [
        {"role": "system", "content": _SQL_ONLY_SYSTEM.format(schema=schema)},
        {"role": "user",   "content": f"Question: {prompt}\n\nSQL:"},
    ]

    last_error = None
    for model in FALLBACK_MODELS:
        try:
            raw  = _try_llm(model, messages)
            sqls = _extract_sql(raw)
            if sqls:
                return {"sql": sqls, "explanation": _clean_answer(raw), "error": None}
            return {"sql": [], "explanation": raw.strip(),
                    "error": "No SQL found. Try rephrasing."}
        except Exception as e:
            last_error = e
            if _is_rate_limit(e):
                time.sleep(3)
                continue
            return {"sql": [], "explanation": "", "error": str(e)}

    return {"sql": [], "explanation": "",
            "error": str(last_error) or "Rate limit on all models. Please wait."}