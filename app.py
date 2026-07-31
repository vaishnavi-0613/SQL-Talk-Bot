import streamlit as st
import tempfile
import os
import datetime
import re as _re
from agent import build_agent, query_agent
from converter import convert_to_sqlite, is_supported, get_file_label, SUPPORTED_TYPES

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SQL Talk Bot",
    page_icon="🗄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "agent":                None,
    "db":                   None,
    "messages":             [],
    "db_name":              None,
    "schema_info":          None,
    "temp_db_path":         None,
    "file_stats":           None,
    "file_type_label":      None,
    "sql_history":          [],
    "sql_expanded":         set(),
    "active_tab":           "chat",
    "sql_always_visible":   True,
    "log_generated_sql":    [],
    "log_generated_prompt": "",
    "log_generated_answer": "",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── LangSmith status ──────────────────────────────────────────────────────────
ls_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
ls_project  = os.getenv("LANGCHAIN_PROJECT", "sql-talk-bot")


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_schema_display(db) -> dict:
    schema = {}
    try:
        for table in db.get_table_names():
            info = db.get_table_info([table])
            cols = []
            for line in info.splitlines():
                line = line.strip()
                if line and not line.upper().startswith(("CREATE", ")", "/*", "--")):
                    col = line.split()[0].strip('",')
                    if col:
                        cols.append(col)
            schema[table] = cols
    except Exception:
        pass
    return schema


def cleanup_temp():
    if st.session_state.temp_db_path and os.path.exists(st.session_state.temp_db_path):
        try:
            os.remove(st.session_state.temp_db_path)
        except Exception:
            pass


def fmt_number(n: int) -> str:
    if n >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n/1_000:.1f}K"
    return str(n)


# ── SQL keyword highlighting (plain text, no HTML spans) ──────────────────────
SQL_KEYWORDS = [
    "SELECT","FROM","WHERE","JOIN","LEFT","RIGHT","INNER","OUTER","FULL",
    "ON","GROUP","BY","ORDER","HAVING","LIMIT","OFFSET","INSERT","INTO",
    "VALUES","UPDATE","SET","DELETE","CREATE","TABLE","DROP","ALTER",
    "AND","OR","NOT","IN","IS","NULL","AS","DISTINCT","COUNT","SUM",
    "AVG","MIN","MAX","CASE","WHEN","THEN","ELSE","END","UNION","ALL",
    "EXISTS","BETWEEN","LIKE","WITH","OVER","PARTITION","RANK","COALESCE",
]


def extract_all_sql(steps: list) -> list[str]:
    """Pull every SQL string from agent intermediate steps."""
    queries = []
    seen    = set()

    SQL_STARTERS = ("SELECT","INSERT","UPDATE","DELETE","CREATE",
                    "DROP","ALTER","WITH","EXPLAIN","PRAGMA")

    def _add(candidate: str):
        if not isinstance(candidate, str):
            return
        s = candidate.strip()
        s = _re.sub(r"^```[\w]*\n?", "", s).rstrip("`").strip()
        upper = s.upper()
        if any(upper.startswith(kw) for kw in SQL_STARTERS) and len(s) > 10:
            key = _re.sub(r"\s+", " ", s).lower()
            if key not in seen:
                seen.add(key)
                queries.append(s)

    def _scan_text(text: str):
        if not isinstance(text, str):
            return
        for m in _re.finditer(
            r'Action\s+Input\s*:\s*((?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|EXPLAIN|PRAGMA)\b[^\n]+(?:\n(?!Action|Observation|Thought)[^\n]*)*)',
            text, _re.IGNORECASE
        ):
            _add(m.group(1).strip())
        for m in _re.finditer(
            r'((?:SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|EXPLAIN|PRAGMA)\b[^;`]{8,})',
            text, _re.IGNORECASE | _re.DOTALL
        ):
            _add(m.group(1).strip())

    for step in steps:
        if isinstance(step, tuple) and len(step) == 2:
            action, _observation = step
            # Only read tool_input — agent.py sets log="" to avoid double-scan.
            # Never scan observation: it contains DB row data, not SQL.
            if hasattr(action, "tool_input"):
                ti = action.tool_input
                if isinstance(ti, dict):
                    for v in ti.values():
                        _add(v) if isinstance(v, str) else None
                elif isinstance(ti, str):
                    _add(ti)
        elif isinstance(step, str):
            _scan_text(step)

    return queries


def render_sql_panel(sql: str, db_name: str, query_index: int = 1,
                     total_queries: int = 1, timestamp: str = "",
                     panel_id: str = ""):
    """Render a SQL code block using native Streamlit components."""
    lines  = sql.strip().splitlines()
    n      = len(lines)

    # Header row
    header_parts = [f"🗄️ Generated SQL Query"]
    if total_queries > 1:
        header_parts.append(f"· Query {query_index}/{total_queries}")
    if timestamp:
        header_parts.append(f"· ⏱ {timestamp}")
    header_parts.append(f"· {n} line{'s' if n != 1 else ''}")
    header_parts.append(f"· DB: {db_name or 'database'}")
    header_parts.append("· dialect: SQLite · ✓ Executed")

    st.caption("  ".join(header_parts))
    st.code(sql.strip(), language="sql")


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🗄️ SQL Talk Bot")
    st.markdown("Talk to any data file in plain English.")
    st.divider()

    # SQL transparency toggle
    sql_vis_label = (
        "🟢  SQL Always Visible" if st.session_state.sql_always_visible
        else "⬜  SQL: Click to Reveal"
    )
    if st.button(sql_vis_label, key="sql_vis_toggle", use_container_width=True):
        st.session_state.sql_always_visible = not st.session_state.sql_always_visible
        st.rerun()

    st.divider()

    # ── LangSmith status ──────────────────────────────────────────────────────
    if ls_enabled:
        st.success(f"🔭 LangSmith Tracing: **ACTIVE**\n\nProject: `{ls_project}`")
        st.markdown("[Open Dashboard ↗](https://smith.langchain.com)")
    else:
        st.info(
            "🔭 LangSmith Tracing: **OFF**\n\n"
            "Add `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY` to `.env` to enable."
        )

    st.divider()

    # ── Supported formats ─────────────────────────────────────────────────────
    st.markdown("### 📁 Supported Formats")
    formats = [
        ("🗄️ SQLite",  ".db · .sqlite · .sqlite3"),
        ("📊 Excel",   ".xlsx · .xls · .xlsm"),
        ("📄 CSV",     ".csv"),
        ("📋 JSON",    ".json"),
        ("⚡ Parquet", ".parquet"),
    ]
    for label, exts in formats:
        st.markdown(f"**{label}** — `{exts}`")

    st.divider()

    # ── File uploader ─────────────────────────────────────────────────────────
    uploaded_file = st.file_uploader(
        "Upload your data file",
        type=list(SUPPORTED_TYPES.keys()),
        help="SQLite, Excel, CSV, JSON, or Parquet",
    )

    if uploaded_file:
        if uploaded_file.name != st.session_state.db_name:
            cleanup_temp()
            st.session_state.messages = []
            with st.spinner("🔄 Converting & loading…"):
                try:
                    if not is_supported(uploaded_file.name):
                        st.error("❌ Unsupported file type.")
                    else:
                        db_path, tables, counts = convert_to_sqlite(uploaded_file)
                        agent, db = build_agent(db_path)
                        total_rows = sum(counts.values())
                        schema     = get_schema_display(db)
                        total_cols = sum(len(c) for c in schema.values())

                        st.session_state.agent           = agent
                        st.session_state.db              = db
                        st.session_state.db_name         = uploaded_file.name
                        st.session_state.temp_db_path    = db_path
                        st.session_state.schema_info     = schema
                        st.session_state.file_type_label = get_file_label(uploaded_file.name)
                        st.session_state.file_stats      = {
                            "tables": len(tables),
                            "rows":   total_rows,
                            "cols":   total_cols,
                        }
                        st.success(f"✅ Ready: **{uploaded_file.name}**")
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    st.session_state.agent = None

    # ── Schema viewer ─────────────────────────────────────────────────────────
    if st.session_state.schema_info:
        st.divider()
        st.markdown("### 📊 Schema")
        for table, cols in st.session_state.schema_info.items():
            with st.expander(f"📋 {table}", expanded=False):
                for c in cols:
                    st.markdown(f"• `{c}`")

    if st.session_state.agent:
        st.divider()
        if st.button("🗑️  Clear Chat", key="clear_chat", use_container_width=True):
            st.session_state.messages    = []
            st.session_state.sql_history = []
            st.session_state.sql_expanded = set()
            st.rerun()
        st.divider()
        st.markdown("### 💡 Try asking")
        tips = [
            "Show me all tables",
            "How many rows in each table?",
            "First 5 rows of [table]",
            "Total sales by category",
            "Top 10 by [column]",
            "Find duplicates in [column]",
            "Average [col] grouped by [col]",
            "Rows where [col] > 1000",
        ]
        for tip in tips:
            st.markdown(f"› `{tip}`")


# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_badges = st.columns([3, 1])
with col_title:
    st.title("🗄️ SQL Talk Bot")
    st.caption("Powered by **Llama-3.3** · **LangChain** · **Groq** · **LangSmith**")
with col_badges:
    if ls_enabled:
        st.success(f"● LangSmith: {ls_project}")
    else:
        st.caption("○ LangSmith off")

st.divider()


# ── No file state ─────────────────────────────────────────────────────────────
if not st.session_state.agent:
    st.info(
        "**📂 No data file loaded**\n\n"
        "Upload your file from the sidebar to get started.\n\n"
        "Supported: `.db` · `.xlsx` · `.csv` · `.json` · `.parquet`\n\n"
        "*All processing is local — your data never leaves your machine.*"
    )
    st.stop()


# ── File info banner ──────────────────────────────────────────────────────────
if st.session_state.file_stats:
    s  = st.session_state.file_stats
    ft = st.session_state.file_type_label or ""
    fn = st.session_state.db_name or ""

    st.markdown(f"**📁 {fn}** &nbsp; `{ft}`")
    c1, c2, c3 = st.columns(3)
    c1.metric("Tables",  s["tables"])
    c2.metric("Rows",    fmt_number(s["rows"]))
    c3.metric("Columns", s["cols"])
    st.divider()


# ── Tab navigation ────────────────────────────────────────────────────────────
sql_count   = len(st.session_state.sql_history)
chat_active = st.session_state.active_tab == "chat"
log_pill    = f"  ({sql_count})" if sql_count else ""

tab_col1, tab_col2 = st.columns(2)
with tab_col1:
    if st.button("💬  Chat", key="tab_chat", use_container_width=True,
                 type="primary" if chat_active else "secondary"):
        st.session_state.active_tab = "chat"
        st.rerun()
with tab_col2:
    if st.button(f"🗃️  SQL Query Log{log_pill}", key="tab_log",
                 use_container_width=True,
                 type="primary" if not chat_active else "secondary"):
        st.session_state.active_tab = "sql_log"
        st.rerun()

st.write("")


# ══════════════════════════════════════════════════════════
# TAB A — CHAT
# ══════════════════════════════════════════════════════════
if st.session_state.active_tab == "chat":

    # Render chat history
    for idx, msg in enumerate(st.session_state.messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.write(msg["content"])
        else:
            # Sanitize answer
            safe_content = msg["content"]
            safe_content = _re.sub(r'<!--.*?-->', '', safe_content, flags=_re.DOTALL)
            safe_content = _re.sub(r'<(?!(?:br|strong|em|code|b|i)\b)[^>]+>', '', safe_content)

            with st.chat_message("assistant"):
                st.write(safe_content)

                # ── Show actual query result rows ─────────────────────────
                raw_result = msg.get("query_result")
                if raw_result and not str(raw_result).startswith("Execution error"):
                    try:
                        import ast, pandas as pd
                        parsed = ast.literal_eval(raw_result)
                        # LangChain db.run() returns a string like:
                        # "[(1, 'Alice', ...), (2, 'Bob', ...), ...]"
                        if isinstance(parsed, list) and parsed:
                            df = pd.DataFrame(parsed)
                            st.caption("📊 Query Results")
                            st.dataframe(df, use_container_width=True)
                        elif parsed:
                            st.caption("📊 Query Results")
                            st.write(parsed)
                    except Exception:
                        # Not parseable as Python — show as plain text
                        if raw_result.strip():
                            st.caption("📊 Query Results")
                            st.text(raw_result)

                sql_queries = msg.get("sql_queries") or (
                    [msg["sql"]] if msg.get("sql") else []
                )
                msg_ts = msg.get("timestamp", "")

                if sql_queries:
                    always_on = st.session_state.sql_always_visible

                    if always_on:
                        if len(sql_queries) > 1:
                            st.caption(f"🔍 {len(sql_queries)} SQL Queries Executed")
                        for qi, q in enumerate(sql_queries, 1):
                            render_sql_panel(
                                q,
                                st.session_state.db_name or "",
                                query_index=qi,
                                total_queries=len(sql_queries),
                                timestamp=msg_ts,
                                panel_id=str(idx),
                            )
                    else:
                        is_open = idx in st.session_state.sql_expanded
                        btn_lbl = "🔼  Hide SQL" if is_open else "🔽  View SQL Query"
                        tcol, _ = st.columns([3, 7])
                        with tcol:
                            if st.button(btn_lbl, key=f"sqlt_{idx}", use_container_width=True):
                                exp = st.session_state.sql_expanded
                                if idx in exp:
                                    exp.discard(idx)
                                else:
                                    exp.add(idx)
                                st.rerun()
                        if is_open:
                            for qi, q in enumerate(sql_queries, 1):
                                render_sql_panel(
                                    q,
                                    st.session_state.db_name or "",
                                    query_index=qi,
                                    total_queries=len(sql_queries),
                                    timestamp=msg_ts,
                                    panel_id=str(idx),
                                )

            # Self-correction log
            corrections = msg.get("corrections", [])
            if corrections:
                with st.expander(f"🔁 Self-Correction Log  ({len(corrections)} fix{'es' if len(corrections)!=1 else ''})"):
                    for c in corrections:
                        st.error(f"**Attempt {c['attempt']} failed:** {c['error']}")
                        st.caption("❌ Failed SQL")
                        st.code(c["failed_sql"], language="sql")
                        if c.get("corrected_sql"):
                            st.caption("✅ Corrected SQL")
                            st.code(c["corrected_sql"], language="sql")
                        elif c.get("note"):
                            st.warning(c["note"])

            # LangSmith trace events
            trace_events = msg.get("trace_events", [])
            if trace_events:
                with st.expander(f"🔭 LangSmith Trace  ({len(trace_events)} events)"):
                    for ev in trace_events:
                        label = ev.get("event", "event")
                        ts    = ev.get("ts", "")
                        data  = {k: v for k, v in ev.items() if k not in ("event", "ts")}
                        st.markdown(f"`{ts}`  **{label}**")
                        if data:
                            st.json(data)

    # ── Chat input ────────────────────────────────────────────────────────────
    user_input = st.chat_input(f"Ask something about {st.session_state.db_name}…")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.spinner("🧠 Querying your data…"):
            result = query_agent(st.session_state.agent, user_input)

        if result["error"]:
            answer      = f"⚠️ {result['error']}"
            sql_queries = []
            query_result = None
        else:
            raw_answer = result["answer"] or ""
            raw_answer = _re.sub(r'```[\w]*.*?```', '', raw_answer, flags=_re.DOTALL)
            raw_answer = _re.sub(r'<!--.*?-->', '',  raw_answer, flags=_re.DOTALL)
            raw_answer = _re.sub(r'<[^>]+>',    '',  raw_answer)
            raw_answer = _re.sub(r'\n{3,}', '\n\n',  raw_answer)
            answer     = raw_answer.strip() or "Done."

            # ── Extract SQL (deduplicated) — primary path only; no fallback
            # re-scan to avoid the same query appearing twice (once from
            # tool_input and once from the observation scan).
            sql_queries = extract_all_sql(result["steps"])

            # ── Extract the actual query result rows from the last observation
            query_result = None
            for step in reversed(result["steps"]):
                if isinstance(step, tuple) and len(step) == 2:
                    _, observation = step
                    obs_str = str(observation).strip()
                    # Observations that are DB rows look like lists/tuples of data
                    if obs_str and not obs_str.upper().startswith(("SELECT","INSERT",
                            "UPDATE","DELETE","CREATE","DROP","ALTER","WITH","ERROR")):
                        query_result = obs_str
                        break

        now = datetime.datetime.now().strftime("%H:%M:%S")
        for q in sql_queries:
            st.session_state.sql_history.append({
                "n":        len(st.session_state.sql_history) + 1,
                "question": user_input,
                "sql":      q,
                "time":     now,
                "db":       st.session_state.db_name or "",
            })

        new_idx = len(st.session_state.messages)
        st.session_state.messages.append({
            "role":         "assistant",
            "content":      answer,
            "sql":          sql_queries[0] if sql_queries else None,
            "sql_queries":  sql_queries,
            "query_result": query_result if not result["error"] else None,
            "corrections":  result.get("corrections", []),
            "trace_events": result.get("trace_events", []),
            "steps":        [],
            "timestamp":    now,
        })

        if sql_queries and not st.session_state.sql_always_visible:
            st.session_state.sql_expanded.add(new_idx)

        st.rerun()


# ══════════════════════════════════════════════════════════
# TAB B — SQL QUERY LOG
# ══════════════════════════════════════════════════════════
else:
    history = st.session_state.sql_history

    # ── Prompt → SQL generator ────────────────────────────────────────────────
    st.subheader("✨ Generate SQL from a Prompt")
    st.caption("Type a plain-English question — the SQL will be generated instantly below for you to review, copy, or save to the log.")

    log_prompt_col, log_btn_col = st.columns([5, 1])
    with log_prompt_col:
        log_prompt = st.text_input(
            label="log_prompt_input",
            placeholder="e.g. Show total sales grouped by category",
            label_visibility="collapsed",
            key="log_prompt_text",
        )
    with log_btn_col:
        log_generate = st.button("⚡ Generate", key="log_generate_btn",
                                 use_container_width=True)

    if log_generate and log_prompt.strip():
        if not st.session_state.db:
            st.warning("⚠️ Upload a data file first to enable SQL generation.")
        else:
            with st.spinner("🧠 Generating SQL for your prompt…"):
                from agent import generate_sql_for_prompt
                gen_result = generate_sql_for_prompt(
                    st.session_state.db, log_prompt.strip()
                )

            if gen_result["error"] and not gen_result["sql"]:
                st.error(f"❌ {gen_result['error']}")
                st.session_state.log_generated_sql    = []
                st.session_state.log_generated_prompt = ""
                st.session_state.log_generated_answer = ""
            else:
                st.session_state.log_generated_sql    = gen_result["sql"]
                st.session_state.log_generated_prompt = log_prompt.strip()
                st.session_state.log_generated_answer = gen_result["explanation"]
                st.rerun()

    elif log_generate and not log_prompt.strip():
        st.warning("⚠️ Please enter a prompt first.")

    # ── Live SQL code preview ─────────────────────────────────────────────────
    if st.session_state.log_generated_sql:
        gen_sqls   = st.session_state.log_generated_sql
        gen_prompt = st.session_state.log_generated_prompt
        gen_answer = st.session_state.log_generated_answer

        st.markdown("**🔍 Generated SQL — Review before saving**")

        if gen_answer:
            with st.container(border=True):
                st.caption("💬 Answer")
                st.write(gen_answer)

        for qi, sql in enumerate(gen_sqls, 1):
            n = len(sql.strip().splitlines())
            multi_lbl = f"Query {qi}/{len(gen_sqls)}" if len(gen_sqls) > 1 else ""
            label_parts = ["Generated SQL"]
            if multi_lbl:
                label_parts.append(multi_lbl)
            label_parts.append(f"{n} line{'s' if n != 1 else ''}")
            st.caption("  ·  ".join(label_parts))
            st.code(sql.strip(), language="sql")

        # Action buttons: Save to Log / Discard
        act_col1, act_col2 = st.columns([2, 1])
        with act_col1:
            if st.button("💾  Save to Log", key="save_to_log_btn",
                         use_container_width=True, type="primary"):
                now = datetime.datetime.now().strftime("%H:%M:%S")
                for q in gen_sqls:
                    st.session_state.sql_history.append({
                        "n":        len(st.session_state.sql_history) + 1,
                        "question": gen_prompt,
                        "sql":      q,
                        "time":     now,
                        "db":       st.session_state.db_name or "",
                        "source":   "log",
                    })
                st.session_state.log_generated_sql    = []
                st.session_state.log_generated_prompt = ""
                st.session_state.log_generated_answer = ""
                st.rerun()
        with act_col2:
            if st.button("🗑️  Discard", key="discard_gen_btn",
                         use_container_width=True):
                st.session_state.log_generated_sql    = []
                st.session_state.log_generated_prompt = ""
                st.session_state.log_generated_answer = ""
                st.rerun()

        st.divider()

    # ── Log entries ───────────────────────────────────────────────────────────
    if not history:
        st.info(
            "**🗃️ No SQL queries yet**\n\n"
            "Ask a question in the Chat tab or use the generator above — "
            "every SQL query generated will appear here automatically."
        )
    else:
        hc1, hc2 = st.columns([5, 1])
        with hc1:
            cnt = len(history)
            st.markdown(f"**🗃️ SQL Query Log** — {cnt} quer{'y' if cnt == 1 else 'ies'}")
        with hc2:
            if st.button("🗑️ Clear", key="clear_sql_log", use_container_width=True):
                st.session_state.sql_history = []
                st.rerun()

        # Render newest first
        for entry in reversed(history):
            source       = entry.get("source", "chat")
            source_label = "📝 Log" if source == "log" else "💬 Chat"
            n_lines      = len(entry["sql"].strip().splitlines())

            with st.container(border=True):
                top_left, top_right = st.columns([3, 1])
                with top_left:
                    st.markdown(
                        f"**#{entry['n']:02d}** &nbsp; `{source_label}` &nbsp; "
                        f"{entry['question']}"
                    )
                with top_right:
                    st.caption(f"⏱ {entry['time']}  ✓ Executed")

                st.code(entry["sql"].strip(), language="sql")
                st.caption(
                    f"🔗 {entry['db']}  ·  {n_lines} line{'s' if n_lines != 1 else ''}"
                )