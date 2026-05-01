import streamlit as st
import tempβile, os, datetime, re as _re
from agent import build_agent, query_agent
from converter import convert_to_sqlite, is_supported, get_βile_label, SUPPORTED_TYPES
st.set_page_conβig(page_title="SQL Talk Bot", page_icon="🗄", layout="wide")
for k, v in {"agent":None,"db":None,"messages":[],"db_name":None,"schema_info":None,
"temp_db_path":None,"βile_stats":None,"βile_type_label":None,
"sql_history":[],"sql_always_visible":True}.items():
if k not in st.session_state: st.session_state[k] = v
def get_schema(db):
schema = {}
try:
for t in db.get_table_names():
info = db.get_table_info([t])
cols = [l.split()[0].strip('",') for l in info.splitlines()
if l.strip() and not l.upper().startswith(("CREATE",")","/","--"))]
schema[t] = [c for c in cols if c]
except: pass
return schema
def clean(text):
text = _re.sub(r'```[\w]*.*?```','',text,βlags=_re.DOTALL)
text = _re.sub(r'<[^>]+>','',text)
return _re.sub(r'\n{3,}','\n\n',text).strip()
# ── Sidebar
with st.sidebar:
st.markdown("## 🗄 SQL Talk Bot")
if st.button("🟢 SQL Always Visible" if st.session_state.sql_always_visible
else "⬜ SQL: Click to Reveal", use_container_width=True):
st.session_state.sql_always_visible = not st.session_state.sql_always_visible
st.rerun()
st.divider()
uploaded = st.βile_uploader("Upload data βile",
type=list(SUPPORTED_TYPES.keys()))
if uploaded and uploaded.name != st.session_state.db_name:
if st.session_state.temp_db_path and os.path.exists(st.session_state.temp_db_path):
try: os.remove(st.session_state.temp_db_path)
except: pass
st.session_state.messages = []
with st.spinner("Loading..."):
try:
db_path, tables, counts = convert_to_sqlite(uploaded)
agent, db = build_agent(db_path)

25
schema = get_schema(db)
st.session_state.update({"agent":agent,"db":db,"db_name":uploaded.name,
"temp_db_path":db_path,"schema_info":schema,
"βile_type_label":get_βile_label(uploaded.name),
"βile_stats":{"tables":len(tables),"rows":sum(counts.values()),
"cols":sum(len(c) for c in schema.values())}})
st.success(f"✅ Ready: {uploaded.name}")
except Exception as e:
st.error(f"❌ {e}"); st.session_state.agent = None
if st.session_state.schema_info:
st.divider(); st.markdown("### 📊 Schema")
for t, cols in st.session_state.schema_info.items():
with st.expander(f"📋 {t}"):
[st.markdown(f"• `{c}`") for c in cols]
if st.session_state.agent:
st.divider()
if st.button("🗑 Clear Chat", use_container_width=True):
st.session_state.messages = []; st.rerun()
# ── Main
st.title("🗄 SQL Talk Bot")
st.caption("Powered by **Llama-3** · **LangChain** · **Groq**")
st.divider()
if not st.session_state.agent:
st.info("📂Upload a data βile from the sidebar to get started.\n\n"
"Supported: `.db` · `.xlsx` · `.csv` · `.json` · `.parquet`")
st.stop()
s = st.session_state.βile_stats
c1,c2,c3 = st.columns(3)
c1.metric("Tables", s["tables"]); c2.metric("Rows", s["rows"]); c3.metric("Cols", s["cols"])
st.divider()
# ── Chat history
for msg in st.session_state.messages:
with st.chat_message(msg["role"]):
st.write(clean(msg["content"]))
if msg.get("sql") and st.session_state.sql_always_visible:
st.code(msg["sql"], language="sql")
if msg.get("query_result"):
try:
import ast, pandas as pd
parsed = ast.literal_eval(msg["query_result"])
if isinstance(parsed, list) and parsed:
st.caption("📊 Results"); st.dataframe(pd.DataFrame(parsed),
use_container_width=True)
except: st.text(msg["query_result"])
# ── Input

26
if user_input := st.chat_input(f"Ask about {st.session_state.db_name}..."):
st.session_state.messages.append({"role":"user","content":user_input})
with st.spinner("🧠 Querying..."):
result = query_agent(st.session_state.agent, user_input)
answer = f"⚠ {result['error']}" if result["error"] else clean(result["answer"] or "Done.")
sql = next((s for step in result.get("steps",[])
if hasattr(step[0],"tool_input")
for s in [step[0].tool_input] if isinstance(s,str)), None) if result.get("steps") else None
query_result = next((str(step[1]) for step in reversed(result.get("steps",[]))
if isinstance(step,tuple)), None)
now = datetime.datetime.now().strftime("%H:%M:%S")
if sql: st.session_state.sql_history.append({"question":user_input,"sql":sql,"time":now})
st.session_state.messages.append({"role":"assistant","content":answer,
"sql":sql,"query_result":query_result})
st.rerun()
