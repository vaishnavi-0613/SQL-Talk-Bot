import os, re, time, datetime
from langchain_community.utilities import SQLDatabase
from langchain_groq import ChatGroq
from conβig import GROQ_API_KEY, MODEL_NAME

FALLBACK_MODELS = [MODEL_NAME, "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-
32768"]

MAX_RETRIES = 3
SYSTEM_PROMPT = """You are a helpful data analyst with access to a SQLite database.
Schema: {schema}
Instructions:
1. Write SQL inside ```sql ... ``` blocks only.
2. Use only tables/columns from the schema.
3. After the SQL block write: **Answer:** <one-sentence plain English description>
4. If question can't be answered from schema, say so clearly."""
CORRECTION_PROMPT = """You are an expert SQLite debugger.
Schema: {schema} | Question: {question}
Failed SQL: ```sql {failed_sql} ``` | Error: {error}
Write ONLY the corrected SQL inside ```sql ... ``` block."""
SUMMARY_PROMPT = """You are a data analyst. Summarise these SQL results in plain English.
No SQL code or markdown tables in your reply.
Question: {question}
Results: {results}"""
def _llm(model):
return ChatGroq(api_key=GROQ_API_KEY, model_name=model,
temperature=0, max_tokens=1024, request_timeout=25)
def _extract_sql(text):

27
found, seen = [], set()
KW =
("SELECT","INSERT","UPDATE","DELETE","WITH","CREATE","DROP","ALTER","PRAGMA")
for m in re.βinditer(r'```(?:sql)?\s*(.*?)```', text, re.DOTALL|re.IGNORECASE):
s = m.group(1).strip()
if any(s.upper().startswith(k) for k in KW):
key = re.sub(r"\s+"," ",s).lower()
if key not in seen: seen.add(key); found.append(s)
if not found:
for m in re.βinditer(r'((?:SELECT|INSERT|UPDATE|DELETE|WITH)\b[^;`]{8,})',
text, re.IGNORECASE|re.DOTALL):
s = m.group(1).strip()
key = re.sub(r"\s+"," ",s).lower()
if key not in seen and len(s)>10: seen.add(key); found.append(s)
return found
def _clean(text):
text = re.sub(r'```[\w]*.*?```','',text,βlags=re.DOTALL)
text = re.sub(r'<[^>]+>','',text)
m = re.search(r'\*\*Answer:\*\*\s*(.*)', text, re.DOTALL|re.IGNORECASE)
return (m.group(1) if m else text).strip()
def _is_error(r):
return any(k in r.lower() for k in ("operationalerror","no such table",
"no such column","syntax error","ambiguous column","error:"))
def _is_rate_limit(e):
return "429" in str(e) or "rate_limit_exceeded" in str(e)
def _run_with_correction(db, sql, schema, question, model):
corrections, current = [], sql
for attempt in range(1, MAX_RETRIES+1):
try: result = db.run(current)
except Exception as e: result = f"Error: {e}"
if not _is_error(str(result)):
return current, result, corrections
if attempt == MAX_RETRIES:
corrections.append({"attempt":attempt,"failed_sql":current,
"error":str(result),"note":"Max retries reached"})
return current, result, corrections
try:
βix_raw = _llm(model).invoke([
{"role":"system","content":CORRECTION_PROMPT.format(
schema=schema,question=question,failed_sql=current,error=result)},
{"role":"user","content":"Provide corrected SQL."}]).content
βixed = _extract_sql(βix_raw)
corrected = βixed[0] if βixed else current
except: corrected = current
corrections.append({"attempt":attempt,"failed_sql":current,
"error":str(result),"corrected_sql":corrected})
current = corrected

28
return current, result, corrections
# ── Public API
def build_agent(db_path):
db = SQLDatabase.from_uri(f"sqlite:///{db_path}")
return db, db
def query_agent(agent, question):
db = agent
try: schema = db.get_table_info()
except Exception as e:
return {"answer":None,"steps":[],"error":f"Schema error: {e}","corrections":[]}
for model in FALLBACK_MODELS:
try:
raw = _llm(model).invoke([
{"role":"system","content":SYSTEM_PROMPT.format(schema=schema)},
{"role":"user","content":question}]).content
sqls = _extract_sql(raw)
if not sqls:
return {"answer":_clean(raw),"steps":[],"error":None,"corrections":[]}
steps, all_corrections, results = [], [], []
for sql in sqls:
βinal_sql, result, corrections = _run_with_correction(
db, sql, schema, question, model)
all_corrections.extend(corrections)
results.append((βinal_sql, result))
steps.append((type("A",(),{"tool_input":βinal_sql,"log":""})(), result))
results_block = "\n\n".join(f"SQL:\n{s}\n\nResult:\n{r}" for s,r in results)
answer = _clean(_llm(model).invoke([
{"role":"system","content":SUMMARY_PROMPT.format(
question=question, results=results_block)},
{"role":"user","content":"Give a clear plain-English answer."}]).content)
return {"answer":answer or "Query completed.","steps":steps,
"error":None,"corrections":all_corrections}
except Exception as e:
if _is_rate_limit(e) and FALLBACK_MODELS.index(model) < len(FALLBACK_MODELS)-1:
time.sleep(3); continue
return {"answer":None,"steps":[],"error":str(e),"corrections":[]}
return {"answer":None,"steps":[],"error":"All models failed.","corrections":[]}
def generate_sql_for_prompt(db, prompt):
try: schema = db.get_table_info()
except Exception as e: return {"sql":[],"explanation":"","error":str(e)}
for model in FALLBACK_MODELS:
try:
raw = _llm(model).invoke([
{"role":"system","content":SYSTEM_PROMPT.format(schema=schema)},
{"role":"user","content":f"Question: {prompt}\n\nSQL:"}]).content
sqls = _extract_sql(raw)
return {"sql":sqls,"explanation":_clean(raw),"error":None} if sqls \
else {"sql":[],"explanation":raw.strip(),"error":"No SQL found."}
