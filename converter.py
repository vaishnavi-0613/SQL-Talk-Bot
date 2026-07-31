import os
import tempfile
import sqlite3
import pandas as pd


SUPPORTED_TYPES = {
    "db":      "SQLite Database",
    "sqlite":  "SQLite Database",
    "sqlite3": "SQLite Database",
    "csv":     "CSV Spreadsheet",
    "xlsx":    "Excel Workbook",
    "xls":     "Excel Workbook (Legacy)",
    "xlsm":    "Excel Macro Workbook",
    "json":    "JSON File",
    "parquet": "Parquet File",
}


def get_extension(filename: str) -> str:
    return os.path.splitext(filename)[-1].lstrip(".").lower()


def is_supported(filename: str) -> bool:
    return get_extension(filename) in SUPPORTED_TYPES


def get_file_label(filename: str) -> str:
    ext = get_extension(filename)
    return SUPPORTED_TYPES.get(ext, "Unknown File")


def convert_to_sqlite(uploaded_file) -> tuple[str, list[str], dict]:
    """
    Convert any supported file to a temporary SQLite .db file.
    Returns:
        - path to the temp SQLite file
        - list of table names created
        - dict of {table: row_count}
    """
    ext = get_extension(uploaded_file.name)
    raw_bytes = uploaded_file.read()

    # ── Already SQLite ────────────────────────────────────────────────────────
    if ext in ("db", "sqlite", "sqlite3"):
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
        tmp.write(raw_bytes)
        tmp.flush()
        tmp.close()
        # Introspect existing tables
        conn = sqlite3.connect(tmp.name)
        cur  = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [r[0] for r in cur.fetchall()]
        counts = {}
        for t in tables:
            cur.execute(f"SELECT COUNT(*) FROM [{t}]")
            counts[t] = cur.fetchone()[0]
        conn.close()
        return tmp.name, tables, counts

    # ── Build temp SQLite for tabular formats ─────────────────────────────────
    import io
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".db")
    tmp.close()
    conn   = sqlite3.connect(tmp.name)
    tables = []
    counts = {}

    try:
        # CSV ─────────────────────────────────────────────────────────────────
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(raw_bytes))
            table_name = _safe_name(uploaded_file.name)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            tables.append(table_name)
            counts[table_name] = len(df)

        # Excel (.xlsx / .xls / .xlsm) ────────────────────────────────────────
        elif ext in ("xlsx", "xls", "xlsm"):
            xf = pd.ExcelFile(io.BytesIO(raw_bytes))
            for sheet in xf.sheet_names:
                df = xf.parse(sheet)
                if df.empty:
                    continue
                table_name = _safe_name(sheet)
                df.to_sql(table_name, conn, if_exists="replace", index=False)
                tables.append(table_name)
                counts[table_name] = len(df)

        # JSON ────────────────────────────────────────────────────────────────
        elif ext == "json":
            df = pd.read_json(io.BytesIO(raw_bytes))
            # If nested, normalize one level
            if any(df.applymap(lambda x: isinstance(x, (dict, list))).any()):
                df = pd.json_normalize(df.to_dict(orient="records"))
            table_name = _safe_name(uploaded_file.name)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            tables.append(table_name)
            counts[table_name] = len(df)

        # Parquet ─────────────────────────────────────────────────────────────
        elif ext == "parquet":
            df = pd.read_parquet(io.BytesIO(raw_bytes))
            table_name = _safe_name(uploaded_file.name)
            df.to_sql(table_name, conn, if_exists="replace", index=False)
            tables.append(table_name)
            counts[table_name] = len(df)

        else:
            raise ValueError(f"Unsupported file type: .{ext}")

    finally:
        conn.close()

    return tmp.name, tables, counts


def _safe_name(filename: str) -> str:
    """Convert filename to a safe SQL table name."""
    name = os.path.splitext(os.path.basename(filename))[0]
    name = name.replace(" ", "_").replace("-", "_")
    # Strip non-alphanumeric except underscore
    name = "".join(c for c in name if c.isalnum() or c == "_")
    # Must not start with digit
    if name and name[0].isdigit():
        name = "t_" + name
    return name.lower() or "data"