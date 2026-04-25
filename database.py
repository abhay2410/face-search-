"""
face_search/database.py – MS SQL Connection Pool & CRUD
=======================================================
Thin, self-contained DB layer using pyodbc with thread-local connections.

Key design points:
  - Thread-local persistent ODBC connections (one per asyncio executor thread).
  - All blocking calls executed in asyncio's default ThreadPoolExecutor.
  - In-memory TTL cache for employee lookups (5-minute default).
  - Schema bootstrapped on first run (no external migration tool required).
  - `employee_code` is treated as the "employee_number" visible to the API.
"""

import asyncio
import logging
import threading
import time
from functools import partial, wraps
from typing import Dict, List, Optional

import numpy as np
import pyodbc

import config  # shared config (reads .env via face_search/config.py)

log = logging.getLogger("face_search.database")

# ──────────────────────────────────────────────────────────────────────────────
#  In-memory employee cache  (TTL = 5 minutes)
# ──────────────────────────────────────────────────────────────────────────────

_CACHE_TTL = 300.0  # seconds

_employee_cache: Dict[int, dict] = {}
_employee_cache_ts: Dict[int, float] = {}


def _cache_get(emp_id: int) -> Optional[dict]:
    if emp_id not in _employee_cache:
        return None
    if time.monotonic() - _employee_cache_ts.get(emp_id, 0.0) > _CACHE_TTL:
        _employee_cache.pop(emp_id, None)
        _employee_cache_ts.pop(emp_id, None)
        return None
    return _employee_cache[emp_id]


def _cache_set(emp_id: int, data: dict):
    _employee_cache[emp_id] = data
    _employee_cache_ts[emp_id] = time.monotonic()


def clear_cache(emp_id: Optional[int] = None):
    """Invalidate cache after enrollment/update. Pass None to clear all."""
    if emp_id is not None:
        _employee_cache.pop(emp_id, None)
        _employee_cache_ts.pop(emp_id, None)
    else:
        _employee_cache.clear()
        _employee_cache_ts.clear()


# ──────────────────────────────────────────────────────────────────────────────
#  Thread-local connection pool
# ──────────────────────────────────────────────────────────────────────────────

_tl = threading.local()


# Build connection string once at module load — avoids rebuilding on every call
_CONN_STR: str = (
    f"DRIVER={{{config.MSSQL_DRIVER}}};"
    f"SERVER={config.MSSQL_SERVER};"
    f"DATABASE={config.MSSQL_DB};"
    f"UID={config.MSSQL_USER};"
    f"PWD={config.MSSQL_PASSWORD};"
    f"TrustServerCertificate={config.MSSQL_TRUST_CERT};"
    "Encrypt=no;"
)

def _conn_str() -> str:
    return _CONN_STR


def _get_conn() -> pyodbc.Connection:
    """Return the thread-local ODBC connection; reconnect if stale."""
    conn = getattr(_tl, "conn", None)
    last_used = getattr(_tl, "last_used", 0)
    now = time.monotonic()
    
    if conn is None:
        conn = pyodbc.connect(_conn_str(), autocommit=True)
        _tl.conn = conn
        _tl.last_used = now
        return conn
        
    # Only ping if idle for more than 120 seconds to minimise unnecessary DB round-trips
    if now - last_used > 120.0:
        try:
            conn.cursor().execute("SELECT 1")
        except pyodbc.Error:
            log.warning("[DB] Thread-local connection lost — reconnecting.")
            try:
                conn.close()
            except Exception:
                pass
            conn = pyodbc.connect(_conn_str(), autocommit=True)
            _tl.conn = conn
            
    _tl.last_used = now
    return conn


def close_all_connections():
    """Best-effort: close the current thread's connection."""
    conn = getattr(_tl, "conn", None)
    if conn:
        try:
            conn.close()
        except Exception:
            pass
        _tl.conn = None


# ──────────────────────────────────────────────────────────────────────────────
#  Retry decorator
# ──────────────────────────────────────────────────────────────────────────────

def _db_retry(max_attempts: int = 3, delay: float = 2.0):
    def decorator(fn):
        @wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Exception = RuntimeError("Empty retry")
            for attempt in range(1, max_attempts + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    log.warning("[DB] %s attempt %d/%d failed: %s", fn.__name__, attempt, max_attempts, exc)
                    if attempt < max_attempts:
                        await asyncio.sleep(delay)
            raise last_exc
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
#  Serialisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _emb_to_bytes(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def _bytes_to_emb(raw: bytes) -> np.ndarray:
    return np.frombuffer(raw, dtype=np.float32).copy()


def _multi_embs_to_bytes(vecs: List[np.ndarray]) -> bytes:
    mat = np.vstack([v.astype(np.float32) for v in vecs])  # (N, 512)
    return mat.tobytes()


def _bytes_to_multi_embs(raw: bytes) -> np.ndarray:
    flat = np.frombuffer(raw, dtype=np.float32).copy()
    n = flat.shape[0] // 512
    return flat.reshape(n, 512)  # (N, 512)


# ──────────────────────────────────────────────────────────────────────────────
#  Schema bootstrap
# ──────────────────────────────────────────────────────────────────────────────

def _init_db_sync():
    """Create tables if they don't exist. Safe to call repeatedly."""
    conn = _get_conn()
    cur = conn.cursor()

    # employees table
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='employees' AND xtype='U')
            BEGIN
                CREATE TABLE employees (
                    id               INT IDENTITY(1,1) PRIMARY KEY,
                    name             NVARCHAR(255)    NOT NULL UNIQUE,
                    employee_code    NVARCHAR(100),
                    department       NVARCHAR(100),
                    embedding        VARBINARY(MAX),
                    embeddings_multi VARBINARY(MAX),
                    rf_card          NVARCHAR(100),
                    img_count        INT DEFAULT 0,
                    enrolled_at      DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            END
            ELSE
            BEGIN
                IF NOT EXISTS (
                    SELECT * FROM sys.columns
                    WHERE object_id = OBJECT_ID('employees') AND name = 'embeddings_multi'
                )
                BEGIN
                    ALTER TABLE employees ADD embeddings_multi VARBINARY(MAX)
                END
                IF NOT EXISTS (
                    SELECT * FROM sys.columns
                    WHERE object_id = OBJECT_ID('employees') AND name = 'employee_code'
                )
                BEGIN
                    ALTER TABLE employees ADD employee_code NVARCHAR(100)
                END
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] employees migration: %s", exc)

    # face_search_log table (lightweight search audit)
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='face_search_log' AND xtype='U')
            BEGIN
                CREATE TABLE face_search_log (
                    id              INT IDENTITY(1,1) PRIMARY KEY,
                    employee_id     INT,
                    employee_code   NVARCHAR(100),
                    confidence      FLOAT,
                    matched         BIT DEFAULT 0,
                    searched_at     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] face_search_log migration: %s", exc)

    # detection_history table (High-detail logs with images for the dashboard)
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='detection_history' AND xtype='U')
            BEGIN
                CREATE TABLE detection_history (
                    id              INT IDENTITY(1,1) PRIMARY KEY,
                    employee_id     INT,
                    employee_code   NVARCHAR(100),
                    name            NVARCHAR(255),
                    confidence      FLOAT,
                    base64_image    NVARCHAR(MAX),
                    detected_at     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] detection_history migration: %s", exc)

    # faiss_index table
    try:
        cur.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='faiss_index' AND xtype='U')
            BEGIN
                CREATE TABLE faiss_index (
                    id          INT PRIMARY KEY DEFAULT 1,
                    index_blob  VARBINARY(MAX),
                    updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            END
        """)
        conn.commit()
    except Exception as exc:
        log.warning("[DB] faiss_index migration: %s", exc)

    log.info("[DB] Schema ready.")


@_db_retry(max_attempts=5, delay=3.0)
async def init_db():
    """Async wrapper — initialise schema with retry."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _init_db_sync)


# ──────────────────────────────────────────────────────────────────────────────
#  Employee CRUD
# ──────────────────────────────────────────────────────────────────────────────

def _upsert_employee_sync(
    name: str,
    employee_code: str,
    embedding: np.ndarray,
    department: str = "",
    rf_card: str = "",
    num_images: int = 1,
    multi_embeddings: Optional[List[np.ndarray]] = None,
) -> int:
    """
    Insert or update an employee.

    Strategy:
    - If employee exists (matched by `name`): compute running weighted mean
      embedding and replace multi_embeddings with the new ones.
    - Returns the internal SQL `id` (used as FAISS index key).
    """
    raw = _emb_to_bytes(embedding)
    conn = _get_conn()
    cur = conn.cursor()

    cur.execute("SELECT id, embedding, img_count FROM employees WHERE name=?", (name,))
    row = cur.fetchone()

    if row:
        emp_id = row[0]
        old_raw = row[1]
        old_count = row[2] or 0

        if old_raw:
            old_emb = _bytes_to_emb(bytes(old_raw))
            combined = (old_emb * old_count) + (embedding * num_images)
            new_count = old_count + num_images
            new_emb = combined / new_count
            n = np.linalg.norm(new_emb)
            if n > 0:
                new_emb /= n
            raw = _emb_to_bytes(new_emb)
        else:
            new_count = num_images

        if multi_embeddings:
            cur.execute(
                "UPDATE employees SET embedding=?, embeddings_multi=?, employee_code=?, img_count=? WHERE id=?",
                (raw, _multi_embs_to_bytes(multi_embeddings), employee_code, new_count, emp_id),
            )
        else:
            cur.execute(
                "UPDATE employees SET embedding=?, employee_code=?, img_count=? WHERE id=?",
                (raw, employee_code, new_count, emp_id),
            )
    else:
        if multi_embeddings:
            cur.execute(
                "INSERT INTO employees (name, employee_code, department, rf_card, embedding, embeddings_multi, img_count) "
                "VALUES (?,?,?,?,?,?,?)",
                (name, employee_code, department, rf_card, raw, _multi_embs_to_bytes(multi_embeddings), num_images),
            )
        else:
            cur.execute(
                "INSERT INTO employees (name, employee_code, department, rf_card, embedding, img_count) "
                "VALUES (?,?,?,?,?,?)",
                (name, employee_code, department, rf_card, raw, num_images),
            )
        cur.execute("SELECT SCOPE_IDENTITY()")
        emp_id = int(cur.fetchone()[0])

    conn.commit()

    # Warm cache
    _cache_set(emp_id, {
        "id": emp_id,
        "name": name,
        "employee_code": employee_code,
        "department": department,
        "rf_card": rf_card,
    })

    return emp_id


@_db_retry(max_attempts=3, delay=2.0)
async def upsert_employee(
    name: str,
    employee_code: str,
    embedding: np.ndarray,
    department: str = "",
    rf_card: str = "",
    num_images: int = 1,
    multi_embeddings: Optional[List[np.ndarray]] = None,
) -> int:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        partial(
            _upsert_employee_sync,
            name, employee_code, embedding,
            department, rf_card, num_images, multi_embeddings,
        ),
    )


def _get_employee_by_id_sync(emp_id: int) -> Optional[dict]:
    cached = _cache_get(emp_id)
    if cached is not None:
        return cached

    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, name, employee_code, department, rf_card FROM employees WHERE id=?",
        (emp_id,),
    )
    row = cur.fetchone()
    if not row:
        return None
    cols = [c[0] for c in cur.description]
    res = dict(zip(cols, row))
    _cache_set(emp_id, res)
    return res


@_db_retry(max_attempts=3, delay=1.0)
async def get_employee_by_id(emp_id: int) -> Optional[dict]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_get_employee_by_id_sync, emp_id))


def _get_all_multi_embeddings_sync() -> list:
    """
    Return all employees with their embedding matrices.
    Used by vector_engine.rebuild_index_from_db() to build the FAISS index.

    Returns: List of {"id": int, "embeddings": np.ndarray shape (N, 512)}
    """
    log.info("[DB] Fetching all embeddings...")
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute("SELECT id, embedding, embeddings_multi FROM employees ORDER BY id")
    results = []
    for row in cur.fetchall():
        emp_id, raw_mean, raw_multi = row[0], row[1], row[2]
        if raw_multi:
            mat = _bytes_to_multi_embs(bytes(raw_multi))
            results.append({"id": emp_id, "embeddings": mat})
        elif raw_mean:
            vec = _bytes_to_emb(bytes(raw_mean)).reshape(1, 512)
            results.append({"id": emp_id, "embeddings": vec})
    log.info("[DB] Fetched %d employee embedding sets.", len(results))
    return results


@_db_retry(max_attempts=3, delay=1.0)
async def get_all_multi_embeddings() -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_all_multi_embeddings_sync)


def _log_search_sync(employee_id: Optional[int], employee_code: Optional[str], confidence: float, matched: bool):
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO face_search_log (employee_id, employee_code, confidence, matched) VALUES (?,?,?,?)",
            (employee_id, employee_code, confidence, 1 if matched else 0),
        )
        conn.commit()
    except Exception as exc:
        log.debug("[DB] Search log write failed (non-critical): %s", exc)


async def log_search(employee_id: Optional[int], employee_code: Optional[str], confidence: float, matched: bool):
    """Fire-and-forget audit log for each search."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        partial(_log_search_sync, employee_id, employee_code, confidence, matched),
    )


def _log_detection_sync(employee_id: int, employee_code: str, name: str, confidence: float, base64_image: str):
    """Internal sync logger for detailed detections with images."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO detection_history (employee_id, employee_code, name, confidence, base64_image) VALUES (?,?,?,?,?)",
            (employee_id, employee_code, name, confidence, base64_image),
        )
        conn.commit()
    except Exception as exc:
        log.error("[DB] Detection log write failed: %s", exc)


async def log_detection(employee_id: int, employee_code: str, name: str, confidence: float, base64_image: str):
    """Fire-and-forget detailed detection log with image."""
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        None,
        partial(_log_detection_sync, employee_id, employee_code, name, confidence, base64_image),
    )


def _get_recent_matches_sync(limit: int = 20) -> list:
    """Fetch latest high-detail detections from the dedicated history table."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("""
            SELECT TOP (?) 
                id, employee_id, employee_code, name, confidence, base64_image, detected_at
            FROM detection_history
            ORDER BY detected_at DESC
        """, (limit,))
        
        rows = cur.fetchall()
        results = []
        for r in rows:
            results.append({
                "id": r[0],
                "emp_id": r[1],
                "employee_number": r[2],
                "name": r[3],
                "confidence": r[4],
                "base64_image": r[5],
                "time": r[6].strftime("%H:%M:%S") if r[6] else "",
                "ts": r[6].timestamp() if r[6] else 0
            })
        return results
    except Exception as exc:
        log.error("[DB] Failed to fetch detection history: %s", exc)
        return []


async def get_recent_matches(limit: int = 20) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, partial(_get_recent_matches_sync, limit))


def _clear_old_detections_sync():
    """Delete detections from previous days to keep DB light."""
    try:
        conn = _get_conn()
        cur = conn.cursor()
        # Delete rows older than RETENTION_DAYS
        cur.execute(
            "DELETE FROM detection_history WHERE detected_at < DATEADD(day, -?, GETDATE())",
            (config.RETENTION_DAYS,)
        )
        count = cur.rowcount
        conn.commit()
        if count > 0:
            log.info("[DB] EOD Cleanup: Removed %d old detection records.", count)
    except Exception as exc:
        log.error("[DB] EOD Cleanup failed: %s", exc)


async def clear_old_detections_loop():
    """Background loop to clear old detections periodically."""
    log.info("[DB] Starting background cleanup loop...")
    while True:
        try:
            await clear_old_detections()
        except Exception as e:
            log.error("[DB] Cleanup loop error: %s", e)
        # Check every 6 hours
        await asyncio.sleep(21600)


async def clear_old_detections():
    """Async wrapper for EOD cleanup."""
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _clear_old_detections_sync)


# ──────────────────────────────────────────────────────────────────────────────
#  FAISS Index Blob Sync
# ──────────────────────────────────────────────────────────────────────────────

def _save_faiss_index_sync(blob: bytes):
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("IF EXISTS (SELECT 1 FROM faiss_index WHERE id=1) UPDATE faiss_index SET index_blob=?, updated_at=GETDATE() WHERE id=1 ELSE INSERT INTO faiss_index (id, index_blob) VALUES (1, ?)", (blob, blob))
        conn.commit()
    except Exception as exc:
        log.error("[DB] Failed to save FAISS index blob: %s", exc)


async def save_faiss_index(blob: bytes):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _save_faiss_index_sync, blob)


def _load_faiss_index_sync() -> Optional[bytes]:
    try:
        conn = _get_conn()
        cur = conn.cursor()
        cur.execute("SELECT index_blob FROM faiss_index WHERE id=1")
        row = cur.fetchone()
        return bytes(row[0]) if row and row[0] else None
    except Exception as exc:
        log.error("[DB] Failed to load FAISS index blob: %s", exc)
        return None


async def load_index_from_sql() -> Optional[bytes]:
    """Internal alias compatible with engine expectation."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _load_faiss_index_sync)

async def load_faiss_index() -> Optional[bytes]:
    """Public alias compatible with engine expectation."""
    return await load_index_from_sql()
