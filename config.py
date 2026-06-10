"""
config.py – Centralised Configuration [v2.4 — Unified Minimalist]
========================================================================
Simplified configuration for the standalone face search service.
"""

import os
import sys
from dotenv import load_dotenv

# ── Base directory ────────────────────────────────────────────────────────────

if getattr(sys, "frozen", False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

env_path = os.path.join(BASE_DIR, ".env")
load_dotenv(env_path)

# ── MS SQL Server ─────────────────────────────────────────────────────────────

MSSQL_SERVER     = os.getenv("MSSQL_SERVER",   "192.168.0.251,1433")
MSSQL_USER       = os.getenv("MSSQL_USER",     "sa")
MSSQL_PASSWORD   = os.getenv("MSSQL_PASSWORD", "sa@123")
MSSQL_DB         = os.getenv("MSSQL_DB",       "hospital_face")
MSSQL_DRIVER     = os.getenv("MSSQL_DRIVER",   "ODBC Driver 18 for SQL Server")
MSSQL_TRUST_CERT = os.getenv("MSSQL_TRUST_CERT", "yes")

# ── Face Recognition ──────────────────────────────────────────────────────────

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Detection resolution:
# Detection resolution:
# Increased to 640x640 for room monitoring (better for small/distant faces)
ARC_FACE_DET_SIZE_MONITOR = (640, 640)
ARC_FACE_DET_SIZE_ENROL   = (640, 640)

# Matching threshold
FAISS_COSINE_THRESHOLD = float(os.getenv("FAISS_COSINE_THRESHOLD", "0.50"))
DET_THRESHOLD          = float(os.getenv("DET_THRESHOLD",          "0.35"))

# HNSW Index Parameters
# Optimized for 10,000+ persons (1 embedding each = ~10k vectors)
# efSearch=64 gives >99% recall at this scale and is 2x faster than 128
HNSW_M           = int(os.getenv("HNSW_M",           "48"))
HNSW_EF_SEARCH   = int(os.getenv("HNSW_EF_SEARCH",   "64"))
HNSW_EF_CONSTRUCT= int(os.getenv("HNSW_EF_CONSTRUCT","400"))

# Multi-Embedding Per Person (Set to 1 for hospital single-photo upload)
MULTI_EMB_COUNT = int(os.getenv("MULTI_EMB_COUNT", "1"))

# Enrollment (Single photo mode)
ONBOARD_FRAMES = int(os.getenv("ONBOARD_FRAMES", "1"))

# ── Matching Logic Filters ────────────────────────────────────────────────────

# Optimized for room monitoring: lower min size and lower blur threshold
FACE_MIN_SIZE       = int(os.getenv("FACE_MIN_SIZE",       "30"))
BLUR_THRESHOLD      = float(os.getenv("BLUR_THRESHOLD",    "40.0"))
CONSENSUS_WINDOW    = int(os.getenv("CONSENSUS_WINDOW",    "6"))
CONSENSUS_THRESHOLD = int(os.getenv("CONSENSUS_THRESHOLD", "3"))

# ── RTSP Cameras ──────────────────────────────────────────────────────────────

# Support comma-separated URLs in .env for up to 4 cameras
_raw_urls = os.getenv("RTSP_URLS", os.getenv("RTSP_URL", ""))
RTSP_URLS = [u.strip() for u in _raw_urls.split(",") if u.strip()]

# If no URLs found, provide a placeholder list
if not RTSP_URLS:
    RTSP_URLS = ["rtsp://test:admin123@192.168.1.213:554/stream"]

RTSP_URL = RTSP_URLS[0]

# ── Service ───────────────────────────────────────────────────────────────────
LOG_COOLDOWN   = int(os.getenv("LOG_COOLDOWN",   "600"))
RETENTION_DAYS = int(os.getenv("RETENTION_DAYS", "1"))

# ── ROI ───────────────────────────────────────────────────────────────────────
ROI_TOP    = int(os.getenv("ROI_TOP",    "0"))
ROI_BOTTOM = int(os.getenv("ROI_BOTTOM", "100"))
ROI_LEFT   = int(os.getenv("ROI_LEFT",   "0"))
ROI_RIGHT  = int(os.getenv("ROI_RIGHT",  "100"))

API_HOST      = os.getenv("API_HOST",      "0.0.0.0")

API_PORT      = int(os.getenv("API_PORT",  "8001"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "1234")

# Re-expose some dicts for backward compatibility if needed, but they're empty
RTSP_CAMERAS = {"Default": RTSP_URL}
ENABLED_CAMERAS = {"Default": True}
SPEAKER_DEVICE_IDS = {}
EXTERNAL_API_URLS = {}
