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
MSSQL_DB         = os.getenv("MSSQL_DB",       "face_attendance")
MSSQL_DRIVER     = os.getenv("MSSQL_DRIVER",   "ODBC Driver 18 for SQL Server")
MSSQL_TRUST_CERT = os.getenv("MSSQL_TRUST_CERT", "yes")

# ── Face Recognition ──────────────────────────────────────────────────────────

EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "512"))

# Detection resolution:
# Detection resolution:
# Optimized for 2GB VRAM (640x640 is heavy for 4 cameras)
ARC_FACE_DET_SIZE_MONITOR = (480, 480)
ARC_FACE_DET_SIZE_ENROL   = (640, 640)

# Matching threshold
FAISS_COSINE_THRESHOLD = float(os.getenv("FAISS_COSINE_THRESHOLD", "0.60"))
DET_THRESHOLD          = float(os.getenv("DET_THRESHOLD",          "0.35"))

# HNSW Index Parameters
# Tuned for <1000 employees: M=16 + efSearch=32 gives >99% recall at this scale
# (Previous values M=48/efSearch=128/efConstruct=400 were for millions of vectors)
HNSW_M           = int(os.getenv("HNSW_M",           "16"))
HNSW_EF_SEARCH   = int(os.getenv("HNSW_EF_SEARCH",   "32"))
HNSW_EF_CONSTRUCT= int(os.getenv("HNSW_EF_CONSTRUCT","64"))

# Multi-Embedding Per Person
MULTI_EMB_COUNT = int(os.getenv("MULTI_EMB_COUNT", "3"))

# Enrollment
ONBOARD_FRAMES = int(os.getenv("ONBOARD_FRAMES", "20"))

# ── Matching Logic Filters ────────────────────────────────────────────────────

FACE_MIN_SIZE       = int(os.getenv("FACE_MIN_SIZE",       "60"))
BLUR_THRESHOLD      = float(os.getenv("BLUR_THRESHOLD",    "45.0"))
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

API_HOST      = os.getenv("API_HOST",      "0.0.0.0")
API_PORT      = int(os.getenv("API_PORT",  "8001"))
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "admin")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "1234")

# Re-expose some dicts for backward compatibility if needed, but they're empty
RTSP_CAMERAS = {"Default": RTSP_URL}
ENABLED_CAMERAS = {"Default": True}
SPEAKER_DEVICE_IDS = {}
EXTERNAL_API_URLS = {}
