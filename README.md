# 🔍 Face Search Standalone v2.0

A high-performance, fully standalone face recognition service powered by **ArcFace (InsightFace)** and **FAISS**.

This service allows enrollment and identification of employees independently of the main RTSP recognition system, while sharing the same MS SQL database and embedding tables.

---

## ⚡ Main Features

- **Standalone Live Camera**: Built-in RTSP monitoring thread with MJPEG streaming.
- **Dynamic Enrollment**: Enroll employees via image upload or direct capture from the RTSP stream.
- **Live Search**: "Grab & Search" functionality to identify who is currently in front of the camera.
- **Dual Interface**:
    - **Interactive Dashboard**: Modern web interface at `http://localhost:8001/`
    - **REST API**: Full OpenAPI documentation at `http://localhost:8001/docs`
- **FAISS Persistence**: Automatic sync of the vector index to MS SQL for disaster recovery.
- **Zero Configuration Conflict**: Runs on port **8001**, allowing it to co-exist with the main system (port 8000).

---

## 📂 Directory Structure

```text
face_search/
├── app.py            ← FastAPI Standalone Service (v2.0)
├── database.py       ← MS SQL CRUD + FAISS SQL Blob Sync
├── engine.py         ← ArcFace + FAISS Core Logic
├── config.py         ← Centralized Settings (Shared .env)
├── silencer.py       ← Suppress ORT C++ Console Logs
├── .env              ← Environment Variables
├── start.bat         ← One-click Launcher
└── data/
    └── faiss_hnsw.index   ← Local FAISS Cache
```

---

## 🚀 Getting Started

1. **Configure Camera**: Open `.env` and set your `RTSP_URL`.
2. **Start Service**:
   ```bat
   cd face_search
   start.bat
   ```
3. **Access Dashboard**: Open `http://localhost:8001/` in your browser.

---

## 🛠 API Endpoints

### `POST /enroll`
Enroll a new employee. If `file` is omitted, the service grabs a frame from the live RTSP stream.

### `POST /search`
Upload an image to identify an employee.

### `POST /search/live`
Instantly grab the current frame from the camera and identify the person.

### `GET /stream`
RTSP camera stream with real-time face bounding boxes and names.

---

## ⚙️ How it Works

1. **Thread-Safe Camera**: A background thread continuously reads from the RTSP stream to ensure zero lag.
2. **ArcFace Pipeline**: Faces are detected using SCRFD and aligned to 112x112 landmarks for maximum ArcFace accuracy.
3. **HNSW Indexed Search**: Uses FAISS HNSW (Hierarchical Navigable Small World) for sub-millisecond similarity search even as the database grows.
4. **SQL Sync**: When an employee is enrolled, their embeddings are saved to the `employees` table, and the updated FAISS index is backed up to the `faiss_index` table.
