# 🛡️ Face Search — Standalone Camera Monitor & API Server (v3.0)

A high-performance, real-time face recognition and monitoring system. Designed to ingest CCTV camera feeds (RTSP), detect and recognize faces within a configurable **Watch Zone (ROI)**, match them against a database using FAISS (Facebook AI Similarity Search) and ArcFace embeddings, log detections directly to Microsoft SQL Server, and expose recent matches through a secured REST API.

---

## ✨ Features & Architecture

- **Watch Zone (ROI / Region of Interest)**: Configure a localized bounding region to restrict detection, saving resources and ignoring background noise.
- **Consensus Matching Filter**: Prevents false alarms by requiring a person to be detected multiple times (e.g., 3 out of 6 consecutive frames) before registering a match.
- **Automatic Camera Reconnection**: The camera loop includes watchdog monitoring to automatically reconnect if the RTSP stream drops.
- **FastAPI REST Service**: Exposes real-time system status, recent matches, patient/person counts, and endpoint-based FAISS index reloading. Protected with HTTP Basic Authentication.
- **Database Rebuilder**: Walk folder structures (e.g., `data/known person/{MRN}/`), extract high-quality face embeddings dynamically, and update the MS SQL database + FAISS index.
- **GPU Accelerated**: Designed for real-time multi-stream execution using ONNX Runtime with CUDA / TensorRT / DirectML acceleration.
- **Persistence & Sync**: Automatically synchronizes FAISS index binaries between disk and SQL server, allowing stateless server restarts.

---

## 🛠️ Project Structure

- **[face_check.py](file:///c:/Users/Abhay/Desktop/face_search/face_check.py)**: Main surveillance and detection loop. Feeds from RTSP stream, filters blur/small faces, applies consensus checks, and logs matches.
- **[api.py](file:///c:/Users/Abhay/Desktop/face_search/api.py)**: Secured FastAPI server providing REST endpoints for dashboards and integrations.
- **[rebuild_db.py](file:///c:/Users/Abhay/Desktop/face_search/rebuild_db.py)**: Utility to wipe tables/indexes, process image files in `data/known person/{MRN}/`, extract ArcFace features, and compile a new FAISS HNSW index.
- **[engine.py](file:///c:/Users/Abhay/Desktop/face_search/engine.py)**: AI engine wrapping InsightFace (ArcFace) for face analysis and FAISS HNSW for nearest-neighbor vector search.
- **[database.py](file:///c:/Users/Abhay/Desktop/face_search/database.py)**: Self-contained database access layer using thread-safe `pyodbc` connections.
- **[config.py](file:///c:/Users/Abhay/Desktop/face_search/config.py)**: Environment configuration loader.
- **[start.bat](file:///c:/Users/Abhay/Desktop/face_search/start.bat)**: Simple helper batch file to run the camera monitor from the local virtual environment.

---

## 🚀 Quick Start

### 1. Prerequisites
Ensure you have the following installed on your machine:
- **Python 3.10+** (64-bit version recommended)
- **NVIDIA GPU Drivers & CUDA Toolkit** (for GPU acceleration)
- **Microsoft ODBC Driver for SQL Server** (Driver 18 or 17)

### 2. Installation
Clone the repository and set up a local virtual environment:

```bash
# Navigate to the project directory
cd face_search

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Configuration
Copy or edit the `.env` file in the root directory to match your environment settings:

```ini
# --- Camera ---
RTSP_URL=rtsp://your_username:your_password@your_camera_ip:554/stream

# --- Database ---
MSSQL_SERVER=192.168.0.251,1433
MSSQL_USER=sa
MSSQL_PASSWORD=your_secure_password
MSSQL_DB=hospital_face
MSSQL_DRIVER=ODBC Driver 18 for SQL Server
MSSQL_TRUST_CERT=yes

# --- API Security ---
AUTH_USERNAME=admin
AUTH_PASSWORD=your_api_password
```

---

## ⚙️ Configuration Properties (.env)

The application behavior can be fully customized using environment variables:

| Setting | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| **`RTSP_URL`** | String | *Required* | Primary RTSP camera stream URL. |
| **`RTSP_URLS`** | Comma-separated Strings | *Optional* | List of multiple camera streams to fall back or run against. |
| **`MSSQL_SERVER`** | String | `192.168.0.251,1433` | MS SQL Server host and port. |
| **`MSSQL_USER`** | String | `sa` | SQL Server username. |
| **`MSSQL_PASSWORD`** | String | `sa@123` | SQL Server password. |
| **`MSSQL_DB`** | String | `hospital_face` | Destination database name. |
| **`MSSQL_DRIVER`** | String | `ODBC Driver 18 for SQL Server` | ODBC Driver name installed on the system. |
| **`MSSQL_TRUST_CERT`** | String | `yes` | Set to `yes` to trust SQL self-signed SSL certificates. |
| **`FAISS_COSINE_THRESHOLD`**| Float | `0.50` | Cosine similarity threshold (higher = stricter match). |
| **`FACE_MIN_SIZE`** | Integer | `30` | Minimum bounding box size in pixels to process a face. |
| **`BLUR_THRESHOLD`** | Float | `40.0` | Laplacian variance threshold (higher = sharper required). |
| **`CONSENSUS_WINDOW`** | Integer | `6` | Frame history size to verify a match consensus. |
| **`CONSENSUS_THRESHOLD`** | Integer | `3` | Required matches within the history window to log the match. |
| **`LOG_COOLDOWN`** | Integer | `600` | Seconds to wait before logging the same person again (default: 10m). |
| **`RETENTION_DAYS`** | Integer | `1` | Database cleanup period in days (if cleanup loop is enabled). |
| **`API_HOST`** | String | `0.0.0.0` | Bind IP for the FastAPI application. |
| **`API_PORT`** | Integer | `8001` | Server port for the FastAPI application. |
| **`ROI_TOP` / `ROI_BOTTOM`** | Integer (0-100) | `0` / `100` | Top and Bottom percentage boundary for the Watch Zone. |
| **`ROI_LEFT` / `ROI_RIGHT`** | Integer (0-100) | `0` / `100` | Left and Right percentage boundary for the Watch Zone. |

---

## 🏃 Running the Application

### 1. Camera Monitoring Stream
To start the live video feed monitor:
- **GUI Window Mode**:
  ```bash
  python face_check.py
  ```
  *(Press **'Q'** or **'ESC'** while focusing on the video window to hide the GUI and continue running in the background).*
- **Headless / Background Mode**:
  ```bash
  python face_check.py --no-window
  ```
- **Windows Launcher**:
  Double-click `start.bat` to launch the camera feed from the local virtual environment.

### 2. REST API Server
To launch the backend API server:
```bash
python api.py
```
By default, the server binds to `http://localhost:8001`. You can view the automated OpenAPI documentation at `http://localhost:8001/docs`.

**Key Endpoints:**
- `GET /` — Root welcome status check.
- `GET /status` — Returns GPU/VRAM statistics and FAISS index size. (Requires Authentication)
- `GET /matches/recent?limit=20` — Fetches list of recent matched detections. (Requires Authentication)
- `POST /index/reload` — Commands the engine to reload the FAISS index from the database. (Requires Authentication)
- `GET /patients` — Returns total number of registered individuals in the index. (Requires Authentication)

### 3. Database Rebuilding & Importing
If you are initializing the system or have imported new photographs under `data/known person/{MRN}/`:
```bash
python rebuild_db.py
```
This utility:
1. Wipes all rows in `patients` and `faiss_index` SQL tables.
2. Deletes local `.index` files to prevent cache conflicts.
3. Walks each subdirectory inside `data/known person/`. The subdirectory name is registered as the patient's **MRN**.
4. Extracts face embeddings from all supported image formats (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`).
5. Computes a central representative embedding and inserts it into SQL.
6. Rebuilds and saves the FAISS HNSW search index, syncing it back to SQL.

---

## 📦 Compilation & Packaging

To compile the application into a single standalone Windows executable (`.exe`), use PyInstaller with the provided spec file:

```bash
# Install PyInstaller inside the virtual environment
pip install pyinstaller

# Build the executable
pyinstaller FaceSearch.spec
```

The compiled binary will be generated under the `dist/` folder. It packages dependencies and the `.env` settings structure for quick distribution on production servers.

---
*Face Search Surveillance Core — Version 3.0*
