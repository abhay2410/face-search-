# Face Search & Recognition System: Full System Documentation

## 1. Introduction
This system is a high-performance, standalone face recognition solution designed for real-time monitoring and historical data ingestion. It leverages state-of-the-art Deep Learning models (ArcFace) and high-speed vector search (FAISS) to identify individuals from camera streams or database records.

---

## 2. System Architecture

The system is composed of four primary modules:

### 2.1 Face Engine (`engine.py`)
- **Core Model**: Uses **InsightFace (ArcFace)** for biometric feature extraction.
- **Search Engine**: Uses **FAISS (Facebook AI Similarity Search)** with **HNSW (Hierarchical Navigable Small World)** indexing for millisecond-latency matching.
- **Acceleration**: Automatically detects and utilizes NVIDIA GPUs (CUDA/TensorRT) or Windows DirectML for high-speed processing.
- **Preprocessing**: Includes filters for blur detection (Laplacian variance) and minimum face size to ensure only high-quality detections are indexed.

### 2.2 Database Layer (`database.py`)
- **Engine**: MS SQL Server using `pyodbc`.
- **Connection Management**: Features thread-local persistent connections and automatic reconnection logic.
- **Caching**: Implements an in-memory TTL (Time-To-Live) cache (5-minute default) for employee lookups to minimize DB round-trips.
- **Key Tables**:
  - `employees`: Stores names, MRNs, and biometric embeddings (VARBINARY).
  - `detection_history`: Detailed logs including Base64 images and confidence scores.
  - `faiss_index`: Stores the compiled search index for persistence across system restarts.

### 2.3 Monitoring Service (`face_check.py`)
- **Streaming**: Supports high-stability RTSP ingestion with an integrated **Watchdog** that automatically restarts the stream if the connection drops.
- **ROI (Watch Zones)**: Allows defining a "Region of Interest" in the frame to focus detection on specific areas (e.g., a door or a corridor).
- **Consensus Logic**: Requires multiple consecutive matches (e.g., 3 out of 6 frames) to confirm an identity, significantly reducing false positives.
- **Cooldown**: Prevents repeated alerts for the same person within a defined window.

### 2.4 Migration & Enrollment (`migrate_base64.py`)
- **SQL Migration**: Ingests Base64 images from legacy database tables (internal or external).
- **Folder Migration**: Ingests photos from structured folders (`Photos/MRN/random.jpg`).
- **Diversity Selection**: Intelligently selects the best representative photos from a set to build a robust profile.

---

## 3. Configuration (`.env`)

The system is configured via environment variables. Key settings include:

| Variable | Description | Default |
| :--- | :--- | :--- |
| `MSSQL_SERVER` | IP address and port of the SQL Server | `192.168.0.251,1433` |
| `RTSP_URLS` | Comma-separated list of camera URLs | - |
| `FAISS_COSINE_THRESHOLD` | Minimum similarity score for a match (0.0 to 1.0) | `0.60` |
| `ROI_TOP/BOTTOM/LEFT/RIGHT` | Watch Zone boundaries (as % of frame) | `0-100` |
| `LOG_COOLDOWN` | Seconds to wait before logging the same person again | `600` |
| `DET_THRESHOLD` | Confidence threshold for face detection | `0.35` |

---

## 4. Database Schema Details

### `employees` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INT | Primary Key (FAISS internal ID) |
| `name` | NVARCHAR | Full name (Unique) |
| `employee_code` | NVARCHAR | MRN or unique identifier |
| `embedding` | VARBINARY | 512-D float32 vector (Mean) |
| `embeddings_multi` | VARBINARY | Matrix of all enrolled vectors |

### `detection_history` Table
| Column | Type | Description |
| :--- | :--- | :--- |
| `id` | INT | Primary Key |
| `name` | NVARCHAR | Name of matched person |
| `confidence` | FLOAT | Match score |
| `base64_image` | NVARCHAR(MAX) | Full frame captured at time of match |
| `detected_at` | DATETIME | Timestamp of detection |

---

## 5. Deployment & Operations

### 5.1 Starting the System
Run the `start.bat` file or use the CLI:
```powershell
python face_check.py --camera "rtsp://..."
```

### 5.2 Performing Migration
Refer to the `MIGRATION_GUIDE.md` for detailed instructions on ingesting data from SQL or Folders.

### 5.3 Maintenance
The system automatically performs an EOD (End of Day) cleanup of the `detection_history` table based on the `RETENTION_DAYS` setting to keep the database size manageable.

---

> [!IMPORTANT]
> **Safety & Privacy**: This system processes biometric data. Ensure that you are compliant with local data protection regulations (GDPR, HIPAA, etc.) regarding the storage and processing of facial images and identifiers.
