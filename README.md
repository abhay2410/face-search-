# 🛡️ Face Search — Standalone Camera Monitor (v3.0)

A high-performance, standalone face recognition system designed to monitor a CCTV feed, detect faces in a specific "Watch Zone," and log matches directly to an MS SQL database.

## ✨ Key Features
- **Watch Zone (ROI)**: Define a specific area of the screen to monitor, ignoring background movement.
- **Consensus Matching**: Requires a person to be seen multiple times (3/6 frames) before logging, ensuring near 100% accuracy.
- **Auto-Reconnect**: Watchdog logic automatically restarts the camera stream if it drops.
- **Automatic Cleanup**: Keeps the database light by automatically deleting logs older than 1 day.
- **GPU Accelerated**: Uses NVIDIA CUDA for real-time inference.

## 🚀 Quick Start
1.  **Configure**: Open `.env` and set your `RTSP_URL` and `MSSQL` credentials.
2.  **Launch**: Double-click `start.bat`.
3.  **Monitor**: The system will open a window showing the detection. Press **'Q'** to hide the window and continue running in the background.

## ⚙️ Configuration (.env)
| Setting | Description |
| :--- | :--- |
| `RTSP_URL` | Your camera stream URL. |
| `LOG_COOLDOWN` | Seconds to wait before logging the same person again (default: 600s/10m). |
| `RETENTION_DAYS` | Automatically delete logs older than this many days (default: 1). |
| `ROI_TOP/LEFT/etc` | Define the percentage (0-100) of the screen to monitor. |

## 🛠️ Architecture
- **`face_check.py`**: Main entry point and camera loop.
- **`engine.py`**: ArcFace / InsightFace AI logic.
- **`database.py`**: MS SQL connection and logging.
- **`config.py`**: Environment variable loader.

---
*Standalone version — No API or Frontend required.*
