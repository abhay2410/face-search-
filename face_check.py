"""
face_check.py — Standalone Camera Face Checker (Pro Version + ROI)
==================================================================
Updates:
- Region of Interest (ROI): Define a specific "Watch Zone" in .env
- Watchdog & Consensus logic included
"""

import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import time
from collections import deque
from pathlib import Path
from datetime import datetime

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8" # Quiet
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

import cv2
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_check")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
import database as db
import engine

# ─────────────────────────────────────────────────────────────────────────────
#  1. RTSP Watchdog Stream
# ─────────────────────────────────────────────────────────────────────────────

class VideoStream:
    def __init__(self, source, target_fps=10):
        self.source = source
        self.cap = None
        self.stopped = False
        self.last_frame_time = time.time()
        self.lock = threading.Lock()
        
        # Pre-allocated buffer to prevent memory fragmentation
        self._buf = np.zeros((1080, 1920, 3), dtype=np.uint8)
        self._h, self._w = 0, 0
        self._target_interval = 1.0 / target_fps
        self._connect()

    def _connect(self):
        if self.cap: self.cap.release()
        log.info("Connecting to camera: %s", self.source)
        # Optimized FFmpeg options for low latency
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|fflags;nobuffer|buffer_size;1024000"
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        last_retrieval = 0
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                now = time.time()
                # Ingest at target FPS only
                if now - last_retrieval >= self._target_interval:
                    with self.lock:
                        h, w = frame.shape[:2]
                        if h > 1080 or w > 1920:
                            # Safety scale if camera is 4K+
                            scale = min(1080/h, 1920/w)
                            frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
                            h, w = frame.shape[:2]
                        
                        self._h, self._w = h, w
                        np.copyto(self._buf[:h, :w], frame)
                        self.last_frame_time = now
                    last_retrieval = now
            else:
                if time.time() - self.last_frame_time > 5.0:
                    log.warning("Camera timeout — Reconnecting...")
                    self._connect()
                    time.sleep(1.0)
                time.sleep(0.01)

    def read(self):
        with self.lock:
            if self._h == 0: return None
            return self._buf[:self._h, :self._w].copy()

    def release(self):
        self.stopped = True
        if self.cap: self.cap.release()

# ─────────────────────────────────────────────────────────────────────────────
#  2. Consensus Tracker
# ─────────────────────────────────────────────────────────────────────────────

class ConsensusTracker:
    def __init__(self, threshold=3, window_size=6):
        self.history = deque(maxlen=window_size)
        self.threshold = threshold

    def add_match(self, emp_id):
        self.history.append(emp_id)
        if emp_id is not None:
            count = sum(1 for x in self.history if x == emp_id)
            return count >= self.threshold
        return False

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers & ROI calculation
# ─────────────────────────────────────────────────────────────────────────────

def get_roi_coords(h, w):
    """Convert ROI percentages to pixel coordinates."""
    y1 = int(h * config.ROI_TOP / 100)
    y2 = int(h * config.ROI_BOTTOM / 100)
    x1 = int(w * config.ROI_LEFT / 100)
    x2 = int(w * config.ROI_RIGHT / 100)
    return x1, y1, x2, y2

def frame_to_base64(frame: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""

def draw_result(frame: np.ndarray, bbox, name: str, confidence: float, matched: bool, offset_x=0, offset_y=0):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    # Add ROI offsets to draw on the original full frame
    x1 += offset_x; x2 += offset_x
    y1 += offset_y; y2 += offset_y
    
    color = (0, 220, 0) if matched else (0, 80, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name} {confidence * 100:.1f}%" if matched else "Unknown"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

class TokenManager:
    """Manages queue numbers in memory to avoid DB round-trips."""
    def __init__(self):
        self._current_token = 0
        self._last_date = None

    async def get_next(self):
        now_date = datetime.now().date()
        # Reset if day changed or first run
        if self._last_date != now_date:
            log.info("[Queue] New day detected. Fetching starting token from DB...")
            self._current_token = await db.get_next_queue_no()
            self._last_date = now_date
        else:
            self._current_token += 1
        return self._current_token

token_manager = TokenManager()

# ─────────────────────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(camera_source, threshold: float, cooldown: int, show_window: bool):
    await db.init_db()
    # await db.clear_old_detections() # Disabled as requested (no data deletion)
    
    loaded = await engine.load_index_from_disk()
    if not loaded: await engine.load_index()

    vs = VideoStream(camera_source, target_fps=10).start()
    time.sleep(2.0)
    
    trackers: dict[int, ConsensusTracker] = {}
    cooldown_map: dict[int, float] = {}
    
    # Precise Processing Governor
    PROCESSING_FPS = 5 
    INTERVAL = 1.0 / PROCESSING_FPS

    log.info("System Ready. Target Processing: %d FPS", PROCESSING_FPS)

    while not vs.stopped:
        t_start = time.monotonic()
        full_frame = vs.read()
        if full_frame is None:
            await asyncio.sleep(0.01)
            continue

        h, w = full_frame.shape[:2]
        rx1, ry1, rx2, ry2 = get_roi_coords(h, w)
        roi_frame = full_frame[ry1:ry2, rx1:rx2]

        display = None
        if show_window:
            # Only copy if we are actually showing the window
            display = full_frame.copy()

        # Detect faces ONLY in the ROI frame
        faces = await engine.extract_faces_full(roi_frame)

        # ── Filter pass: size + blur ────────────────────────────────────────
        valid_faces = []
        for face_info in faces:
            bbox = face_info["bbox"]
            if (bbox[2] - bbox[0]) < config.FACE_MIN_SIZE:
                continue
            
            # Fast crop
            x1, y1, x2, y2 = [int(v) for v in bbox]
            face_crop = roi_frame[max(0, y1):int(y2), max(0, x1):int(x2)]
            
            is_sharp, _ = engine.check_blur(face_crop)
            if is_sharp:
                valid_faces.append(face_info)

        if valid_faces:
            # ── Async Batch FAISS search (Parallelized) ─────────────────────
            embeddings_batch = np.array([f["embedding"] for f in valid_faces], dtype=np.float32)
            search_results = await asyncio.to_thread(engine.search_index_multi, embeddings_batch)

            for face_info, (emp_id, confidence) in zip(valid_faces, search_results):
                bbox = face_info["bbox"]
                matched = emp_id is not None and confidence >= threshold

                if matched:
                    if emp_id not in trackers:
                        trackers[emp_id] = ConsensusTracker(threshold=3, window_size=6)

                    is_confirmed = trackers[emp_id].add_match(emp_id)

                    # ── Fast cache-first lookup (no async overhead on hit) ──
                    emp = db._cache_get(emp_id)
                    if emp is None:
                        emp = await db.get_patient_by_id(emp_id)  # DB hit only on cold miss
                    
                    name = emp["name"]      if emp else f"ID:{emp_id}"
                    mrn  = emp.get("mrn", "") if emp else ""

                    if show_window and display is not None:
                        draw_result(display, bbox, name, confidence, matched=True, offset_x=rx1, offset_y=ry1)

                    if is_confirmed:
                        now = time.monotonic()
                        if now - cooldown_map.get(emp_id, 0.0) >= cooldown:
                            cooldown_map[emp_id] = now
                            
                            # ── In-Memory Token Increment (Instant) ─────────
                            q_no = await token_manager.get_next()
                            
                            asyncio.create_task(db.log_detection(mrn, name, round(confidence, 4), "", queue_no=q_no))
                            log.info("✅ TOKEN #%-3d | MRN:%-12s | %-15s Score=%.3f", q_no, mrn, name, confidence)
                            
                            if show_window and display is not None:
                                # Show token on screen briefly
                                cv2.putText(display, f"TOKEN: {q_no}", (rx1 + 10, ry2 - 20), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                else:
                    if show_window and display is not None:
                        draw_result(display, bbox, "Unknown", 0.0, matched=False, offset_x=rx1, offset_y=ry1)

        if show_window and display is not None:
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(display, "WATCH ZONE", (rx1 + 5, ry1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.imshow("Face Check", display)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                log.info("Switching to BACKGROUND MODE...")
                cv2.destroyAllWindows()
                show_window = False

        # FPS Governor: Sleep to maintain PROCESSING_FPS
        elapsed = time.monotonic() - t_start
        await asyncio.sleep(max(0, INTERVAL - elapsed))

    vs.release()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", default=config.RTSP_URL)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    asyncio.run(run(camera_source, config.FAISS_COSINE_THRESHOLD, config.LOG_COOLDOWN, not args.no_window))

if __name__ == "__main__":
    main()
