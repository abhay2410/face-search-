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
    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.stopped = False
        self.last_frame_time = time.time()
        self.lock = threading.Lock()
        self._connect()

    def _connect(self):
        if self.cap: self.cap.release()
        log.info("Connecting to camera...")
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.last_frame_time = time.time()

    def start(self):
        t = threading.Thread(target=self.update, daemon=True)
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame
                    self.last_frame_time = time.time()
            else:
                if time.time() - self.last_frame_time > 5.0:
                    log.warning("Camera timeout — Reconnecting...")
                    self._connect()
                time.sleep(0.1)

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

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

# ─────────────────────────────────────────────────────────────────────────────
#  Main Loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(camera_source, threshold: float, cooldown: int, show_window: bool):
    await db.init_db()
    await db.clear_old_detections()
    
    loaded = await engine.load_index_from_disk()
    if not loaded: await engine.load_index()

    vs = VideoStream(camera_source).start()
    time.sleep(2.0)
    
    trackers: dict[int, ConsensusTracker] = {}
    cooldown_map: dict[int, float] = {}
    PROCESS_EVERY = 3
    frame_idx = 0

    log.info("System Ready. Watch Zone: T:%d%% B:%d%% L:%d%% R:%d%%", 
             config.ROI_TOP, config.ROI_BOTTOM, config.ROI_LEFT, config.ROI_RIGHT)

    while not vs.stopped:
        full_frame = vs.read()
        if full_frame is None:
            await asyncio.sleep(0.01)
            continue

        h, w = full_frame.shape[:2]
        rx1, ry1, rx2, ry2 = get_roi_coords(h, w)
        
        # CROP to Region of Interest
        roi_frame = full_frame[ry1:ry2, rx1:rx2]

        frame_idx += 1
        display = full_frame.copy() if show_window else None

        if frame_idx % PROCESS_EVERY == 0:
            # Detect faces ONLY in the ROI frame
            faces = await engine.extract_faces_full(roi_frame)

            for face_info in faces:
                bbox, embedding = face_info["bbox"], face_info["embedding"]
                
                # Filter small faces
                if (bbox[2]-bbox[0]) < config.FACE_MIN_SIZE: continue
                
                # Check blur
                face_crop = roi_frame[max(0, int(bbox[1])):int(bbox[3]), max(0, int(bbox[0])):int(bbox[2])]
                is_sharp, _ = engine.check_blur(face_crop)
                if not is_sharp: continue

                emp_id, confidence = engine.search_index(embedding)
                matched = emp_id is not None and confidence >= threshold

                if matched:
                    if emp_id not in trackers:
                        trackers[emp_id] = ConsensusTracker(threshold=3, window_size=6)
                    
                    is_confirmed = trackers[emp_id].add_match(emp_id)
                    
                    emp = await db.get_employee_by_id(emp_id)
                    name = emp["name"] if emp else f"ID:{emp_id}"
                    emp_code = emp.get("employee_code", "") if emp else ""

                    if show_window: draw_result(display, bbox, name, confidence, matched=True, offset_x=rx1, offset_y=ry1)

                    if is_confirmed:
                        now = time.monotonic()
                        if now - cooldown_map.get(emp_id, 0.0) >= cooldown:
                            cooldown_map[emp_id] = now
                            # Log detection with the FULL frame for context
                            asyncio.create_task(db.log_detection(emp_id, emp_code, name, round(confidence, 4), frame_to_base64(full_frame)))
                            log.info("✅ CONFIRMED %-15s Score=%.3f", name, confidence)
                else:
                    if show_window: draw_result(display, bbox, "Unknown", 0.0, matched=False, offset_x=rx1, offset_y=ry1)

        if show_window:
            # Draw the ROI boundary for visual reference
            cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)
            cv2.putText(display, "WATCH ZONE", (rx1 + 5, ry1 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Face Check", display)
            if (cv2.waitKey(1) & 0xFF) in (ord("q"), ord("Q"), 27):
                log.info("Switching to BACKGROUND MODE...")
                cv2.destroyAllWindows()
                show_window = False

        await asyncio.sleep(0)

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
