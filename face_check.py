"""
face_check.py — Standalone Camera Face Checker
===============================================
Opens the local webcam or RTSP stream, detects faces, matches against DB,
and saves matched snapshots. Optimized for RTSP stability.
"""

import argparse
import asyncio
import base64
import logging
import os
import sys
import threading
import time
from pathlib import Path

# Force TCP for RTSP to prevent H264 artifacts and decode errors
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_check")

# ── Add project root to path ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
import database as db
import engine


# ─────────────────────────────────────────────────────────────────────────────
#  RTSP Buffer Manager (Keeps video real-time)
# ─────────────────────────────────────────────────────────────────────────────

class VideoStream:
    """Helper to keep reading frames in a thread so the buffer never lags."""
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame = None
        self.stopped = False
        self.lock = threading.Lock()

    def start(self):
        t = threading.Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            if not ret:
                self.stopped = True
                continue
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def release(self):
        self.stopped = True
        self.cap.release()

# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buf.tobytes()).decode("utf-8") if ok else ""


def draw_result(frame: np.ndarray, bbox, name: str, confidence: float, matched: bool):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 220, 0) if matched else (0, 80, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    label = f"{name} {confidence * 100:.1f}%" if matched else f"Unknown {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Main async loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(camera_source, threshold: float, cooldown: int, show_window: bool):
    await db.init_db()
    
    loaded = await engine.load_index_from_disk()
    if not loaded:
        await engine.load_index()

    log.info("Opening camera: %s", camera_source)
    
    # Use Threaded VideoStream for RTSP stability and zero lag
    vs = VideoStream(camera_source).start()
    time.sleep(2.0) # wait for first frame
    
    if vs.stopped:
        log.error("Cannot open camera: %s", camera_source)
        return

    log.info("Camera opened. Preview: %s", show_window)
    cooldown_map: dict[int, float] = {}
    frame_idx = 0
    PROCESS_EVERY = 3 

    while not vs.stopped:
        frame = vs.read()
        if frame is None:
            await asyncio.sleep(0.01)
            continue

        frame_idx += 1
        display = frame.copy()

        if frame_idx % PROCESS_EVERY == 0:
            faces = await engine.extract_faces_full(frame)

            for face_info in faces:
                bbox, embedding, det_score = face_info["bbox"], face_info["embedding"], face_info["score"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                
                if (x2-x1) < config.FACE_MIN_SIZE or (y2-y1) < config.FACE_MIN_SIZE:
                    continue

                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                is_sharp, _ = engine.check_blur(face_crop)
                if not is_sharp:
                    cv2.putText(display, "Blurry", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    continue

                emp_id, confidence = engine.search_index(embedding)
                matched = emp_id is not None and confidence >= threshold

                if matched:
                    emp = await db.get_employee_by_id(emp_id)
                    name = emp["name"] if emp else f"ID:{emp_id}"
                    emp_code = emp.get("employee_code", "") if emp else ""
                    draw_result(display, bbox, name, confidence, matched=True)

                    now = time.monotonic()
                    if now - cooldown_map.get(emp_id, 0.0) >= cooldown:
                        cooldown_map[emp_id] = now
                        asyncio.create_task(db.log_detection(emp_id, emp_code, name, round(confidence, 4), frame_to_base64(frame)))
                        log.info("✅ MATCH %-20s Code=%-10s Score=%.3f", name, emp_code, confidence)
                else:
                    draw_result(display, bbox, "Unknown", confidence if confidence > 0 else det_score, matched=False)

        if show_window:
            cv2.imshow("Face Check", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                log.info("Window closed. Switching to BACKGROUND MODE. (Press Ctrl+C in terminal to stop entirely)")
                cv2.destroyAllWindows()
                show_window = False

        await asyncio.sleep(0)

    vs.release()
    if show_window:
        cv2.destroyAllWindows()
    log.info("Done.")


def main():
    parser = argparse.ArgumentParser(description="Standalone camera face checker")
    parser.add_argument("--camera", default=config.RTSP_URL)
    parser.add_argument("--threshold", type=float, default=config.FAISS_COSINE_THRESHOLD)
    parser.add_argument("--cooldown", type=int, default=config.LOG_COOLDOWN)
    parser.add_argument("--no-window", action="store_true")
    args = parser.parse_args()

    camera_source = int(args.camera) if args.camera.isdigit() else args.camera
    asyncio.run(run(camera_source, args.threshold, args.cooldown, not args.no_window))

if __name__ == "__main__":
    main()
