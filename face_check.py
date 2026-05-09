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
from typing import Dict, List, Optional, Tuple, Union

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
#  2. Box Tracker (IOU-based)
# ─────────────────────────────────────────────────────────────────────────────

class TrackedFace:
    def __init__(self, bbox, track_id):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.track_id = track_id
        self.age = 0
        self.hits = 0
        self.consensus = ConsensusTracker(
            threshold=config.CONSENSUS_THRESHOLD,
            window_size=config.CONSENSUS_WINDOW
        )
        self.last_emp_id = None
        self.last_confidence = 0.0

class BoxTracker:
    def __init__(self, iou_threshold=config.IOU_THRESHOLD, max_age=config.TRACKER_MAX_AGE):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: List[TrackedFace] = []
        self.next_id = 0

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0]); yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2]); yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        if interArea == 0: return 0.0
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        return interArea / float(boxAArea + boxBArea - interArea)

    def update(self, detections: List[np.ndarray]):
        # detections: list of [x1, y1, x2, y2]
        new_tracks = []
        matched_indices = set()

        for track in self.tracks:
            track.age += 1
            best_iou = -1.0
            best_idx = -1
            for i, det in enumerate(detections):
                if i in matched_indices: continue
                iou = self._iou(track.bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self.iou_threshold:
                track.bbox = detections[best_idx]
                track.age = 0
                track.hits += 1
                matched_indices.add(best_idx)
                new_tracks.append(track)
            elif track.age <= self.max_age:
                new_tracks.append(track)

        for i, det in enumerate(detections):
            if i not in matched_indices:
                new_track = TrackedFace(det, self.next_id)
                self.next_id += 1
                new_tracks.append(new_track)

        self.tracks = new_tracks
        return self.tracks

# ─────────────────────────────────────────────────────────────────────────────
#  3. Consensus Tracker
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
    
    tracker = BoxTracker()
    cooldown_map: dict[int, float] = {}
    PROCESS_EVERY = 2 # Increased frequency due to resolution optimization
    frame_idx = 0

    log.info("System Ready. Watch Zone: T:%d%% B:%d%% L:%d%% R:%d%%", 
             config.ROI_TOP, config.ROI_BOTTOM, config.ROI_LEFT, config.ROI_RIGHT)

    while not vs.stopped:
        full_frame = vs.read()
        if full_frame is None:
            await asyncio.sleep(0.01)
            continue

        frame_idx += 1
        if frame_idx % PROCESS_EVERY != 0:
            await asyncio.sleep(0)
            continue

        h, w = full_frame.shape[:2]
        rx1, ry1, rx2, ry2 = get_roi_coords(h, w)
        
        # CROP to Region of Interest
        roi_frame = full_frame[ry1:ry2, rx1:rx2]
        display = full_frame.copy() if show_window else None

        # 1. Detect faces ONLY in the ROI frame (using MONITOR resolution)
        faces = await engine.extract_faces_full(roi_frame, enrol_mode=False)

        # 2. Update Tracker
        detections = [f["bbox"] for f in faces]
        tracks = tracker.update(detections)

        # 3. Filter valid tracks and collect embeddings for batch search
        valid_tracks = []
        embeddings_to_search = []

        for track in tracks:
            # Find the face info corresponding to this track's current bbox
            face_info = next((f for f in faces if np.array_equal(f["bbox"], track.bbox)), None)

            if face_info:
                bbox = face_info["bbox"]
                # Filter small faces
                if (bbox[2]-bbox[0]) < config.FACE_MIN_SIZE: continue
                
                # Check blur
                face_crop = roi_frame[max(0, int(bbox[1])):int(bbox[3]), max(0, int(bbox[0])):int(bbox[2])]
                is_sharp, _ = engine.check_blur(face_crop)
                if not is_sharp:
                    track.consensus.add_match(None) # Record a "no-match" for blur
                    continue

                valid_tracks.append(track)
                embeddings_to_search.append(face_info["embedding"])
            else:
                # Track was not matched in this frame (age > 0)
                pass

        # 4. Batch search embeddings
        if embeddings_to_search:
            search_results = engine.search_index_multi(np.array(embeddings_to_search))

            for track, (emp_id, confidence) in zip(valid_tracks, search_results):
                matched = emp_id is not None and confidence >= threshold

                # Update track state
                track.last_emp_id = emp_id if matched else None
                track.last_confidence = confidence if matched else 0.0

                # 5. Consensus Check
                is_confirmed = track.consensus.add_match(emp_id if matched else None)

                if matched:
                    emp = await db.get_employee_by_id(emp_id)
                    name = emp["name"] if emp else f"ID:{emp_id}"
                    emp_code = emp.get("employee_code", "") if emp else ""

                    if show_window: draw_result(display, track.bbox, name, confidence, matched=True, offset_x=rx1, offset_y=ry1)

                    if is_confirmed:
                        now = time.monotonic()
                        if now - cooldown_map.get(emp_id, 0.0) >= cooldown:
                            cooldown_map[emp_id] = now
                            # Log detection with the FULL frame for context
                            asyncio.create_task(db.log_detection(emp_id, emp_code, name, round(confidence, 4), frame_to_base64(full_frame)))
                            log.info("✅ CONFIRMED %-15s Score=%.3f (Track %d)", name, confidence, track.track_id)
                else:
                    if show_window: draw_result(display, track.bbox, "Unknown", 0.0, matched=False, offset_x=rx1, offset_y=ry1)

        # Draw stale tracks (those not updated this frame)
        if show_window:
            for track in tracks:
                if track not in valid_tracks and track.age > 0:
                    name = "..."
                    if track.last_emp_id:
                        emp = await db.get_employee_by_id(track.last_emp_id)
                        name = emp["name"] if emp else f"ID:{track.last_emp_id}"
                    draw_result(display, track.bbox, name, track.last_confidence, matched=track.last_emp_id is not None, offset_x=rx1, offset_y=ry1)

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
