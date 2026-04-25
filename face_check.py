"""
face_check.py — Standalone Camera Face Checker
===============================================
Opens the local webcam, detects faces, matches against the DB,
and saves matched snapshots into the detection_history table.

Usage:
    python face_check.py
    python face_check.py --camera 0          # webcam index (default 0)
    python face_check.py --camera rtsp://... # RTSP stream
    python face_check.py --threshold 0.60    # override cosine threshold
    python face_check.py --cooldown 30       # seconds before re-logging same person

No API server required. Press 'Q' to quit the preview window.
"""

import argparse
import asyncio
import base64
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_check")

# ── Add project root to path (so config / engine / database import cleanly) ──
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import config
import database as db
import engine


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────

def frame_to_base64(frame: np.ndarray, quality: int = 80) -> str:
    """Encode a BGR OpenCV frame to a JPEG base64 string."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    ok, buf = cv2.imencode(".jpg", frame, encode_params)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def draw_result(frame: np.ndarray, bbox, name: str, confidence: float, matched: bool):
    """Draw bounding box + label on frame (in-place)."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    color = (0, 220, 0) if matched else (0, 80, 255)

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    label = f"{name}  {confidence * 100:.1f}%" if matched else f"Unknown  {confidence * 100:.1f}%"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
#  Main async loop
# ─────────────────────────────────────────────────────────────────────────────

async def run(camera_source, threshold: float, cooldown: int, show_window: bool):
    # 1. Init DB schema
    log.info("Connecting to database...")
    await db.init_db()

    # 2. Load FAISS index
    log.info("Loading face index...")
    loaded = await engine.load_index_from_disk()
    if not loaded:
        log.info("Disk index missing or stale — rebuilding from DB...")
        await engine.load_index()

    total = engine._index.ntotal if engine._index else 0
    log.info("Index ready: %d face vectors enrolled.", total)
    if total == 0:
        log.warning("No enrolled faces found. Detections will always be Unknown.")

    # 3. Open camera
    log.info("Opening camera: %s", camera_source)
    cap = cv2.VideoCapture(camera_source)
    if not cap.isOpened():
        log.error("Cannot open camera: %s", camera_source)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # low latency

    log.info("Camera opened. Press Q in the preview window to quit.")

    # Per-person cooldown tracker  { employee_id: last_logged_timestamp }
    cooldown_map: dict[int, float] = {}

    frame_idx = 0
    PROCESS_EVERY = 3   # run detection every N frames to save CPU/GPU

    while True:
        ret, frame = cap.read()
        if not ret:
            log.warning("Frame read failed — retrying...")
            await asyncio.sleep(0.1)
            continue

        frame_idx += 1
        display = frame.copy()

        if frame_idx % PROCESS_EVERY == 0:
            # ── Face detection + embedding extraction ──────────────────────
            faces = await engine.extract_faces_full(frame)

            for face_info in faces:
                bbox       = face_info["bbox"]       # [x1,y1,x2,y2]
                embedding  = face_info["embedding"]  # np.ndarray (512,)
                det_score  = face_info["score"]

                # Skip tiny / blurry faces
                x1, y1, x2, y2 = [int(v) for v in bbox]
                face_w = x2 - x1
                face_h = y2 - y1
                if face_w < config.FACE_MIN_SIZE or face_h < config.FACE_MIN_SIZE:
                    continue

                face_crop = frame[max(0, y1):y2, max(0, x1):x2]
                is_sharp, blur_score = engine.check_blur(face_crop)
                if not is_sharp:
                    cv2.putText(display, "Blurry", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1)
                    continue

                # ── FAISS search ───────────────────────────────────────────
                emp_id, confidence = engine.search_index(embedding)
                matched = emp_id is not None and confidence >= threshold

                if matched:
                    # Fetch employee details
                    emp = await db.get_employee_by_id(emp_id)
                    name = emp["name"] if emp else f"ID:{emp_id}"
                    emp_code = emp.get("employee_code", "") if emp else ""

                    draw_result(display, bbox, name, confidence, matched=True)

                    # ── Cooldown check before logging ──────────────────────
                    now = time.monotonic()
                    last = cooldown_map.get(emp_id, 0.0)
                    if now - last >= cooldown:
                        cooldown_map[emp_id] = now

                        # Encode snapshot as base64
                        b64 = frame_to_base64(frame)

                        # Save to detection_history (fire-and-forget)
                        asyncio.create_task(
                            db.log_detection(
                                employee_id  = emp_id,
                                employee_code= emp_code,
                                name         = name,
                                confidence   = round(confidence, 4),
                                base64_image = b64,
                            )
                        )

                        log.info(
                            "✅ MATCH  %-20s  EmpCode=%-10s  Score=%.3f",
                            name, emp_code, confidence,
                        )
                    else:
                        remaining = int(cooldown - (now - last))
                        log.debug("Cooldown active for %s — %ds remaining.", name, remaining)

                else:
                    conf_display = confidence if confidence > 0 else det_score
                    draw_result(display, bbox, "Unknown", conf_display, matched=False)
                    log.debug("No match (best score=%.3f)", confidence)

        # ── Show preview window ────────────────────────────────────────────
        if show_window:
            cv2.putText(display, "Face Check  [Q=Quit]", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Face Check", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                log.info("Quit requested.")
                break

        await asyncio.sleep(0)   # yield control so DB tasks can run

    cap.release()
    if show_window:
        cv2.destroyAllWindows()
    log.info("Camera released. Done.")


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Standalone camera face checker")
    parser.add_argument(
        "--camera", default="0",
        help="Camera index (0, 1, …) or RTSP URL. Default: 0",
    )
    parser.add_argument(
        "--threshold", type=float, default=config.FAISS_COSINE_THRESHOLD,
        help=f"Cosine similarity threshold for a match. Default: {config.FAISS_COSINE_THRESHOLD}",
    )
    parser.add_argument(
        "--cooldown", type=int, default=30,
        help="Seconds before the same person is logged again. Default: 30",
    )
    parser.add_argument(
        "--no-window", action="store_true",
        help="Run headless (no preview window) — useful for servers.",
    )
    args = parser.parse_args()

    # Convert camera arg: if it's a digit string, use int (webcam index)
    camera_source = int(args.camera) if args.camera.isdigit() else args.camera

    asyncio.run(
        run(
            camera_source = camera_source,
            threshold     = args.threshold,
            cooldown      = args.cooldown,
            show_window   = not args.no_window,
        )
    )


if __name__ == "__main__":
    main()
