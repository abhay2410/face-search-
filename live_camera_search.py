"""
Live Camera Face Search
=======================
Continuously monitors an RTSP stream (192.168.1.216) and performs FAISS face search.
Returns the Employee ID if matched. All door, status, and RF logic has been removed.
"""

import asyncio
import cv2
import time
import threading
import numpy as np

# Silence C++ logs before importing ML modules
try:
    import silencer
    silencer.silence_cpp_logs()
except ImportError:
    pass

import config
import database
import engine

# Use RTSP url from unified config
CAMERA_URL = config.RTSP_URL

class RTSPStream:
    """Background thread to read RTSP frames so we always get the newest frame and avoid lag."""
    def __init__(self, src):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return self
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            # If the stream fails, try to reconnect
            if not grabbed:
                time.sleep(1)
                self.cap = cv2.VideoCapture(self.src)

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        if hasattr(self, 'thread'):
            self.thread.join()
        self.cap.release()

async def process_latest_frame(frame: np.ndarray):
    """Processes a single frame for faces and finds the employee ID."""
    # 1. Blur check
    is_sharp, blur_score = engine.check_blur(frame)
    if not is_sharp:
        return

    # 2. Extract faces
    faces = await engine.extract_faces_full(frame, enrol_mode=False)
    
    for f in faces:
        emb = f["embedding"]
        
        # 3. Search index
        emp_id, distance = engine.search_index(emb)
        
        # 4. Output the result if matched
        if emp_id:
            print(f"MATCH: Employee ID {emp_id} detected! (Similarity: {distance:.4f})")
            
            # Note: We just output the ID. 
            # If you want to integrate this with another software, you can write the `emp_id` to a file, database, or API here.

async def live_search():
    # 1. Initialize Database
    try:
        await database.init_db()
    except Exception as e:
        print(f"ERROR: Database initialisation failed: {e}")
        return

    # 2. Load FAISS Index
    loaded = await engine.load_index_from_disk()
    if not loaded:
        print("[INFO] Rebuilding FAISS index from SQL database...")
        await engine.load_index()

    print(f"\n[INFO] Connecting to camera feed: {CAMERA_URL}")
    stream = RTSPStream(CAMERA_URL).start()
    
    # Wait for the stream to warm up
    time.sleep(2)
            
    if not stream.grabbed:
        print(f"[ERROR] Could not connect to camera stream at {CAMERA_URL}")
        stream.stop()
        return

    print("[SUCCESS] Stream connected. Starting real-time face search...")
    print("Press Ctrl+C to stop.\n")

    try:
        while True:
            grabbed, frame = stream.read()
            if grabbed and frame is not None:
                # Process the frame
                await process_latest_frame(frame)
            
            # Short sleep to pace the processing and free the event loop
            await asyncio.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n[INFO] Search utility stopped by user.")
    finally:
        stream.stop()
        await engine.close_engine()

def main():
    asyncio.run(live_search())

if __name__ == "__main__":
    main()
