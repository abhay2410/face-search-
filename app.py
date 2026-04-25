"""
face_search/app.py  –  Standalone Face Search & Enrollment Service  [v3.0 — Multi-Cam Optimizer]
=====================================================================================
Fully self-contained FastAPI service with a 4-camera grid dashboard.
Optimized for 2GB VRAM via serialized GPU access and unified models.
"""

import asyncio
import base64
import datetime
import logging
import os
import threading
import time
from collections import deque
from contextlib import asynccontextmanager
from typing import Optional, List, Dict

# Fix UDP packet loss smearing by forcing TCP for FFmpeg *before* cv2 is imported
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

try:
    import silencer
    silencer.silence_cpp_logs()
except ImportError:
    pass

import config
import database as db
import engine

# ──────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("face_search")

# ──────────────────────────────────────────────────────────────────────────────
#  1. Multi-Camera State management
# ──────────────────────────────────────────────────────────────────────────────

class _CameraState:
    def __init__(self, name: str):
        self.name = name
        self.frame: Optional[np.ndarray] = None
        self.raw_frame: Optional[np.ndarray] = None
        self.last_match: dict = {}
        self.lock = threading.Lock()
        self.running = False
        self.is_searching = True
        self.thread: Optional[threading.Thread] = None
        self.frame_count: int = 0   # monotonic counter for change detection

    def update(self, annotated: np.ndarray, raw: np.ndarray, match: dict):
        with self.lock:
            self.frame = annotated.copy()
            self.raw_frame = raw.copy()
            self.frame_count += 1
            if match:
                self.last_match = match

    def get_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.frame.copy() if self.frame is not None else None

    def get_raw_frame(self) -> Optional[np.ndarray]:
        with self.lock:
            return self.raw_frame.copy() if self.raw_frame is not None else None

    def get_frame_with_count(self):
        """Returns (frame_copy, frame_count). frame_count increments on each update."""
        with self.lock:
            if self.frame is None:
                return None, -1
            return self.frame.copy(), self.frame_count

# Global Registries
_cam_registry: Dict[str, _CameraState] = {}
_consensus_registry: Dict[str, 'ConsensusManager'] = {}

# ──────────────────────────────────────────────────────────────────────────────
#  2. Consensus Logic (Temporal Voting)
# ──────────────────────────────────────────────────────────────────────────────

class ConsensusManager:
    def __init__(self, window_size: int, threshold: int):
        self.window_size = window_size
        self.threshold = threshold
        self.history: deque = deque(maxlen=window_size)  # O(1) append+pop vs list
        self.confirmed_id = None

    def add(self, emp_id: Optional[int]) -> Optional[int]:
        self.history.append(emp_id)

        counts: Dict[int, int] = {}
        for x in self.history:
            if x is not None:
                counts[x] = counts.get(x, 0) + 1

        if not counts:
            self.confirmed_id = None
            return None

        top_id, count = max(counts.items(), key=lambda item: item[1])
        if count >= self.threshold:
            self.confirmed_id = top_id
            return top_id

        self.confirmed_id = None
        return None

# ──────────────────────────────────────────────────────────────────────────────
#  3. High-Performance RTSP Threading
# ──────────────────────────────────────────────────────────────────────────────

class RTSPReader:
    def __init__(self, url: str):
        self.url = url
        os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
        self.cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Clear buffer to prevent stalling & corruption
        self.frame = None
        self.ret = False
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        # We must read as fast as possible to prevent the RTSP buffer from overflowing.
        # cap.read() will block naturally until a frame is available.
        while self.running:
            ok, frame = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = frame
                    self.ret = True
            else:
                log.warning("[RTSPReader] Lost stream: %s. Reconnecting...", self.url)
                self.cap.release()
                time.sleep(2)
                self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self):
        with self.lock:
            if self.frame is None: return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()

# ──────────────────────────────────────────────────────────────────────────────
#  4. The AI Processing Loop
# ──────────────────────────────────────────────────────────────────────────────

async def _rtsp_worker_task(cam_id: str, rtsp_url: str):
    """Async worker using run_in_executor to handle the blocking RTSP/OpenCV logic."""
    cam = _cam_registry[cam_id]
    consensus = _consensus_registry[cam_id]
    reader = RTSPReader(rtsp_url)
    
    last_processed = 0
    cooldowns = {} # emp_id -> time
    
    log.info("[Worker:%s] Started background loop.", cam_id)
    
    try:
        while cam.running:
            ok, frame = reader.read()
            if not ok or frame is None:
                await asyncio.sleep(0.1)
                continue

            now = time.time()
            # 10 FPS for single-camera — double throughput with same VRAM budget
            if now - last_processed > 0.1:
                last_processed = now
                
                try:
                    if cam.is_searching:
                        # 1. Blur Check (CPU) — only skips AI inference, NOT the display update
                        is_sharp, b_score = engine.check_blur(frame)
                        if not is_sharp:
                            if cam.frame_count % 100 == 0:
                                log.warning("[Worker:%s] Skipping AI: Frame blurry (%.1f < %.1f)", cam_id, b_score, config.BLUR_THRESHOLD)
                            # Still push raw frame to stream so dashboard stays live
                            cam.update(frame, frame, {})
                            consensus.add(None)
                            await asyncio.sleep(0.01)
                            continue

                        # 2. Allocate frame copies only on sharp frames (CPU saving preserved)
                        raw = frame.copy()
                        draw = frame.copy()
                        current_match = {}
                        top_frame_id = None
                        top_frame_score = 0
                        top_frame_face = None

                        # 3. AI Recognition (serialised via GPU_LOCK in engine.py)
                        faces = await engine.extract_faces_full(frame, enrol_mode=False)
                        
                        if faces:
                            log.info("[Worker:%s] Detected %d faces.", cam_id, len(faces))

                        for f_dict in faces:
                            f = f_dict["face"]
                            bbox = f.bbox.astype(int)
                            width, height = bbox[2]-bbox[0], bbox[3]-bbox[1]

                            if width < config.FACE_MIN_SIZE or height < config.FACE_MIN_SIZE:
                                continue

                            emp_id, score = engine.search_index(f_dict["embedding"])
                            color = (0, 220, 0) if emp_id else (0, 80, 220)
                            cv2.rectangle(draw, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

                            if emp_id and score > top_frame_score:
                                top_frame_id, top_frame_score, top_frame_face = emp_id, score, f

                        # 4. Consensus logic
                        confirmed_id = consensus.add(top_frame_id)
                        if confirmed_id:
                            emp = db._get_employee_by_id_sync(confirmed_id)
                            name = emp.get("name", "Unknown") if emp else f"ID:{confirmed_id}"
                            code = emp.get("employee_code", "") if emp else ""

                            cv2.putText(draw, f"{name}", (bbox[0], bbox[1]-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            current_match = {"name": name, "code": code,
                                             "score": top_frame_score, "id": confirmed_id}

                            # Logging with cooldown + bounded prune
                            if confirmed_id not in cooldowns or (now - cooldowns[confirmed_id] > 10):
                                cooldowns[confirmed_id] = now
                                if len(cooldowns) > 50:
                                    stale = [k for k, t in cooldowns.items() if now - t > 300]
                                    for k in stale:
                                        del cooldowns[k]
                                b64_face = ""
                                if top_frame_id == confirmed_id and top_frame_face:
                                    try:
                                        b = top_frame_face.bbox.astype(int)
                                        h_r, w_r = raw.shape[:2]
                                        p = 30
                                        crop = raw[max(0, b[1]-p):min(h_r, b[3]+p),
                                                   max(0, b[0]-p):min(w_r, b[2]+p)]
                                        if crop.size > 0:
                                            _, buf = cv2.imencode(".jpg", crop,
                                                                   [cv2.IMWRITE_JPEG_QUALITY, 70])
                                            b64_face = base64.b64encode(buf).decode("utf-8")
                                    except: pass
                                db._log_search_sync(confirmed_id, code, top_frame_score, True)
                                db._log_detection_sync(confirmed_id, code, name, top_frame_score, b64_face)

                        cam.update(draw, raw, current_match)
                    else:
                        draw = frame.copy()
                        cv2.putText(draw, "PAUSED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        cam.update(draw, frame, {})
                except Exception as e:
                    log.error("[Worker:%s] Local error: %s", cam_id, e)
            
            await asyncio.sleep(0.01)
    finally:
        reader.release()

# ──────────────────────────────────────────────────────────────────────────────
#  5. Lifespan & Service Setup
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("[Startup] Init Database...")
    await db.init_db()
    
    # Load or rebuild index
    if not await engine.load_index_from_disk():
        log.warning("[Startup] Index missing or stale. Rebuilding...")
        await engine.load_index()

    # Launch camera workers
    for idx, url in enumerate(config.RTSP_URLS):
        cam_id = f"cam_{idx}"
        cam = _CameraState(cam_id)
        cam.running = True
        _cam_registry[cam_id] = cam
        _consensus_registry[cam_id] = ConsensusManager(config.CONSENSUS_WINDOW, config.CONSENSUS_THRESHOLD)
        
        # We use create_task for the async loop
        asyncio.create_task(_rtsp_worker_task(cam_id, url))
        log.info("[Startup] Loaded %s: %s", cam_id, url.split("@")[-1])

    # Start EOD cleanup
    asyncio.create_task(db.clear_old_detections_loop())

    yield
    for cam in _cam_registry.values():
        cam.running = False
    await engine.close_engine()

app = FastAPI(title="Face Search Premium v3", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ──────────────────────────────────────────────────────────────────────────────
#  6. Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/api/matches")
async def get_matches():
    return await db.get_recent_matches(limit=15)

@app.get("/stream")
def stream(cam: str = "cam_0"):
    c = _cam_registry.get(cam)
    if not c: raise HTTPException(404, "Camera not found")
    
    def _gen():
        while True:
            f = c.get_frame()
            if f is None:
                time.sleep(0.2)
                continue
            # Quality 80 saves ~30% bandwidth with negligible visual difference
            _, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 80])
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
            time.sleep(0.067)  # ~15 FPS — smooth and clear
    return StreamingResponse(_gen(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.post("/enroll")
async def enroll(
    employee_number: str = Form(""),
    name:            str = Form(""),
    cam_id:          str = Form("cam_0"),
    file:            Optional[UploadFile] = File(None),
    image_base64:    str = Form("")
):
    img = None
    if image_base64.strip():
        raw = base64.b64decode(image_base64.split(",")[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    elif file:
        raw = await file.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    else:
        c = _cam_registry.get(cam_id)
        img = c.get_raw_frame() if c else None
    
    if img is None: raise HTTPException(400, "No image source")
    
    faces = await engine.extract_faces_full(img, enrol_mode=True)
    if not faces: raise HTTPException(422, "No face detected")
    
    best = max(faces, key=lambda f: f["score"])
    multi = engine.select_diverse_embeddings([f["embedding"] for f in faces], config.MULTI_EMB_COUNT)
    
    emp_id = await db.upsert_employee(name, employee_number, best["embedding"], multi_embeddings=multi)
    engine._add_to_index_sync(emp_id, multi)
    return {"status": "success", "id": emp_id, "name": name}

@app.post("/search")
async def search_upload(
    file: Optional[UploadFile] = File(None),
    image_base64: str = Form("")
):
    img = None
    if image_base64.strip():
        raw = base64.b64decode(image_base64.split(",")[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    elif file:
        raw = await file.read()
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        
    if img is None: raise HTTPException(400, "Invalid image")
    
    faces = await engine.extract_faces_full(img)
    if not faces: return {"match": False, "reason": "No face"}
    
    best = max(faces, key=lambda f: f["score"])
    emp_id, score = engine.search_index(best["embedding"])
    
    if emp_id:
        emp = await db.get_employee_by_id(emp_id)
        if emp:
            await db.log_search(emp_id, emp.get("employee_code"), score, True)
            return {"match": True, "name": emp.get("name"), "confidence": score}
    return {"match": False, "score": score}

@app.get("/health")
def health():
    return {
        "status": "ok", 
        "state": "online", 
        "cameras": len(_cam_registry), 
        "vectors": engine._index.ntotal if engine._index else 0
    }

# ──────────────────────────────────────────────────────────────────────────────
#  7. API v1 Compatibility (for Sidecar/Legacy)
# ──────────────────────────────────────────────────────────────────────────────

v1_router = APIRouter(prefix="/api/v1")

@v1_router.get("/health")
async def v1_health():
    return health()

@v1_router.get("/detections")
async def v1_detections(limit: int = 10):
    matches = await db.get_recent_matches(limit=limit)
    # Compatibility aliasing for sidecar UI
    for m in matches:
        m["score"] = m.get("confidence", 0)
        m["code"] = m.get("employee_number", "N/A")
        # ISO-like string for JS Date compatibility
        dt = datetime.datetime.fromtimestamp(m.get("ts", time.time()))
        m["timestamp"] = dt.strftime("%Y-%m-%d %H:%M:%S")
        m["date"] = m["timestamp"]
        m["ts_ms"] = int(m.get("ts", 0) * 1000)
    return matches

@v1_router.get("/stream/{cam_id}")
async def v1_stream(cam_id: str):
    return stream(cam_id)

@v1_router.post("/enroll")
async def v1_enroll(
    employee_number: str = Form(""),
    name:            str = Form(""),
    cam_id:          str = Form("cam_0"),
    file:            Optional[UploadFile] = File(None),
    image_base64:    str = Form("")
):
    return await enroll(employee_number, name, cam_id, file, image_base64)

@v1_router.post("/search")
async def v1_search(
    file: Optional[UploadFile] = File(None),
    image_base64: str = Form("")
):
    return await search_upload(file, image_base64)

app.include_router(v1_router)

@app.get("/", response_class=HTMLResponse)
def index():
    cam_items = ""
    cam_count = len(config.RTSP_URLS)
    
    for idx, url in enumerate(config.RTSP_URLS):
        cam_id = f"cam_{idx}"
        url_label = url.split("@")[-1]
        
        cam_items += f'''
        <div class="relative rounded-[12px] bg-[#000] overflow-hidden shadow-[rgba(0,0,0,0.22)_3px_5px_30px_0px]">
            <img src="/stream?cam={cam_id}" alt="Stream {idx}" class="w-full h-full object-cover aspect-video">
            <div class="absolute top-4 left-4 right-4 flex justify-between items-center z-10">
                <span class="bg-black/50 backdrop-blur-md px-3 py-1 rounded-full text-[12px] font-text font-semibold text-white tracking-[-0.12px] uppercase drop-shadow-md">Cam {idx}</span>
                <div class="flex items-center gap-2 bg-black/50 backdrop-blur-md px-3 py-1 rounded-full drop-shadow-md">
                    <span class="w-2 h-2 rounded-full bg-[#34c759] animate-pulse"></span>
                    <span class="text-[12px] font-text font-semibold text-white tracking-[-0.12px] uppercase">LIVE</span>
                </div>
            </div>
        </div>
        '''

    grid_class = "grid-cols-1" if cam_count == 1 else "grid-cols-1 xl:grid-cols-2"

    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8"><title>Face Search | Autonomous Recognition</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        @font-face {{
            font-family: 'SF Pro Display';
            src: local('San Francisco Display'), local('SF Pro Display'), local('-apple-system');
        }}
        @font-face {{
            font-family: 'SF Pro Text';
            src: local('San Francisco Text'), local('SF Pro Text'), local('-apple-system');
        }}
        :root {{
            --apple-blue: #0071e3;
            --apple-bg: #f5f5f7;
            --apple-text: #1d1d1f;
        }}
        
        body {{ 
            background: #000; 
            color: var(--apple-text); 
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
            -webkit-font-smoothing: antialiased;
        }}
        
        .font-display {{
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
        }}
        .font-text {{
            font-family: 'SF Pro Text', -apple-system, BlinkMacSystemFont, "Helvetica Neue", Helvetica, Arial, sans-serif;
        }}

        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: transparent; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.2); border-radius: 980px; }}
        ::-webkit-scrollbar-thumb:hover {{ background: rgba(0,0,0,0.4); }}
        
        input:focus, select:focus, button:focus {{ outline: 2px solid var(--apple-blue); outline-offset: 1px; }}
    </style>
</head>
<body class="min-h-screen flex flex-col font-text overflow-x-hidden">
    <!-- Glass Navigation -->
    <header class="fixed top-0 left-0 right-0 z-50 px-8 py-3 flex justify-between items-center" style="background: rgba(0,0,0,0.8); backdrop-filter: saturate(180%) blur(20px);">
        <div class="flex items-center gap-2">
            <svg class="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"/><path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"/></svg>
            <h1 class="text-white font-text text-[14px] font-semibold tracking-[-0.224px]">FaceSearch</h1>
        </div>
        <div class="flex items-center gap-6">
            <span class="text-white/60 font-text text-[12px] tracking-[-0.12px]">Active Nodes: <span class="text-white font-semibold">{cam_count}</span></span>
        </div>
    </header>

    <!-- Dark Hero Section -->
    <div class="w-full bg-black pt-32 pb-24 px-8 xl:px-16 flex flex-col items-center justify-center min-h-[60vh]">
        <div class="max-w-[1700px] w-full text-center mb-16">
            <h2 class="text-white font-display text-[56px] font-semibold leading-[1.07] tracking-[-0.28px] mb-4">Pro vision.</h2>
            <p class="text-white/60 font-text text-[21px] font-normal leading-[1.19] tracking-[0.231px] max-w-2xl mx-auto">High-performance deep learning across an array of cinematic sensor nodes.</p>
        </div>
        <div class="max-w-[1700px] w-full grid {grid_class} gap-12">
            {cam_items}
        </div>
    </div>

    <!-- Light Content Section -->
    <div class="w-full bg-[#f5f5f7] py-24 px-8 xl:px-16 flex-1">
        <div class="max-w-[1700px] mx-auto grid grid-cols-1 xl:grid-cols-12 gap-16">
            
            <!-- Registration Form -->
            <div class="xl:col-span-4 flex flex-col">
                <h3 class="font-display text-[40px] font-semibold text-[#1d1d1f] tracking-[-0.28px] leading-[1.10] mb-2">Enroll.</h3>
                <p class="font-text text-[17px] text-black/60 mb-8 tracking-[-0.374px] leading-[1.47]">Register a new identity to the biometric index.</p>
                
                <div class="space-y-4">
                    <input id="enNum" placeholder="Employee Code" autocomplete="off" class="w-full bg-white border border-transparent rounded-[12px] px-4 py-4 text-[17px] font-text text-[#1d1d1f] shadow-sm tracking-[-0.374px]">
                    <input id="enName" placeholder="Full Identity Name" autocomplete="off" class="w-full bg-white border border-transparent rounded-[12px] px-4 py-4 text-[17px] font-text text-[#1d1d1f] shadow-sm tracking-[-0.374px]">
                    <select id="enCam" class="w-full bg-white border border-transparent rounded-[12px] px-4 py-4 text-[17px] font-text text-[#1d1d1f] appearance-none shadow-sm cursor-pointer tracking-[-0.374px]">
                        {''.join([f'<option value="cam_{i}">NODE {i}</option>' for i in range(cam_count)])}
                    </select>
                    <button onclick="enroll()" class="w-full bg-[#0071e3] hover:bg-[#0077ED] active:bg-[#006bd6] text-white py-4 text-[17px] font-text rounded-[12px] transition-colors mt-2 shadow-sm">
                        Register Identity
                    </button>
                    <div class="text-center pt-2">
                         <a href="javascript:void(0)" class="text-[#0066cc] hover:underline text-[14px] tracking-[-0.224px] font-text">Learn more tracking policies ></a>
                    </div>
                </div>
            </div>

            <!-- Detection feed List -->
            <div class="xl:col-span-8 flex flex-col max-h-[800px]">
                <h3 class="font-display text-[40px] font-semibold text-[#1d1d1f] tracking-[-0.28px] leading-[1.10] mb-2">Timeline.</h3>
                <p class="font-text text-[17px] text-black/60 mb-8 tracking-[-0.374px] leading-[1.47] border-b border-black/5 pb-8">Real-time identification events.</p>
                
                <div id="detection-feed" class="space-y-4 overflow-y-auto pr-4 custom-scrollbar flex-1 pb-10">
                    <div class="flex flex-col items-center justify-center p-12 opacity-50">
                        <span class="text-[17px] font-text text-[#1d1d1f] tracking-[-0.374px]">Awaiting detections...</span>
                    </div>
                </div>
            </div>
            
        </div>
    </div>

    <script>
        async function fetchLogs() {{
            try {{
                const r = await fetch('/api/matches');
                const data = await r.json();
                const feed = document.getElementById('detection-feed');
                
                if (!data || data.length === 0) return;
                
                feed.innerHTML = data.map(m => `
                    <div class="p-5 bg-white rounded-[12px] shadow-[rgba(0,0,0,0.04)_0px_4px_12px_0px] border border-transparent flex items-center gap-6 transition-all hover:shadow-[rgba(0,0,0,0.12)_0px_8px_24px_0px] cursor-pointer">
                        <div class="w-16 h-16 rounded-[8px] overflow-hidden bg-[#f5f5f7]">
                            ${{m.base64_image ? `<img src="data:image/jpeg;base64,${{m.base64_image}}" class="w-full h-full object-cover">` : ''}}
                        </div>
                        <div class="flex-1">
                            <div class="text-[21px] font-display font-semibold text-[#1d1d1f] tracking-[0.231px] leading-[1.19] mb-1">${{m.name || 'Unknown'}}</div>
                            <div class="text-[14px] font-text text-black/60 tracking-[-0.224px]">
                                <span class="text-[#1d1d1f] font-semibold">${{m.employee_number || '---'}}</span>
                                <span class="mx-2 opacity-30">|</span>
                                <span>${{m.time}}</span>
                            </div>
                        </div>
                        <div class="text-[21px] font-display font-medium text-[#1d1d1f] tracking-[0.231px]">
                            ${{(m.confidence * 100).toFixed(0)}}<span class="text-[14px] text-black/40 ml-1">%</span>
                        </div>
                    </div>
                `).join('');
            }} catch(e) {{ console.error("Feed error:", e); }}
        }}

        async function enroll() {{
            const fd = new FormData();
            const id = document.getElementById('enNum').value;
            const name = document.getElementById('enName').value;
            if (!id || !name) {{ alert("REQUIREMENT: Please provide valid information."); return; }}
            
            fd.append('employee_number', id);
            fd.append('name', name);
            fd.append('cam_id', document.getElementById('enCam').value);
            
            try {{
                const r = await fetch('/enroll', {{method:'POST', body:fd}});
                const d = await r.json(); 
                if (r.ok) alert('Enrollment Successful:\\n' + d.name);
                else alert('Error:\\n' + (d.detail || 'No visible face detected.'));
            }} catch(e) {{ alert('Service Failure'); }}
        }}

        setInterval(fetchLogs, 3000);
        fetchLogs();
    </script>
</body>
</html>
    """
    return html

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host=config.API_HOST, port=config.API_PORT, reload=False)
