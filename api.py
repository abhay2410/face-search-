
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import database as db
import engine
import config
import logging
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI(title="Face Search API", version="1.0.0")

# Security
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    if credentials.username != config.AUTH_USERNAME or credentials.password != config.AUTH_PASSWORD:
        raise HTTPException(status_code=401, detail="Unauthorized")
    return credentials.username

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup():
    await db.init_db()
    # Pre-load engine and index
    loaded = await engine.load_index_from_disk()
    if not loaded:
        await engine.load_index()
    log.info("API Startup Complete.")

@app.get("/")
async def root():
    return {"message": "Face Search API is running", "version": "1.0.0"}

@app.get("/status")
async def get_status(user: str = Depends(authenticate)):
    return {
        "status": "online",
        "vram_gb": round(engine._VRAM_GB, 2),
        "device": engine._device_str,
        "index_size": engine._index.ntotal if engine._index else 0
    }

@app.get("/matches/recent")
async def get_recent(limit: int = 20, user: str = Depends(authenticate)):
    matches = await db.get_recent_matches(limit)
    return {"matches": matches}

@app.post("/index/reload")
async def reload_index(user: str = Depends(authenticate)):
    await engine.load_index()
    return {"message": "Index reloaded from database"}

@app.get("/patients")
async def get_patients(user: str = Depends(authenticate)):
    # This would typically be a more detailed list, but for now we'll just return counts
    all_vecs = await db.get_all_multi_embeddings()
    return {"count": len(all_vecs)}

if __name__ == "__main__":
    uvicorn.run(app, host=config.API_HOST, port=config.API_PORT)
