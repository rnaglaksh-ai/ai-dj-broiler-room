import os
import uuid
import asyncio
from pathlib import Path
from typing import Dict, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from stem_extractor import DJStemExtractor

MAX_UPLOAD_SIZE_MB = 50
MAX_EMBEDDING_DIM = 512
MAX_PLAYLIST_SIZE = 50
MAX_INTERMEDIATE_COUNT = 10

app = FastAPI(title="AI DJ Stem Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

UPLOAD_DIR = "stems_cache/uploads"
STEMS_DIR  = "stems_cache/stems"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(STEMS_DIR,  exist_ok=True)

# ── Global state ──────────────────────────────────────────────────────────────
extractor: Optional[DJStemExtractor] = None

# Track HQ upgrade status: track_id → {"status": "pending"|"ready", "stems": {...}}
hq_status: Dict[str, dict] = {}


@app.on_event("startup")
def startup():
    global extractor
    print("[*] Initializing Antigravity Dual-Engine Stem Extractor...")
    extractor = DJStemExtractor()   # auto-selects CUDA/MPS/CPU
    print("[*] Stem server ready! (Local High-Fidelity Pipeline: HTDemucs exclusively)")


# ── /separate — Dual-engine endpoint ─────────────────────────────────────────
@app.post("/separate")
async def separate_track(
    file: UploadFile = File(...),
    quality: str = Form("balanced"),  # fast=lite-only | balanced=lite+hq-bg | high=hq-sync
):
    """
    Upload an audio file for stem separation.

    quality="fast"     → Lite engine only (MDX-Net, ~5s). Returns immediately.
    quality="balanced" → Lite engine now + HQ engine in background. Poll /separate/status/{track_id}.
    quality="high"     → HQ engine synchronously (hdemucs_mmi, ~60s). Blocks until done.
    """
    # File size guard
    content = await file.read()
    if len(content) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File too large. Max {MAX_UPLOAD_SIZE_MB}MB.")

    safe_filename = "".join(c for c in (file.filename or "upload") if c.isalnum() or c in "._-")[:100]
    track_id = uuid.uuid4().hex[:12]

    upload_path = os.path.join(UPLOAD_DIR, f"{track_id}_{safe_filename}")
    with open(upload_path, "wb") as f:
        f.write(content)

    stem_out_dir = os.path.join(STEMS_DIR, track_id)

    # ── Universal Format Ingestion (.vdjstems) ───────────────────────────────
    if safe_filename.lower().endswith('.vdjstems'):
        try:
            print(f"[*] Intercepting .vdjstems container for {track_id}...")
            # Natively extract streams without AI model
            vdj_paths = await asyncio.to_thread(
                extractor.extract_vdjstems,
                upload_path,
                stem_out_dir,
            )
            print(f"[*] Fast extraction complete for {track_id}")
            
            stems = {name: f"/stems/{track_id}/{Path(p).name}" for name, p in vdj_paths.items()}
            return JSONResponse({
                "track_id": track_id,
                "stems": stems,
                "quality": "hq_native",
                "status": "complete",
                "hq_available": False,
            })
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"VDJ format extraction failed: {str(e)}")

    # ── Fast / Balanced: run Lite engine first (Normal processing) ───────────
    if quality in ("fast", "balanced"):
        try:
            print(f"[*] Starting Lite separation for {track_id}...")
            lite_paths = await asyncio.to_thread(
                extractor.separate_lite,
                upload_path,
                stem_out_dir,
            )
            print(f"[*] Lite separation complete for {track_id}")
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Lite separation failed: {str(e)}")

        stems = {name: f"/stems/{track_id}/{Path(p).name}" for name, p in lite_paths.items()}
        response = {
            "track_id": track_id,
            "stems": stems,
            "quality": "lite",
            "status": "complete",
            "hq_available": False,
        }

        # Balanced: kick off HQ in background
        if quality == "balanced":
            hq_status[track_id] = {"status": "pending", "stems": {}}

            def _on_hq_ready(hq_paths: dict):
                hq_stems = {name: f"/stems/{track_id}/{Path(p).name}" for name, p in hq_paths.items()}
                hq_status[track_id] = {"status": "ready", "stems": hq_stems}
                print(f"[HQ] Hot-swap ready for {track_id}")

            extractor.separate_hq_background(upload_path, stem_out_dir, on_hq_ready=_on_hq_ready)
            response["hq_available"] = True
            response["hq_status_url"] = f"/separate/status/{track_id}"

        return JSONResponse(response)

    # ── High: synchronous HQ ─────────────────────────────────────────────────
    try:
        hq_paths = await asyncio.to_thread(
            extractor._separate_demucs,
            upload_path,
            stem_out_dir,
            shifts=2,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HQ separation failed: {str(e)}")

    stems = {name: f"/stems/{track_id}/{Path(p).name}" for name, p in hq_paths.items()}
    return JSONResponse({
        "track_id": track_id,
        "stems": stems,
        "quality": "hq",
        "status": "complete",
        "hq_available": False,
    })


# ── /separate/status/{track_id} — HQ hot-swap polling ────────────────────────
@app.get("/separate/status/{track_id}")
async def separate_status(track_id: str):
    """
    Poll this endpoint after a 'balanced' separation to check if HQ stems are ready.
    When status == 'ready', swap the stem URLs in the frontend audio buffers.

    Response:
        { "status": "pending" | "ready", "stems": { "vocals": "/stems/...", ... } }
    """
    if ".." in track_id or "/" in track_id or "\\" in track_id:
        raise HTTPException(status_code=400, detail="Invalid track_id")

    if track_id not in hq_status:
        raise HTTPException(status_code=404, detail="No HQ job found for this track_id")

    return JSONResponse(hq_status[track_id])


# ── /check_cache/{track_id} — Cache-First Logic ──────────────────────────────
@app.get("/check_cache/{track_id}")
async def check_cache(track_id: str):
    """
    Check if high-quality stems already exist in cache.
    Returns: {"cached": bool, "stems": {...}}
    """
    if ".." in track_id or "/" in track_id:
        raise HTTPException(status_code=400, detail="Invalid track_id")
    
    stem_dir = os.path.join(STEMS_DIR, track_id)
    required = ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]
    
    if os.path.exists(stem_dir):
        # Verify all 4 stems exist
        missing = [f for f in required if not os.path.exists(os.path.join(stem_dir, f))]
        if not missing:
            # All present!
            stems = {
                name.replace('.wav', ''): f"/stems/{track_id}/{name}" 
                for name in required
            }
            return {
                "cached": True, 
                "stems": stems,
                "msg": "[::: ] Stems Ready (Cache Hit)"
            }
            
    return {"cached": False, "msg": "[ . ] Cache Miss - Processing required"}


# ── /stems/{track_id}/{filename} — Serve stem WAV ────────────────────────────
@app.get("/stems/{track_id}/{filename}")
async def serve_stem(track_id: str, filename: str):
    """Serve a separated stem WAV file with path traversal protection."""
    if ".." in track_id or ".." in filename or "/" in track_id or "\\" in track_id:
        raise HTTPException(status_code=400, detail="Invalid path")

    file_path = os.path.join(STEMS_DIR, track_id, filename)
    resolved = os.path.realpath(file_path)
    if not resolved.startswith(os.path.realpath(STEMS_DIR)):
        raise HTTPException(status_code=403, detail="Access denied")

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Stem not found")

    return FileResponse(file_path, media_type="audio/wav")


# ── Similarity engine endpoints ───────────────────────────────────────────────
from similarity_engine import engine as sim_engine

@app.post("/similarity/register")
async def register_track(data: dict):
    """Register a track's embeddings and key with validation."""
    track_id        = data.get("track_id")
    embedding       = data.get("embedding")
    artist_embedding = data.get("artist_embedding")
    key             = data.get("key")

    if not track_id or not isinstance(track_id, str) or len(track_id) > 100:
        raise HTTPException(status_code=400, detail="Invalid track_id")
    if not embedding or not isinstance(embedding, list) or len(embedding) > MAX_EMBEDDING_DIM:
        raise HTTPException(status_code=400, detail="Invalid embedding")
    if not artist_embedding or not isinstance(artist_embedding, list) or len(artist_embedding) > MAX_EMBEDDING_DIM:
        raise HTTPException(status_code=400, detail="Invalid artist_embedding")

    sim_engine.add_track(track_id, embedding, artist_embedding, key)
    return {"status": "registered"}


@app.get("/similarity/path")
async def get_path(start_id: str, end_id: str, intermediate_count: int = 3, creativity: float = 0.5, noise: float = 0.0):
    """Join the Dots with validated parameters."""
    intermediate_count = max(1, min(intermediate_count, MAX_INTERMEDIATE_COUNT))
    creativity = max(0.0, min(creativity, 1.0))
    noise      = max(0.0, min(noise, 1.0))
    try:
        path = sim_engine.find_path(start_id, end_id, intermediate_count, creativity, noise)
        return {"path": path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/similarity/most_similar")
async def most_similar(data: dict):
    """Deej-AI: Find most similar tracks with validated parameters."""
    positive_ids = data.get("positive_ids", [])
    if not isinstance(positive_ids, list) or len(positive_ids) > 20:
        raise HTTPException(status_code=400, detail="Invalid positive_ids")

    topn       = max(1, min(int(data.get("topn", 5)), 50))
    noise      = max(0.0, min(float(data.get("noise", 0.0)), 1.0))
    creativity = max(0.0, min(float(data.get("creativity", 0.5)), 1.0))
    exclude    = data.get("exclude", [])
    if not isinstance(exclude, list):
        exclude = []

    results = sim_engine.most_similar(positive_ids, topn, noise, creativity, exclude)
    return {"results": [{"track_id": tid, "score": score} for tid, score in results]}


@app.post("/similarity/make_playlist")
async def make_playlist(data: dict):
    """Deej-AI: Generate playlist with validated parameters."""
    seed_ids = data.get("seed_ids", [])
    if not isinstance(seed_ids, list) or len(seed_ids) == 0 or len(seed_ids) > 20:
        raise HTTPException(status_code=400, detail="Invalid seed_ids (1-20 required)")

    size     = max(1, min(int(data.get("size", 10)), MAX_PLAYLIST_SIZE))
    lookback = max(1, min(int(data.get("lookback", 3)), 10))
    noise    = max(0.0, min(float(data.get("noise", 0.0)), 1.0))
    creativity = max(0.0, min(float(data.get("creativity", 0.5)), 1.0))

    playlist = sim_engine.make_playlist(seed_ids, size, lookback, noise, creativity)
    return {"playlist": playlist}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "engines": {
            "lite": "HTDemucs (Fast Mode, shifts=0)",
            "hq":   "HTDemucs (Studio Mode, shifts=2)",
        },
        "device": extractor.device if extractor else "loading",
        "similarity_engine": "deej-ai-v2",
        "registered_tracks": len(sim_engine.tracks),
        "hq_jobs_pending": sum(1 for v in hq_status.values() if v["status"] == "pending"),
    }


# ── RGB Waveform Endpoint (DJ Master Course) ─────────────────────────────────
@app.get("/analyze/rgb/{track_id}")
async def analyze_rgb(track_id: str):
    """
    Generate 'Spectral Vision' RGB waveform, Energy Score, and Structure Tags.
    """
    if ".." in track_id or "/" in track_id:
        raise HTTPException(status_code=400, detail="Invalid track_id")

    # Find the file in upload dir
    # Filename format: {track_id}_{safe_filename}
    files = list(Path(UPLOAD_DIR).glob(f"{track_id}_*"))
    if not files:
        raise HTTPException(status_code=404, detail="Track not found")
    
    file_path = str(files[0])
    
    if not extractor:
        raise HTTPException(status_code=503, detail="Extractor not initialized")
        
    try:
        # Run in thread pool to avoid blocking async event loop
        result = await asyncio.to_thread(extractor.generate_rgb_waveform, file_path)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Dynamic Phrase Analysis Endpoint ─────────────────────────────────────────
from phrase_analyzer import analyze_phrasing

@app.get("/analyze/structure/{track_id}")
async def analyze_structure(track_id: str):
    """
    Generate professional DJ phrasing structure metadata.
    Detects intros, build-ups, drops, breakdowns, and outros on an exact beatgrid.
    """
    if ".." in track_id or "/" in track_id:
        raise HTTPException(status_code=400, detail="Invalid track_id")

    files = list(Path(UPLOAD_DIR).glob(f"{track_id}_*"))
    if not files:
        raise HTTPException(status_code=404, detail="Track not found")
    
    file_path = str(files[0])
    
    try:
        # Run in thread pool to avoid blocking async event loop
        result = await asyncio.to_thread(analyze_phrasing, file_path)
        if result is None:
            raise Exception("Phrasing engine computation failed")
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    print("=" * 50)
    print("  Antigravity AI DJ Stem Server")
    print("  http://localhost:8000")
    print("  Local High-Fidelity Pipeline: HTDemucs")
    print("=" * 50)
    uvicorn.run(app, host="0.0.0.0", port=8000)
