"""
rebuild_db.py – Full wipe & re-import of all face data
=======================================================
Steps:
  1. Delete all rows from the `patients` table.
  2. Clear the FAISS index blob from `faiss_index` table.
  3. Delete local FAISS index files.
  4. Walk `data/known person/{MRN}/` subfolders, extract ArcFace
     embeddings for every photo found, and upsert each patient.
  5. Rebuild FAISS index from the freshly inserted embeddings.

Usage:
    python rebuild_db.py
"""

import asyncio
import logging
import os
import sys
import glob
from pathlib import Path

import numpy as np

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("rebuild")

# ── local imports (must run from the face_search directory) ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import database as db
import engine


# ─────────────────────────────────────────────────────────────────────────────
#  Step 1 – wipe patients + FAISS blob from SQL
# ─────────────────────────────────────────────────────────────────────────────

def _wipe_db_sync():
    conn = db._get_conn()
    cur  = conn.cursor()

    log.info("Dropping patients table if it exists to clean schema…")
    try:
        cur.execute("DROP TABLE IF EXISTS patients")
        conn.commit()
        log.info("✅  Patients table dropped.")
    except Exception as e:
        log.warning("Could not drop patients table: %s. Trying DELETE instead.", e)
        try:
            cur.execute("DELETE FROM patients")
            conn.commit()
            log.info("✅  All patient rows deleted.")
        except Exception as e2:
            log.error("Failed to clear patients table: %s", e2)

    try:
        cur.execute("DELETE FROM faiss_index")
        conn.commit()
        log.info("✅  FAISS index blob cleared from SQL.")
    except Exception as e:
        log.error("Failed to clear faiss_index: %s", e)



# ─────────────────────────────────────────────────────────────────────────────
#  Step 2 – delete local index files
# ─────────────────────────────────────────────────────────────────────────────

def _wipe_local_indexes():
    data_dir = os.path.join(config.BASE_DIR, "data")
    patterns = ["faiss_hnsw.index", "faiss_flat.index", "faiss_search.index"]
    for name in patterns:
        p = os.path.join(data_dir, name)
        if os.path.exists(p):
            os.remove(p)
            log.info("🗑️  Deleted local index: %s", p)


# ─────────────────────────────────────────────────────────────────────────────
#  Step 3 – upload all photos from data/known person/{MRN}/
# ─────────────────────────────────────────────────────────────────────────────

PHOTO_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

async def _upload_all_photos():
    photos_root = Path(config.BASE_DIR) / "data" / "known person"
    if not photos_root.exists():
        log.error("Photo root not found: %s", photos_root)
        return

    mrn_dirs = sorted([d for d in photos_root.iterdir() if d.is_dir()])
    log.info("Found %d MRN folder(s) to process…", len(mrn_dirs))

    ok_count   = 0
    skip_count = 0

    for mrn_dir in mrn_dirs:
        mrn = mrn_dir.name          # folder name = MRN
        name = mrn                  # default display name = MRN

        # collect all image files inside this MRN folder
        image_files = []
        for ext in PHOTO_EXTENSIONS:
            image_files.extend(mrn_dir.glob(ext))
            image_files.extend(mrn_dir.glob(ext.upper()))

        if not image_files:
            log.warning("⚠️  No images found in %s – skipping.", mrn_dir)
            skip_count += 1
            continue

        log.info("Processing MRN=%s  (%d image(s))…", mrn, len(image_files))

        embeddings = []
        for img_path in image_files:
            try:
                with open(img_path, "rb") as f:
                    img_bytes = f.read()
                emb = await engine.extract_embedding(img_bytes)
                if emb is not None:
                    embeddings.append(emb)
                    log.info("  ✔ %s", img_path.name)
                else:
                    log.warning("  ✗ No face detected in %s", img_path.name)
            except Exception as exc:
                log.error("  ✗ Error reading %s: %s", img_path.name, exc)

        if not embeddings:
            log.warning("⚠️  No usable faces for MRN=%s – skipping.", mrn)
            skip_count += 1
            continue

        # diversity-select up to MULTI_EMB_COUNT representatives
        selected = engine.select_diverse_embeddings(embeddings, config.MULTI_EMB_COUNT)

        # mean embedding as the primary single vector
        mean_emb = np.mean(selected, axis=0).astype(np.float32)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm

        patient_id = await db.upsert_patient(
            name            = name,
            mrn             = mrn,
            embedding       = mean_emb,
            department      = "",
            num_images      = len(embeddings),
            multi_embeddings= selected,
        )
        log.info("  ✅  MRN=%s inserted (patient id=%s)", mrn, patient_id)
        ok_count += 1

    log.info("Upload complete: %d patient(s) inserted, %d skipped.", ok_count, skip_count)
    return ok_count


# ─────────────────────────────────────────────────────────────────────────────
#  Step 4 – rebuild FAISS index from the newly inserted data
# ─────────────────────────────────────────────────────────────────────────────

async def _rebuild_index():
    log.info("Rebuilding FAISS index from DB…")
    await engine.load_index()
    log.info("✅  FAISS index rebuilt.  Total vectors: %d",
             engine._index.ntotal if engine._index else 0)


# ─────────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    log.info("=" * 60)
    log.info("  FACE DATABASE FULL REBUILD")
    log.info("=" * 60)

    # 1. Wipe existing data (drops table to clean schema)
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, _wipe_db_sync)
    _wipe_local_indexes()

    # 2. Re-initialize DB schema (creates clean patients table with correct types & IDENTITY)
    await db.init_db()

    # 3. Clear in-memory cache
    db.clear_cache()
    log.info("✅  In-memory cache cleared.")

    # 4. Upload all photos
    inserted = await _upload_all_photos()

    if not inserted:
        log.error("❌  No patients were uploaded. Aborting index rebuild.")
        return

    # 5. Rebuild index
    await _rebuild_index()

    log.info("=" * 60)
    log.info("  REBUILD COMPLETE  –  %d patient(s) in index", inserted)
    log.info("  You can now start face_check.py normally.")
    log.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
