"""
Face Search Utility
==================
Standalone script to find an employee ID from a face image.
Uses data and logic from the main application.

Usage:
    python search_face.py <path_to_image>
"""

import asyncio
import os
import sys
import cv2
import numpy as np
from pathlib import Path

# Silence C++ logs before importing ML modules
try:
    import silencer
    silencer.silence_cpp_logs()
except ImportError:
    pass

# Import core modules
import config
import database
import engine

async def search_by_image(image_path: str):
    # 1. Initialize Database
    try:
        await database.init_db()
    except Exception as e:
        print(f"DATABASE_ERROR: {e}")
        return

    # 2. Load FAISS Index
    loaded = await engine.load_index_from_disk()
    if not loaded:
        # Rebuild from DB if needed
        await engine.load_index()

    # 3. Read and Process Image
    if not os.path.exists(image_path):
        print(f"ERROR: Image not found at {image_path}")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"ERROR: Could not decode image at {image_path}")
        return

    # 4. Extract Embedding
    # We use enrol_mode=True for higher quality detection in this standalone tool
    emb = await engine.extract_embedding(img)
    if emb is None:
        print("NOT_FOUND: No face detected in image")
        return

    # 5. Search Index
    emp_id, distance = engine.search_index(emb)

    if emp_id:
        print(f"MATCH: Found Employee ID {emp_id} (Similarity: {distance:.4f})")
    else:
        print(f"NOT_FOUND: No matching employee found (Best match: {distance:.4f}, Threshold: {config.FAISS_COSINE_THRESHOLD})")

def main():
    if len(sys.argv) < 2:
        print("Usage: python search_face.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    asyncio.run(search_by_image(image_path))

if __name__ == "__main__":
    main()
