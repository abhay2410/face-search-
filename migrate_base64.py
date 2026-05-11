"""
migrate_base64.py – Migrate legacy Base64 records or local photo folders to Face Index
===================================================================================
This script pulls images from a source table or a local folder, extracts ArcFace 
embeddings, and populates the searchable 'employees' table and FAISS index.

Usage (Base64 Mode):
    python migrate_base64.py --table SourceTable --mrn "MRN NUMBER" --name NAME --b64 BASE64

Usage (Folder Mode):
    python migrate_base64.py --folder ./photos/
"""

import asyncio
import base64
import logging
import argparse
import os
import glob
import numpy as np
from tqdm import tqdm
from pathlib import Path

import database as db
import engine
import config

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger("migration")

async def migrate_from_db(args):
    """Base64 Database Migration Mode"""
    log.info("Starting Database migration from table: %s", args.table)
    
    # Decide which connection to use
    if args.src_server:
        log.info("Connecting to EXTERNAL source database: %s", args.src_db)
        conn_str = (
            f"DRIVER={{{config.MSSQL_DRIVER}}};"
            f"SERVER={args.src_server};"
            f"DATABASE={args.src_db};"
            f"UID={args.src_user};"
            f"PWD={args.src_pass};"
            "TrustServerCertificate=yes;Encrypt=no;"
        )
        import pyodbc
        src_conn = pyodbc.connect(conn_str)
    else:
        log.info("Using DEFAULT system database.")
        src_conn = db._get_conn()
        
    cur = src_conn.cursor()
    query = f"SELECT [{args.mrn}], [{args.name}], [{args.b64}] FROM [{args.table}]"
    
    try:
        cur.execute(query)
        rows = cur.fetchall()
    except Exception as e:
        log.error("Failed to fetch data: %s", e)
        return

    log.info("Found %d records to process.", len(rows))
    
    for row in tqdm(rows, desc="Migrating DB"):
        mrn, name, b64_data = str(row[0]), str(row[1]), row[2]
        if not b64_data: continue
        
        try:
            if "," in b64_data: b64_data = b64_data.split(",")[1]
            img_bytes = base64.b64decode(b64_data)
            await process_record(mrn, name, img_bytes)
        except Exception as e:
            log.error("Error processing %s: %s", name, e)

async def migrate_from_folder(photos_root: str):
    """
    Local Folder Migration Mode (Nested Structure)
    Structure: [photos_root]/[MRN_FOLDER]/[RANDOM_FILES].jpg
    """
    log.info("Starting Nested Folder migration from: %s", photos_root)
    root = Path(photos_root)
    if not root.exists():
        log.error("Root folder not found: %s", photos_root)
        return

    # Get all subdirectories (each represents an MRN)
    mrn_dirs = [d for d in root.iterdir() if d.is_dir()]
    log.info("Found %d MRN folders.", len(mrn_dirs))

    for d in tqdm(mrn_dirs, desc="Migrating MRN Folders"):
        mrn = d.name
        name = mrn  # Default name to MRN
        
        # Collect all images in this MRN folder
        exts = ['*.jpg', '*.jpeg', '*.png', '*.webp']
        files = []
        for ext in exts:
            files.extend(list(d.glob(ext)))
        
        if not files:
            continue

        embs = []
        for f in files:
            try:
                with open(f, "rb") as img_file:
                    img_bytes = img_file.read()
                
                # Extract embedding for this specific file
                emb = await engine.extract_embedding(img_bytes)
                if emb is not None:
                    embs.append(emb)
            except Exception as e:
                log.error("Error processing %s in %s: %s", f.name, mrn, e)

        if embs:
            # Use diversity selection if we have many images (optional, but engine.select_diverse_embeddings is available)
            # For now, let's just use the first few or all up to MULTI_EMB_COUNT
            selected = engine.select_diverse_embeddings(embs, config.MULTI_EMB_COUNT)
            
            # Compute mean for the 'embedding' column
            mean_emb = np.mean(selected, axis=0)
            norm = np.linalg.norm(mean_emb)
            if norm > 0: mean_emb /= norm
            
            await db.upsert_employee(
                name=name,
                employee_code=mrn,
                embedding=mean_emb,
                num_images=len(files),
                multi_embeddings=selected
            )
        else:
            log.warning("No faces detected in folder: %s", mrn)

async def process_record(mrn: str, name: str, img_bytes: bytes):
    """Core extraction and upsert logic"""
    emb = await engine.extract_embedding(img_bytes)
    if emb is not None:
        await db.upsert_employee(
            name=name,
            employee_code=mrn,
            embedding=emb,
            num_images=1
        )
    else:
        log.debug("No face detected for %s", mrn)

async def main():
    parser = argparse.ArgumentParser(description="Migrate Base64 or Folders to Face Index")
    
    # Mode Choice: Folder or DB
    parser.add_argument("--folder", help="Path to 'PHOTOS' folder containing [MRN] subfolders")
    
    # DB Mode Source Data
    parser.add_argument("--table",  default="YourTable", help="Source table name")
    parser.add_argument("--mrn",    default="MRN NUMBER", help="MRN column")
    parser.add_argument("--name",   default="NAME",       help="Name column")
    parser.add_argument("--b64",    default="BASE 64",    help="Base64 column")
    
    # External Source DB Details (Optional)
    parser.add_argument("--src-server", help="External source SQL Server address")
    parser.add_argument("--src-db",     help="External source Database name")
    parser.add_argument("--src-user",   help="External source Username")
    parser.add_argument("--src-pass",   help="External source Password")
    
    args = parser.parse_args()
    await db.init_db()

    if args.folder:
        await migrate_from_folder(args.folder)
    else:
        await migrate_from_db(args)

    log.info("Rebuilding FAISS index...")
    await engine.load_index()
    log.info("Migration Complete.")

if __name__ == "__main__":
    asyncio.run(main())
