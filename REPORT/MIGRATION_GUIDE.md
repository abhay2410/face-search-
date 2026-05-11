# Face Search Migration & Enrollment Guide

This documentation covers the usage and architecture of the migration tools designed to populate your Face Search Index from existing data sources (SQL Databases or Local Folders).

---

## 1. Overview
The migration system (`migrate_base64.py`) is a standalone utility designed to ingest historical identity data and convert it into high-performance facial embeddings. It supports:
- **Bulk SQL Ingestion**: Converting Base64 strings from any MS SQL table.
- **Bulk Folder Ingestion**: Importing photos from a structured file system.
- **Automatic Indexing**: Rebuilding the FAISS HNSW search map after migration.

---

## 2. Migration Mode A: SQL Database (Base64)
Use this mode if your identity photos are stored as Base64-encoded strings in a database table.

### Basic Usage (Internal DB)
If the data is in the **same database** as your face search system:
```powershell
python migrate_base64.py --table "SourceTable" --mrn "MRN_COL" --name "NAME_COL" --b64 "BASE64_COL"
```

### Advanced Usage (External Source DB)
If you are pulling data from a **different SQL server**:
```powershell
python migrate_base64.py `
  --src-server "192.168.1.50" `
  --src-db "LegacyRecords" `
  --src-user "sa" `
  --src-pass "password" `
  --table "Employees" --mrn "EmpID" --name "FullName" --b64 "PhotoData"
```

---

## 3. Migration Mode B: Nested Folder Structure
Use this mode if you have a folder containing subdirectories for each person. This is the **preferred method** for higher accuracy because it supports multiple photos per person.

### Folder Structure Requirement
```text
PHOTOS_ROOT/
├── 1001/              <-- This folder name is the MRN
│   ├── front.jpg
│   ├── side_view.png
│   └── old_photo.webp
├── 1002/
│   └── image_1.jpg
└── ...
```

### Usage Command
```powershell
python migrate_base64.py --folder "C:/Path/To/Photos"
```

---

## 4. Technical Concepts

### ArcFace Embeddings
The system uses **ArcFace (via InsightFace)** to convert pixels into a 512-dimensional vector. This vector captures unique facial geometry and is robust against changes in lighting, age, and minor accessories.

### Diversity Selection
When migrating from a folder with multiple images (Mode B):
1. The script extracts embeddings for **every image** in the person's folder.
2. It uses a **Diversity Selection** algorithm to pick the most representative faces (defaulting to the 3 most distinct ones).
3. This ensures the index is accurate without needing hundreds of redundant vectors.

### FAISS HNSW Indexing
After extraction, the script invokes the **FAISS (Facebook AI Similarity Search)** engine:
- It builds a **Hierarchical Navigable Small World (HNSW)** index.
- This allows the system to perform "Nearest Neighbor" searches across thousands of identities in under 10ms.
- The index is automatically saved to disk (`./data/faiss_hnsw.index`) and synced to the SQL `faiss_index` table.

---

## 5. Frequently Asked Questions

**Q: What happens if a face isn't detected in a migration photo?**
A: The script logs a warning for that specific file and continues. That record will not be added to the index.

**Q: Does it handle multiple photos per MRN?**
A: Yes, in Folder Mode (Mode B), it automatically processes all images in the MRN directory and creates a combined "Weighted Mean" embedding.

**Q: How do I update the index after a migration?**
A: You don't need to! The script automatically calls the indexing engine as its final step.

**Q: Can I run migration multiple times?**
A: Yes. The system uses an **Upsert** logic. If the MRN already exists, it will update the existing record with the new biometric data.

---

> [!TIP]
> **Performance Note**: If you have a very large database (>5,000 people), ensure your computer has a dedicated NVIDIA GPU (CUDA) to speed up the extraction process.
