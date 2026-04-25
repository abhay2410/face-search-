import PyInstaller.__main__
import os

# We build as --onedir (a folder containing the EXE and all DLLs/models).
# A --onefile build would extract a 1GB+ payload every time it runs, which takes forever to start.

build_args = [
    'app.py',
    '--name=FaceSearchAI',
    '--onedir',          # Creates a folder "dist/FaceSearchAI"
    '--noconfirm',       # Overwrite output directory
    '--clean',           # Clean cache
    # Add hidden imports for FastAPI / Uvicorn since PyInstaller often misses them
    '--hidden-import=uvicorn.logging',
    '--hidden-import=uvicorn.loops',
    '--hidden-import=uvicorn.loops.auto',
    '--hidden-import=uvicorn.protocols',
    '--hidden-import=uvicorn.protocols.http',
    '--hidden-import=uvicorn.protocols.http.auto',
    '--hidden-import=uvicorn.protocols.websockets',
    '--hidden-import=uvicorn.protocols.websockets.auto',
    '--hidden-import=uvicorn.lifespan',
    '--hidden-import=uvicorn.lifespan.on',
    '--hidden-import=uvicorn.lifespan.off',
    '--hidden-import=insightface',
    '--hidden-import=onnxruntime',
    '--hidden-import=faiss',
    
    # We must explicitly pack the data folder containing the insightface models
    # Windows syntax for add-data is Source;Destination
    '--add-data=data;data',
    
    # We ignore the .env file in the build because the user will want to put a .env file 
    # NEXT to the executable in the dist folder, so they can edit it without recompiling!
]

if __name__ == "__main__":
    print("Starting PyInstaller Build...")
    print("This will take a few minutes as it packages the AI models, OpenCV, and FastAPI.")
    PyInstaller.__main__.run(build_args)
    print("\n--- BUILD COMPLETE ---")
    print("You can find your executable inside the 'dist/FaceSearchAI' folder.")
    print("Make sure to COPY your '.env' file into the 'dist/FaceSearchAI' folder before running!")
