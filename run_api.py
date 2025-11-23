"""
Simple script to run the FastAPI server.
Run from project root: python run_api.py
"""
import uvicorn
from pathlib import Path
import sys

# Add src to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src" / "deployment"))

if __name__ == "__main__":
    uvicorn.run(
        "src.deployment.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

