"""
FastAPI Server Runner for HireGenie
Run this file to start the server: python run.py
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "5000")),
        reload=os.getenv("UVICORN_RELOAD", "false").lower() == "true",
        log_level="info"
    )
