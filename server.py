"""
FastAPI server for Anima with real-time monitoring
"""
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from monitor import monitor
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Anima Server", version="1.0.0")

# Mount the static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include the monitor's WebSocket route
app.include_router(monitor.app)

@app.get("/")
async def root():
    """Serve the monitoring dashboard"""
    try:
        return FileResponse("static/index.html")
    except Exception as e:
        logger.error(f"Error serving dashboard: {e}")
        raise HTTPException(status_code=500, detail="Error serving dashboard")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "connections": len(monitor.manager.active_connections),
        "characters_tracked": len(monitor.character_metrics),
        "scene_metrics_count": len(monitor.scene_metrics)
    }

# Error handlers
@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled error: {exc}")
    return {"detail": "Internal server error"}
