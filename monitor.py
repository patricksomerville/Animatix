"""
Real-time monitoring system for Animatix
Tracks and visualizes power dynamics, performance metrics, and scene structure
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, APIRouter
from typing import Dict, List, Optional, Any
import asyncio
import json
from dataclasses import dataclass, asdict
import numpy as np
import logging
from datetime import datetime
from pydantic import BaseModel, Field, validator, field_validator
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MetricsBase(BaseModel):
    timestamp: float = Field(default_factory=lambda: time.time())
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.timestamp()
        }

class PerformanceMetrics(MetricsBase):
    """Real-time performance metrics for characters"""
    power_level: float = Field(..., ge=-1, le=1)
    emotional_intensity: float = Field(..., ge=0, le=1)
    gesture_frequency: float = Field(..., ge=0, le=1)
    gaze_stability: float = Field(..., ge=0, le=1)
    dialogue_pace: float = Field(..., ge=0, le=1)
    
    @field_validator('*')
    @classmethod
    def validate_metrics(cls, v, info):
        if not isinstance(v, (int, float)):
            raise ValueError(f'{info.field_name} must be a number')
        return float(v)

class SceneMetrics(MetricsBase):
    """Overall scene metrics"""
    tension: float = Field(..., ge=0, le=1)
    power_gradient: float = Field(..., ge=-1, le=1)
    emotional_coherence: float = Field(..., ge=0, le=1)
    beat_progression: List[float] = Field(default_factory=list)
    
    @validator('beat_progression')
    def validate_progression(cls, v):
        if not all(isinstance(x, (int, float)) for x in v):
            raise ValueError('Beat progression must contain only numbers')
        if not all(-1 <= x <= 1 for x in v):
            raise ValueError('Beat progression values must be between -1 and 1')
        return [float(x) for x in v]

class ConnectionManager:
    """Manages WebSocket connections and error handling"""
    
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.connection_attempts: Dict[str, List[float]] = {}
        self.MAX_RETRY_INTERVAL = 30  # Maximum seconds between retries
        self.ATTEMPT_WINDOW = 60  # Time window for counting attempts
        self.MAX_ATTEMPTS = 5  # Maximum attempts within window
        
    async def connect(self, websocket: WebSocket):
        """Handle new connection with rate limiting"""
        client = websocket.client.host
        current_time = time.time()
        
        # Clean up old attempts
        if client in self.connection_attempts:
            self.connection_attempts[client] = [
                t for t in self.connection_attempts[client]
                if current_time - t < self.ATTEMPT_WINDOW
            ]
            
            # Check rate limiting
            if len(self.connection_attempts[client]) >= self.MAX_ATTEMPTS:
                retry_after = self.ATTEMPT_WINDOW - (current_time - self.connection_attempts[client][0])
                logger.warning(f"Rate limit exceeded for {client}. Retry after {retry_after:.1f}s")
                return False
                
        # Accept connection
        await websocket.accept()
        self.active_connections.append(websocket)
        
        # Record attempt
        if client not in self.connection_attempts:
            self.connection_attempts[client] = []
        self.connection_attempts[client].append(current_time)
        
        logger.info(f"Client connected: {client}")
        return True
        
    def disconnect(self, websocket: WebSocket):
        """Handle connection closure"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info(f"Client disconnected: {websocket.client.host}")
            
    async def broadcast(self, message: Any):
        """Broadcast message to all connected clients"""
        if not self.active_connections:
            return
            
        # Prepare message once
        if isinstance(message, BaseModel):
            message = message.dict()
        elif not isinstance(message, (str, bytes)):
            message = json.dumps(message)
            
        # Send to all clients
        for connection in self.active_connections:
            try:
                if isinstance(message, str):
                    await connection.send_text(message)
                elif isinstance(message, bytes):
                    await connection.send_bytes(message)
                else:
                    await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {connection.client.host}: {e}")
                await self.disconnect(connection)

class DramaticMonitor:
    """Real-time monitoring and visualization of dramatic elements"""
    
    def __init__(self):
        self.app = APIRouter()
        self.manager = ConnectionManager()
        self.character_metrics: Dict[str, List[PerformanceMetrics]] = {}
        self.scene_metrics: List[SceneMetrics] = []
        self.HISTORY_LIMIT = 1000  # Maximum number of historical metrics to keep
        self._setup_routes()
        
    def _setup_routes(self):
        @self.app.websocket("/ws/monitor")
        async def websocket_endpoint(websocket: WebSocket):
            if not await self.manager.connect(websocket):
                return
                
            try:
                while True:
                    data = await websocket.receive_text()
                    if data == "ping":
                        await websocket.send_text("pong")
                    elif data == "history":
                        # Send historical data
                        await self._send_history(websocket)
            except WebSocketDisconnect:
                self.manager.disconnect(websocket)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                self.manager.disconnect(websocket)
                
    async def _send_history(self, websocket: WebSocket):
        """Send historical metrics data to client"""
        try:
            history = {
                "characters": {
                    name: [m.dict() for m in metrics[-100:]]  # Last 100 metrics
                    for name, metrics in self.character_metrics.items()
                },
                "scene": [m.dict() for m in self.scene_metrics[-100:]]
            }
            await websocket.send_json(history)
        except Exception as e:
            logger.error(f"Error sending history: {e}")
                
    def update_character_metrics(self, name: str, metrics: Dict):
        """Update metrics for a specific character"""
        try:
            metrics_obj = PerformanceMetrics(**metrics)
            if name not in self.character_metrics:
                self.character_metrics[name] = []
            self.character_metrics[name].append(metrics_obj)
            
            # Trim history if needed
            if len(self.character_metrics[name]) > self.HISTORY_LIMIT:
                self.character_metrics[name] = self.character_metrics[name][-self.HISTORY_LIMIT:]
                
            # Broadcast update
            asyncio.create_task(self.manager.broadcast({
                "type": "character_update",
                "name": name,
                "metrics": metrics_obj.dict()
            }))
            
        except Exception as e:
            logger.error(f"Error updating character metrics: {e}")
            
    def update_scene_metrics(self, metrics: Dict):
        """Update overall scene metrics"""
        try:
            metrics_obj = SceneMetrics(**metrics)
            self.scene_metrics.append(metrics_obj)
            
            # Trim history if needed
            if len(self.scene_metrics) > self.HISTORY_LIMIT:
                self.scene_metrics = self.scene_metrics[-self.HISTORY_LIMIT:]
                
            # Broadcast update
            asyncio.create_task(self.manager.broadcast({
                "type": "scene_update",
                "metrics": metrics_obj.dict()
            }))
            
        except Exception as e:
            logger.error(f"Error updating scene metrics: {e}")
            
    def calculate_tension(self, power_shifts: List[float]) -> float:
        """Calculate scene tension from power shift history"""
        if not power_shifts:
            return 0.0
            
        # Use exponential moving average for recent shifts
        weights = np.exp(np.linspace(-1, 0, len(power_shifts)))
        weighted_shifts = np.abs(power_shifts) * weights
        return float(np.sum(weighted_shifts) / np.sum(weights))
        
    def calculate_coherence(self, emotional_states: List[str]) -> float:
        """Calculate emotional coherence of scene"""
        if not emotional_states:
            return 1.0
            
        # Count transitions between emotions
        transitions = sum(1 for i in range(len(emotional_states)-1)
                        if emotional_states[i] != emotional_states[i+1])
        
        # More transitions = less coherence
        return 1.0 - (transitions / (len(emotional_states) - 1) if len(emotional_states) > 1 else 0)

monitor = DramaticMonitor()
