"""
Enhanced Backend Server
Provides FastAPI endpoints and WebSocket connections for the Animatix system
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import asyncio
import json
import spacy
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
from pipeline import AnimaticPipeline, SceneBeat

# Initialize FastAPI app
app = FastAPI(title="Animatix Backend")

# Configure CORS for web access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins (for development, consider restricting in production)
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

# Initialize spaCy NLP model
try:
    nlp = spacy.load("en_core_web_trf") # Load transformer-based model for better accuracy
except OSError:
    logging.error("Failed to load spaCy model 'en_core_web_trf'.")
    logging.error("Please ensure it is installed. Try: python -m spacy download en_core_web_trf")
    nlp = None # Handle case where model fails to load

# --- Data Models ---
class Scene(BaseModel):
    id: str
    title: str
    content: str
    metadata: Dict
    created_at: datetime
    updated_at: datetime

class Character(BaseModel):
    id: str
    name: str
    description: str
    voice_profile: Dict # Details about voice characteristics
    performance_profile: Dict # Details about performance style

class SceneAnalysis(BaseModel):
    scene_id: str
    emotional_beats: List[Dict] # Key emotional moments in the scene
    power_dynamics: Dict[str, float] # Power level of each character
    tension_curve: List[float] # Tension level throughout the scene
    scene_turns: List[int] # Indices of significant scene turns

class RenderSettings(BaseModel):
    resolution: tuple = (1920, 1080)
    frame_rate: int = 24
    quality: str = "high" # Render quality setting
    format: str = "mp4" # Output video format

# --- WebSocket Connection Manager ---
class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accepts a new WebSocket connection."""
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        """Removes a WebSocket connection."""
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: dict):
        """Broadcasts a JSON message to all active WebSocket connections."""
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager() # Instantiate connection manager

# --- Scene Processing Pipeline ---
class SceneProcessor:
    """Processes scene content to extract and analyze dramatic elements."""
    def __init__(self):
        self.pipeline = AnimaticPipeline()  # Initialize Animatic Pipeline
        self.cache = {} # Consider using a more robust caching mechanism

    async def process_scene(self, scene: Scene) -> SceneAnalysis:
        """
        Analyzes a scene's script content to extract emotional beats,
        power dynamics, tension curve, and scene turns.
        """
        scene_beats = await self.pipeline.process_script(scene.content)
        
        emotional_beats = []
        power_dynamics = {}
        tension_curve = []
        scene_turns = []

        for beat in scene_beats:
            emotions = {}
            for char in beat.shot.characters:
                emotions[char.name] = char.emotional_state
            
            intensity = max([char.power_level for char in beat.shot.characters] or [0]) # Handle empty character list

            emotional_beats.append({
                "text": beat.dialogue,
                "emotions": emotions,
                "intensity": intensity
            })
                
            for char in beat.shot.characters:
                power_dynamics[char.name] = char.power_level
            
            tension_curve.append(beat.power_shift)
        
        # Detect scene turns based on power shifts
        power_shifts = [beat.power_shift for beat in scene_beats]
        threshold = np.mean(power_shifts) + 2 * np.std(power_shifts) if power_shifts else 0 # Avoid std on empty list
        
        for i, beat in enumerate(scene_beats):
            if abs(beat.power_shift) > threshold:
                scene_turns.append(i)
        
        return SceneAnalysis(
            scene_id=scene.id,
            emotional_beats=emotional_beats,
            power_dynamics=power_dynamics,
            tension_curve=tension_curve,
            scene_turns=scene_turns
        )
        
    def _analyze_emotions(self, doc) -> List[Dict]:
        """
        [Placeholder] Extracts emotional beats from text using NLP.
        Currently a stub, needs implementation with NLP techniques.
        """
        beats = []
        for sent in doc.sents:
            # Placeholder emotion extraction
            emotions = self._extract_emotions(sent)
            beats.append({
                "text": sent.text,
                "emotions": emotions,
                "intensity": self._calculate_intensity(sent)
            })
        return beats
    
    def _analyze_power_dynamics(self, doc) -> Dict[str, float]:
        """
        [Placeholder] Analyzes character power dynamics using NLP.
        Currently a stub, needs more sophisticated NLP implementation.
        """
        power_levels = {}
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                # Placeholder power level calculation
                power_levels[ent.text] = self._calculate_power_level(ent)
        return power_levels
    
    def _calculate_tension(
        self,
        emotional_beats: List[Dict],
        power_dynamics: Dict[str, float]
    ) -> np.ndarray:
        """
        [Placeholder] Calculates scene tension curve based on emotional beats
        and power dynamics. Needs refinement for more accurate tension modeling.
        """
        tensions = []
        for beat in emotional_beats:
            # Basic tension calculation (needs improvement)
            tension = beat["intensity"] * 0.6
            if power_dynamics:
                power_variance = np.std(list(power_dynamics.values()))
                tension += power_variance * 0.4
            tensions.append(tension)
        return np.array(tensions)
    
    def _detect_turns(self, tension_curve: np.ndarray) -> List[int]:
        """
        [Placeholder] Detects major scene turns based on tension curve gradient.
        Algorithm needs to be refined for better turn detection accuracy.
        """
        # Calculate gradient of tension curve
        gradient = np.gradient(tension_curve)
        
        # Identify significant changes in tension
        turns = []
        threshold = np.std(gradient) * 2 if len(gradient) > 0 else 0 # Avoid std on empty gradient
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > threshold:
                if abs(gradient[i]) > abs(gradient[i-1]) and \
                   abs(gradient[i]) > abs(gradient[i+1]): # Ensure it's a local peak
                    turns.append(i)
        return turns
    
    def _extract_emotions(self, sent) -> Dict[str, float]:
        """[Placeholder] Stub for emotion extraction from sentence."""
        return {"neutral": 1.0} # Default stub implementation
    
    def _calculate_intensity(self, sent) -> float:
        """[Placeholder] Stub for emotional intensity calculation."""
        return 0.5 # Default stub intensity
    
    def _calculate_power_level(self, entity) -> float:
        """[Placeholder] Stub for character power level calculation."""
        return 0.5 # Default stub power level

processor = SceneProcessor() # Instantiate scene processor

# --- API Routes ---
@app.post("/scenes/", response_model=Scene)
async def create_scene(scene: Scene):
    """API endpoint to create a new scene."""
    # In future: Save scene to database here
    return scene

@app.get("/scenes/{scene_id}", response_model=Scene)
async def get_scene(scene_id: str):
    """API endpoint to retrieve a scene by its ID."""
    # In future: Retrieve scene from database
    pass

@app.post("/scenes/{scene_id}/analyze", response_model=SceneAnalysis)
async def analyze_scene(scene_id: str, scene: Scene):
    """API endpoint to analyze scene content and return SceneAnalysis."""
    analysis = await processor.process_scene(scene)
    return analysis

@app.post("/render/")
async def render_scene(scene_id: str, settings: RenderSettings):
    """[Placeholder] API endpoint to render a scene to video."""
    # Rendering pipeline implementation to be added
    pass

# --- WebSocket Endpoints ---
@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time communication."""
    await manager.connect(websocket) # Accept and add connection
    try:
        while True:
            data = await websocket.receive_text()
            # Broadcast received message to all clients
            await manager.broadcast({
                "client_id": client_id,
                "message": data
            })
    except WebSocketDisconnect:
        manager.disconnect(websocket) # Remove connection on disconnect

# --- Exception Handling ---
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom exception handler for HTTPExceptions."""
    return {
        "error": True,
        "message": str(exc.detail), # Include exception details in response
        "status_code": exc.status_code
    }

# --- Startup and Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    """
    Startup event handler: initializes services when the application starts.
    Currently a placeholder for future initialization tasks.
    """
    # In future: Initialize ML models, database connections, etc.
    pass

@app.on_event("shutdown")
async def shutdown_event():
    """
    Shutdown event handler: cleans up resources when the application stops.
    Currently a placeholder for future cleanup tasks.
    """
    # In future: Cleanup resources, close connections, etc.
    pass

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # Run the FastAPI app using Uvicorn
