"""
Technical Utilities and Shared Functionality
Provides core tools and data structures used across the Animatix system
"""
from typing import Dict, List, Optional, Union, TypeVar, Generic
from dataclasses import dataclass
from enum import Enum
import math
import numpy as np
from pathlib import Path

# Type definitions
Number = Union[int, float]
T = TypeVar('T')

class SceneLocation(Enum):
    INTERIOR = "interior"
    EXTERIOR = "exterior"
    
class TimeOfDay(Enum):
    DAY = "day"
    NIGHT = "night"
    DAWN = "dawn"
    DUSK = "dusk"
    
class LensType(Enum):
    WIDE = "16mm"
    NORMAL = "35mm"
    PORTRAIT = "85mm"
    TELEPHOTO = "135mm"

class FramingType(Enum):
    EXTREME_WIDE = "extreme_wide"
    WIDE = "wide"
    MEDIUM_WIDE = "medium_wide"
    MEDIUM = "medium"
    MEDIUM_CLOSE = "medium_close"
    CLOSE_UP = "close_up"
    EXTREME_CLOSE = "extreme_close"

@dataclass
class Point3D:
    x: float
    y: float
    z: float
    
    def distance_to(self, other: 'Point3D') -> float:
        return math.sqrt(
            (self.x - other.x) ** 2 +
            (self.y - other.y) ** 2 +
            (self.z - other.z) ** 2
        )

@dataclass
class CameraSettings:
    lens: LensType
    framing: FramingType
    height: float  # in meters
    tilt: float  # in degrees
    roll: float  # in degrees (dutch angle)
    focus_distance: float  # in meters
    aperture: float  # f-stop

@dataclass
class CharacterPlacement:
    position: Point3D
    rotation: float  # in degrees
    scale: float  # relative to frame height
    eye_level: float  # in meters
    power_level: float  # 0-1 scale

class ShotConfiguration:
    """Manages technical aspects of shot composition"""
    
    def __init__(self):
        self.default_shots = {
            "single": {
                "lens": LensType.PORTRAIT,
                "framing": FramingType.MEDIUM_CLOSE,
                "blocking": [Point3D(0, 0, 2)]
            },
            "two_shot": {
                "lens": LensType.NORMAL,
                "framing": FramingType.MEDIUM,
                "blocking": [
                    Point3D(-1, 0, 2),
                    Point3D(1, 0, 2)
                ]
            },
            "over_shoulder": {
                "lens": LensType.PORTRAIT,
                "framing": FramingType.MEDIUM_CLOSE,
                "blocking": [
                    Point3D(-0.3, 0, 1),  # Foreground shoulder
                    Point3D(0, 0, 2)  # Subject
                ]
            }
        }
    
    def calculate_camera_settings(
        self,
        emotional_intensity: float,
        power_dynamics: Dict[str, float],
        character_positions: List[Point3D]
    ) -> CameraSettings:
        """Calculate optimal camera settings based on scene context"""
        
        # Base settings
        settings = CameraSettings(
            lens=LensType.NORMAL,
            framing=FramingType.MEDIUM,
            height=1.6,  # Average eye level
            tilt=0.0,
            roll=0.0,
            focus_distance=2.0,
            aperture=2.8
        )
        
        # Adjust for emotional intensity
        if emotional_intensity > 0.7:
            settings.lens = LensType.PORTRAIT
            settings.framing = FramingType.CLOSE_UP
            settings.aperture = 1.8  # Shallow DOF for emotion
        
        # Adjust for power dynamics
        power_delta = max(power_dynamics.values()) - min(power_dynamics.values())
        if power_delta > 0.6:
            settings.height += power_delta * 0.5  # Higher angle for power
            settings.tilt = -15 * power_delta  # Look down on less powerful
        
        # Calculate focus distance based on character positions
        if character_positions:
            avg_distance = sum(p.z for p in character_positions) / len(character_positions)
            settings.focus_distance = avg_distance
        
        return settings

class SceneAnalysisTools:
    """Utility functions for scene analysis"""
    
    @staticmethod
    def calculate_tension_curve(
        power_shifts: List[float],
        emotional_beats: List[float],
        window_size: int = 5
    ) -> np.ndarray:
        """Calculate tension over time using power shifts and emotional beats"""
        
        # Combine power and emotional data
        combined = np.array([
            ps * 0.7 + eb * 0.3  # Weight power shifts more heavily
            for ps, eb in zip(power_shifts, emotional_beats)
        ])
        
        # Smooth using moving average
        tension = np.convolve(
            combined,
            np.ones(window_size) / window_size,
            mode='valid'
        )
        
        return tension
    
    @staticmethod
    def find_scene_turns(
        tension: np.ndarray,
        threshold: float = 0.7
    ) -> List[int]:
        """Identify major turning points in scene tension"""
        
        # Calculate rate of change
        gradient = np.gradient(tension)
        
        # Find peaks in absolute gradient
        peaks = []
        for i in range(1, len(gradient) - 1):
            if abs(gradient[i]) > threshold:
                if abs(gradient[i]) > abs(gradient[i-1]) and \
                   abs(gradient[i]) > abs(gradient[i+1]):
                    peaks.append(i)
        
        return peaks

class Cache(Generic[T]):
    """Simple caching system for expensive computations"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.cache: Dict[str, T] = {}
    
    def get(self, key: str) -> Optional[T]:
        """Retrieve item from cache"""
        return self.cache.get(key)
    
    def set(self, key: str, value: T) -> None:
        """Store item in cache"""
        if len(self.cache) >= self.max_size:
            # Remove oldest item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        self.cache[key] = value

# Utility functions
def normalize_vector(v: np.ndarray) -> np.ndarray:
    """Normalize a vector to unit length"""
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

def interpolate_power(start: float, end: float, steps: int) -> List[float]:
    """Generate smooth power level transitions"""
    return [
        start + (end - start) * (i / (steps - 1))
        for i in range(steps)
    ]

def calculate_eye_lines(
    char_positions: List[Point3D],
    power_levels: List[float]
) -> List[float]:
    """Calculate character eye lines based on position and power"""
    eye_lines = []
    for i, pos in enumerate(char_positions):
        for j, other_pos in enumerate(char_positions):
            if i != j:
                # Calculate base angle
                dx = other_pos.x - pos.x
                dy = other_pos.y - pos.y
                base_angle = math.atan2(dy, dx)
                
                # Adjust for power differential
                power_diff = power_levels[j] - power_levels[i]
                angle_adjust = power_diff * 15  # Up to 15 degree adjustment
                
                eye_lines.append(math.degrees(base_angle) + angle_adjust)
    
    return eye_lines

# Example usage
if __name__ == "__main__":
    # Test shot configuration
    shot_config = ShotConfiguration()
    camera_settings = shot_config.calculate_camera_settings(
        emotional_intensity=0.8,
        power_dynamics={"char1": 0.3, "char2": 0.9},
        character_positions=[Point3D(-1, 0, 2), Point3D(1, 0, 2)]
    )
    
    # Test scene analysis
    analysis_tools = SceneAnalysisTools()
    power_shifts = [0.1, 0.2, 0.4, 0.7, 0.9, 0.8, 0.6]
    emotional_beats = [0.2, 0.3, 0.5, 0.6, 0.8, 0.7, 0.5]
    
    tension = analysis_tools.calculate_tension_curve(
        power_shifts,
        emotional_beats
    )
    
    turns = analysis_tools.find_scene_turns(tension)
