"""
Enhanced Storyboard System
Handles visual composition, shot planning, and scene visualization
"""
from typing import Dict, List, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from enum import Enum
import math

class CompositionRule(Enum):
    RULE_OF_THIRDS = "rule_of_thirds"
    GOLDEN_RATIO = "golden_ratio"
    SYMMETRICAL = "symmetrical"
    DIAGONAL = "diagonal"
    FRAME_WITHIN_FRAME = "frame_within_frame"
    LEADING_LINES = "leading_lines"

class CameraAngle(Enum):
    EYE_LEVEL = "eye_level"
    LOW_ANGLE = "low_angle"
    HIGH_ANGLE = "high_angle"
    DUTCH = "dutch"
    BIRDS_EYE = "birds_eye"
    WORMS_EYE = "worms_eye"

@dataclass
class Frame:
    width: int = 1920
    height: int = 1080
    aspect_ratio: float = 16/9
    safe_margins: Dict[str, int] = None
    
    def __post_init__(self):
        if self.safe_margins is None:
            self.safe_margins = {
                "top": int(self.height * 0.1),
                "bottom": int(self.height * 0.1),
                "left": int(self.width * 0.1),
                "right": int(self.width * 0.1)
            }

@dataclass
class CharacterPlacement:
    character_id: str
    position: Tuple[float, float, float]  # x, y, z coordinates (normalized 0-1)
    facing_direction: float  # angle in degrees
    eye_line: float  # vertical angle in degrees
    scale: float  # size in frame (0-1)

@dataclass
class Shot:
    composition_rule: CompositionRule
    camera_angle: CameraAngle
    focal_length: float  # in mm
    characters: List[CharacterPlacement]
    depth_of_field: float  # in meters
    camera_movement: Optional[str]
    duration: float  # in seconds

class StoryboardEngine:
    def __init__(self, frame: Optional[Frame] = None):
        self.frame = frame or Frame()
        self.golden_ratio = (1 + math.sqrt(5)) / 2
        
    def compose_shot(
        self,
        characters: List[Dict],
        power_dynamics: Dict[str, float],
        emotional_intensity: float,
        scene_tension: float
    ) -> Shot:
        """Generate optimal shot composition based on scene dynamics"""
        
        # Determine composition rule based on emotional context
        composition_rule = self._select_composition_rule(
            len(characters),
            emotional_intensity,
            scene_tension
        )
        
        # Calculate camera angle based on power dynamics
        camera_angle = self._calculate_camera_angle(power_dynamics)
        
        # Position characters in frame
        character_placements = self._position_characters(
            characters,
            power_dynamics,
            composition_rule
        )
        
        # Select focal length based on emotional proximity
        focal_length = self._determine_focal_length(
            emotional_intensity,
            len(characters)
        )
        
        # Calculate depth of field
        depth_of_field = self._calculate_depth_of_field(
            emotional_intensity,
            scene_tension
        )
        
        # Determine camera movement
        camera_movement = self._plan_camera_movement(
            emotional_intensity,
            scene_tension
        )
        
        return Shot(
            composition_rule=composition_rule,
            camera_angle=camera_angle,
            focal_length=focal_length,
            characters=character_placements,
            depth_of_field=depth_of_field,
            camera_movement=camera_movement,
            duration=self._calculate_shot_duration(emotional_intensity)
        )
    
    def _select_composition_rule(
        self,
        character_count: int,
        emotional_intensity: float,
        scene_tension: float
    ) -> CompositionRule:
        """Select appropriate composition rule based on scene context"""
        if scene_tension > 0.8:
            return CompositionRule.DIAGONAL  # Use diagonals to imply tension
        elif emotional_intensity > 0.7:
            return CompositionRule.FRAME_WITHIN_FRAME  # Claustrophobic framing
        elif character_count > 2:
            return CompositionRule.RULE_OF_THIRDS  # Balance multiple characters
        else:
            return CompositionRule.GOLDEN_RATIO  # Classic composition
    
    def _calculate_camera_angle(
        self,
        power_dynamics: Dict[str, float]
    ) -> CameraAngle:
        """Determine camera angle based on power dynamics"""
        max_power = max(power_dynamics.values())
        min_power = min(power_dynamics.values())
        power_delta = max_power - min_power
        
        if power_delta > 0.7:
            return CameraAngle.LOW_ANGLE if max_power > 0.8 else CameraAngle.HIGH_ANGLE
        elif power_delta > 0.4:
            return CameraAngle.DUTCH
        else:
            return CameraAngle.EYE_LEVEL
    
    def _position_characters(
        self,
        characters: List[Dict],
        power_dynamics: Dict[str, float],
        composition_rule: CompositionRule
    ) -> List[CharacterPlacement]:
        """Position characters in frame according to composition rules"""
        placements = []
        
        if composition_rule == CompositionRule.RULE_OF_THIRDS:
            # Place on third intersections
            third_points = [
                (1/3, 1/3), (2/3, 1/3),
                (1/3, 2/3), (2/3, 2/3)
            ]
            for idx, char in enumerate(characters):
                power = power_dynamics[char['id']]
                point = third_points[idx % len(third_points)]
                placements.append(CharacterPlacement(
                    character_id=char['id'],
                    position=(point[0], point[1], power),
                    facing_direction=self._calculate_facing_direction(idx, len(characters)),
                    eye_line=power * 30 - 15,  # -15 to +15 degrees
                    scale=0.3 + power * 0.4  # 0.3 to 0.7 scale
                ))
        
        elif composition_rule == CompositionRule.GOLDEN_RATIO:
            # Place using golden ratio points
            golden_points = [
                (1/self.golden_ratio, 1/self.golden_ratio),
                (1 - 1/self.golden_ratio, 1 - 1/self.golden_ratio)
            ]
            for idx, char in enumerate(characters):
                power = power_dynamics[char['id']]
                point = golden_points[idx % len(golden_points)]
                placements.append(CharacterPlacement(
                    character_id=char['id'],
                    position=(point[0], point[1], power),
                    facing_direction=self._calculate_facing_direction(idx, len(characters)),
                    eye_line=power * 20 - 10,  # -10 to +10 degrees
                    scale=0.4 + power * 0.3  # 0.4 to 0.7 scale
                ))
        
        return placements
    
    def _calculate_facing_direction(self, index: int, total: int) -> float:
        """Calculate character facing direction based on position"""
        if total == 1:
            return 0  # Face camera
        elif total == 2:
            return 90 if index == 0 else -90  # Face each other
        else:
            return (360 / total) * index  # Evenly distributed
    
    def _determine_focal_length(
        self,
        emotional_intensity: float,
        character_count: int
    ) -> float:
        """Select focal length based on scene context"""
        if character_count == 1:
            # Single character: 85mm (portrait) to 135mm (intense close-up)
            return 85 + emotional_intensity * 50
        elif character_count == 2:
            # Two characters: 50mm (natural) to 85mm (intimate)
            return 50 + emotional_intensity * 35
        else:
            # Groups: 35mm (wide) to 50mm (natural)
            return 35 + emotional_intensity * 15
    
    def _calculate_depth_of_field(
        self,
        emotional_intensity: float,
        scene_tension: float
    ) -> float:
        """Calculate depth of field in meters"""
        # More shallow DOF for emotional/tense scenes
        base_depth = 3.0  # meters
        return base_depth * (1 - (emotional_intensity * 0.5 + scene_tension * 0.5))
    
    def _plan_camera_movement(
        self,
        emotional_intensity: float,
        scene_tension: float
    ) -> Optional[str]:
        """Determine camera movement based on scene dynamics"""
        if scene_tension > 0.8:
            return "handheld"
        elif emotional_intensity > 0.7:
            return "dolly_in"
        elif emotional_intensity < 0.3:
            return "dolly_out"
        else:
            return None  # Static shot
    
    def _calculate_shot_duration(self, emotional_intensity: float) -> float:
        """Calculate appropriate shot duration in seconds"""
        # Faster cuts for higher emotional intensity
        base_duration = 4.0  # seconds
        return base_duration * (1 - emotional_intensity * 0.5)  # 2-4 seconds

# Example usage
if __name__ == "__main__":
    engine = StoryboardEngine()
    
    # Example scene data
    characters = [
        {"id": "john", "name": "John"},
        {"id": "sarah", "name": "Sarah"}
    ]
    
    power_dynamics = {
        "john": 0.3,  # Submissive
        "sarah": 0.8  # Dominant
    }
    
    shot = engine.compose_shot(
        characters=characters,
        power_dynamics=power_dynamics,
        emotional_intensity=0.7,
        scene_tension=0.6
    )
