"""
Enhanced Performance System
Handles character animation, movement, and staging based on scene dynamics
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path

class GestureType(Enum):
    EMPHATIC = "emphatic"
    DESCRIPTIVE = "descriptive"
    BEAT = "beat"
    DEICTIC = "deictic"
    ADAPTORS = "adaptors"
    EMBLEMS = "emblems"

class PostureState(Enum):
    DOMINANT = "dominant"
    SUBMISSIVE = "submissive"
    OPEN = "open"
    CLOSED = "closed"
    FORWARD = "forward"
    BACKWARD = "backward"

@dataclass
class BodyPose:
    spine_angle: float  # degrees from vertical
    shoulder_width: float  # normalized 0-1
    weight_distribution: float  # -1 (left) to 1 (right)
    head_tilt: float  # degrees
    arm_openness: float  # 0 (closed) to 1 (open)

@dataclass
class Gesture:
    type: GestureType
    intensity: float
    duration: float
    hand_position: Tuple[float, float, float]  # x, y, z coordinates
    trajectory: List[Tuple[float, float, float]]  # movement path

@dataclass
class PerformanceFrame:
    timestamp: float
    pose: BodyPose
    gesture: Optional[Gesture]
    facial_expression: Dict[str, float]  # muscle group -> intensity
    gaze_target: Optional[Tuple[float, float, float]]

class PerformanceEngine:
    def __init__(self):
        # Base performance parameters
        self.frame_rate = 24  # fps
        self.gesture_library = self._load_gesture_library()
        self.expression_library = self._load_expression_library()
        
    def generate_performance(
        self,
        dialogue_beat: Dict,
        emotional_state: str,
        power_level: float,
        scene_tension: float,
        character_relationships: Dict[str, float]
    ) -> List[PerformanceFrame]:
        """Generate detailed performance frames for a dialogue beat"""
        
        frames = []
        duration = dialogue_beat['timing']['total_duration']
        frame_count = int(duration * self.frame_rate)
        
        # Generate base pose
        base_pose = self._calculate_base_pose(power_level, emotional_state)
        
        # Generate performance frames
        for i in range(frame_count):
            timestamp = i / self.frame_rate
            
            # Calculate pose variations
            pose = self._vary_pose(
                base_pose,
                timestamp,
                scene_tension
            )
            
            # Generate gesture if needed
            gesture = self._generate_gesture(
                timestamp,
                dialogue_beat,
                emotional_state,
                power_level
            )
            
            # Calculate facial expression
            expression = self._calculate_expression(
                timestamp,
                emotional_state,
                power_level
            )
            
            # Determine gaze direction
            gaze = self._calculate_gaze(
                timestamp,
                character_relationships,
                power_level
            )
            
            frames.append(PerformanceFrame(
                timestamp=timestamp,
                pose=pose,
                gesture=gesture,
                facial_expression=expression,
                gaze_target=gaze
            ))
        
        return frames
    
    def _calculate_base_pose(
        self,
        power_level: float,
        emotional_state: str
    ) -> BodyPose:
        """Calculate base body pose based on character state"""
        
        # Spine angle varies with power level
        spine_angle = -5 + power_level * 10  # -5 to +5 degrees
        
        # Shoulder width increases with power
        shoulder_width = 0.6 + power_level * 0.4
        
        # Weight distribution based on emotional state
        weight_distribution = 0.0  # centered by default
        if emotional_state == "nervous":
            weight_distribution = np.sin(power_level * np.pi) * 0.3
        
        # Head tilt varies inversely with power
        head_tilt = (1 - power_level) * 5  # 0 to 5 degrees
        
        # Arm openness based on power and emotion
        arm_openness = power_level
        if emotional_state in ["afraid", "nervous"]:
            arm_openness *= 0.5
        
        return BodyPose(
            spine_angle=spine_angle,
            shoulder_width=shoulder_width,
            weight_distribution=weight_distribution,
            head_tilt=head_tilt,
            arm_openness=arm_openness
        )
    
    def _vary_pose(
        self,
        base_pose: BodyPose,
        timestamp: float,
        scene_tension: float
    ) -> BodyPose:
        """Add natural variations to base pose over time"""
        
        # Add subtle movement based on breathing
        breath_cycle = np.sin(timestamp * 2 * np.pi * 0.2)  # 5-second cycle
        
        # More movement when tense
        variation_amount = 0.2 + scene_tension * 0.3
        
        return BodyPose(
            spine_angle=base_pose.spine_angle + breath_cycle * variation_amount,
            shoulder_width=base_pose.shoulder_width,
            weight_distribution=base_pose.weight_distribution + breath_cycle * variation_amount * 0.1,
            head_tilt=base_pose.head_tilt + breath_cycle * variation_amount * 0.5,
            arm_openness=base_pose.arm_openness
        )
    
    def _generate_gesture(
        self,
        timestamp: float,
        dialogue_beat: Dict,
        emotional_state: str,
        power_level: float
    ) -> Optional[Gesture]:
        """Generate appropriate gesture for the moment"""
        
        # Check if we're at a gesture trigger point
        words = dialogue_beat.get('text', '').split()
        word_timing = dialogue_beat['timing']['word_spacing']
        current_word_idx = int(timestamp / word_timing)
        
        if current_word_idx >= len(words):
            return None
        
        # Determine if current word should trigger gesture
        if self._should_gesture(words[current_word_idx], emotional_state, power_level):
            gesture_type = self._select_gesture_type(
                words[current_word_idx],
                emotional_state
            )
            
            gesture_params = self.generate_gesture_params(emotional_state, power_level)
            
            return Gesture(
                type=gesture_type,
                intensity=gesture_params['amplitude'],
                duration=gesture_params['speed'],
                hand_position=(0.0, 0.0, 0.0),
                trajectory=self._generate_gesture_trajectory(gesture_type)
            )
        
        return None
    
    def _calculate_expression(
        self,
        timestamp: float,
        emotional_state: str,
        power_level: float
    ) -> Dict[str, float]:
        """Calculate facial expression muscle activations"""
        
        # Base expression from emotional state
        expression = self.expression_library.get(emotional_state, {})
        
        # Modify based on power level
        for muscle, intensity in expression.items():
            # More powerful characters have more controlled expressions
            expression[muscle] = intensity * (0.7 + power_level * 0.3)
        
        return expression
    
    def _calculate_gaze(
        self,
        timestamp: float,
        character_relationships: Dict[str, float],
        power_level: float
    ) -> Optional[Tuple[float, float, float]]:
        """Calculate gaze direction based on relationships"""
        
        if not character_relationships:
            return None
        
        # Find most significant relationship
        target_char = max(character_relationships.items(), key=lambda x: abs(x[1]))
        relationship_value = target_char[1]
        
        # Calculate gaze behavior
        if power_level > 0.7:
            # Powerful characters maintain more direct gaze
            gaze_stability = 0.9
        else:
            # Less powerful characters show more gaze aversion
            gaze_stability = 0.5
        
        # Add natural gaze movement
        if np.random.random() > gaze_stability:
            # Generate slight gaze shift
            return (
                np.random.normal(0, 0.1),
                np.random.normal(1.6, 0.1),  # Average eye level
                np.random.normal(-relationship_value, 0.2)
            )
        
        return (0.0, 1.6, -relationship_value)
    
    def _should_gesture(self, word: str, emotional_state: str, power_level: float) -> bool:
        """Determine if a word should trigger a gesture based on linguistic and emotional context"""
        # Cache common words for performance
        EMPHASIS_WORDS = {'must', 'never', 'always', 'absolutely'}
        POWER_WORDS = {'command', 'demand', 'insist', 'allow'}
        
        word_lower = word.lower()
        base_prob = 0.2
        
        # Quick checks first
        if len(word) <= 3:  # Skip short words
            return False
            
        # Power level affects gesture frequency
        power_modifier = 0.3 if power_level > 0.7 else 0.1
        
        # Word importance check
        if word_lower in EMPHASIS_WORDS:
            base_prob += 0.4
        elif word_lower in POWER_WORDS:
            base_prob += 0.3 * power_level  # Scale with power
            
        # Emotional state affects gesture probability
        emotion_modifiers = {
            'angry': 0.3,
            'nervous': 0.4,
            'confident': 0.25
        }
        base_prob += emotion_modifiers.get(emotional_state, 0.0)
        
        return np.random.random() < min(base_prob + power_modifier, 0.8)

    def generate_gesture_params(self, emotional_state: str, power_level: float) -> Dict[str, float]:
        """Generate gesture parameters optimized for the character's state"""
        # Base parameters
        params = {
            'amplitude': 1.0,
            'speed': 1.0,
            'repetitions': 1
        }
        
        # Power level affects gesture size and speed
        if power_level > 0.7:
            params['amplitude'] *= 1.3
            params['speed'] *= 0.9  # More deliberate
        else:
            params['amplitude'] *= 0.7
            params['speed'] *= 1.1  # More nervous
            
        # Emotional adjustments
        emotion_effects = {
            'angry': {'amplitude': 1.4, 'speed': 1.3},
            'nervous': {'amplitude': 0.6, 'speed': 1.4},
            'confident': {'amplitude': 1.2, 'speed': 0.9}
        }
        
        if emotional_state in emotion_effects:
            effect = emotion_effects[emotional_state]
            params['amplitude'] *= effect['amplitude']
            params['speed'] *= effect['speed']
            
        # Add subtle variation (vectorized operation)
        variation = np.random.normal(1, 0.1, size=len(params))
        params = {k: v * variation[i] for i, (k, v) in enumerate(params.items())}
        
        return params
    
    def _select_gesture_type(
        self,
        word: str,
        emotional_state: str
    ) -> GestureType:
        """Select appropriate gesture type"""
        # Implement gesture selection logic
        return GestureType.BEAT
    
    def _generate_gesture_trajectory(
        self,
        gesture_type: GestureType
    ) -> List[Tuple[float, float, float]]:
        """Generate movement path for gesture"""
        # Implement trajectory generation
        return [(0, 0, 0), (0.1, 0.1, 0), (0, 0, 0)]
    
    def _load_gesture_library(self) -> Dict:
        """Load predefined gesture patterns"""
        # Implement gesture library loading
        return {}
    
    def _load_expression_library(self) -> Dict:
        """Load predefined facial expressions"""
        return {
            "neutral": {
                "brow_raise": 0.0,
                "brow_furrow": 0.0,
                "smile": 0.0,
                "squint": 0.0
            },
            "angry": {
                "brow_raise": 0.0,
                "brow_furrow": 0.8,
                "smile": 0.0,
                "squint": 0.6
            },
            "happy": {
                "brow_raise": 0.3,
                "brow_furrow": 0.0,
                "smile": 0.7,
                "squint": 0.2
            }
        }

# Example usage
if __name__ == "__main__":
    engine = PerformanceEngine()
    
    # Example performance generation
    dialogue_beat = {
        "text": "I didn't think you'd come here today.",
        "timing": {
            "total_duration": 2.0,
            "word_spacing": 0.3
        }
    }
    
    performance = engine.generate_performance(
        dialogue_beat=dialogue_beat,
        emotional_state="nervous",
        power_level=0.3,
        scene_tension=0.7,
        character_relationships={"other_char": -0.5}
    )
