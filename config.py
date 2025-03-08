"""
Configuration management for Animatix
Centralizes all configurable parameters and thresholds
"""
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Set

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    NERVOUS = "nervous"
    CONFIDENT = "confident"
    SAD = "sad"
    HAPPY = "happy"

class GestureType(Enum):
    EMPHASIS = "emphasis"
    POWER = "power"
    EMOTIONAL = "emotional"
    NEUTRAL = "neutral"

@dataclass
class PowerDynamicsConfig:
    """Power dynamics calculation parameters"""
    dominance_threshold: float = 0.7
    submission_threshold: float = 0.3
    base_power_shift: float = 0.1
    max_cumulative_shift: float = 0.8
    position_weight_factor: float = 1.0

@dataclass
class GestureConfig:
    """Gesture generation parameters"""
    base_probability: float = 0.2
    power_high_modifier: float = 0.3
    power_low_modifier: float = 0.1
    min_word_length: int = 3
    max_probability: float = 0.8
    
    # Cached word sets for performance
    emphasis_words: Set[str] = frozenset({
        'must', 'never', 'always', 'absolutely', 'definitely'
    })
    power_words: Set[str] = frozenset({
        'command', 'demand', 'insist', 'allow', 'permit', 'forbid'
    })
    
    # Emotion probability modifiers
    emotion_modifiers: Dict[EmotionalState, float] = {
        EmotionalState.ANGRY: 0.3,
        EmotionalState.NERVOUS: 0.4,
        EmotionalState.CONFIDENT: 0.25,
        EmotionalState.SAD: 0.15,
        EmotionalState.HAPPY: 0.35
    }
    
    # Gesture parameters per emotional state
    emotion_effects: Dict[EmotionalState, Dict[str, float]] = {
        EmotionalState.ANGRY: {'amplitude': 1.4, 'speed': 1.3},
        EmotionalState.NERVOUS: {'amplitude': 0.6, 'speed': 1.4},
        EmotionalState.CONFIDENT: {'amplitude': 1.2, 'speed': 0.9},
        EmotionalState.SAD: {'amplitude': 0.5, 'speed': 0.7},
        EmotionalState.HAPPY: {'amplitude': 1.1, 'speed': 1.2}
    }

@dataclass
class AnimatixConfig:
    """Global configuration for Animatix"""
    power_dynamics: PowerDynamicsConfig = PowerDynamicsConfig()
    gesture: GestureConfig = GestureConfig()
    
    # Singleton instance
    _instance = None
    
    @classmethod
    def get_instance(cls) -> 'AnimatixConfig':
        """Get or create singleton config instance"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def reload_from_env(self):
        """Reload configuration from environment variables"""
        import os
        
        # Example: ANIMATIX_POWER_DOMINANCE_THRESHOLD=0.8
        if threshold := os.getenv('ANIMATIX_POWER_DOMINANCE_THRESHOLD'):
            self.power_dynamics.dominance_threshold = float(threshold)
            
        # Add more environment variable handling as needed
