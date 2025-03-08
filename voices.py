"""
Enhanced Voice and Dialogue System
Handles character voice generation, dialogue timing, and performance modulation
"""
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import re

class EmotionalState(Enum):
    NEUTRAL = "neutral"
    ANGRY = "angry"
    SAD = "sad"
    HAPPY = "happy"
    AFRAID = "afraid"
    CONFIDENT = "confident"
    NERVOUS = "nervous"
    CONTEMPT = "contempt"

class VocalCharacteristic(Enum):
    PITCH = "pitch"
    SPEED = "speed"
    VOLUME = "volume"
    CLARITY = "clarity"
    BREATHINESS = "breathiness"
    ROUGHNESS = "roughness"

@dataclass
class Phoneme:
    sound: str
    duration: float
    emphasis: float

@dataclass
class Viseme:
    shape: str
    duration: float
    intensity: float

@dataclass
class DialogueBeat:
    text: str
    emotion: EmotionalState
    power_level: float
    timing: Dict[str, float]
    phonemes: List[Phoneme]
    visemes: List[Viseme]
    pauses: List[float]

class VoiceEngine:
    def __init__(self):
        # Base vocal parameters
        self.base_speed = 160  # words per minute
        self.base_pitch = 1.0  # normalized pitch
        self.base_volume = 0.7  # normalized volume
        
        # Emotional modifiers
        self.emotion_modifiers = {
            EmotionalState.ANGRY: {
                VocalCharacteristic.SPEED: 1.2,
                VocalCharacteristic.PITCH: 1.3,
                VocalCharacteristic.VOLUME: 1.4,
                VocalCharacteristic.ROUGHNESS: 0.8
            },
            EmotionalState.SAD: {
                VocalCharacteristic.SPEED: 0.7,
                VocalCharacteristic.PITCH: 0.8,
                VocalCharacteristic.VOLUME: 0.6,
                VocalCharacteristic.BREATHINESS: 0.4
            },
            EmotionalState.HAPPY: {
                VocalCharacteristic.SPEED: 1.1,
                VocalCharacteristic.PITCH: 1.2,
                VocalCharacteristic.VOLUME: 1.1,
                VocalCharacteristic.CLARITY: 0.8
            },
            EmotionalState.AFRAID: {
                VocalCharacteristic.SPEED: 1.3,
                VocalCharacteristic.PITCH: 1.4,
                VocalCharacteristic.VOLUME: 0.5,
                VocalCharacteristic.BREATHINESS: 0.7
            },
            EmotionalState.CONFIDENT: {
                VocalCharacteristic.SPEED: 0.9,
                VocalCharacteristic.PITCH: 1.1,
                VocalCharacteristic.VOLUME: 1.2,
                VocalCharacteristic.CLARITY: 0.9
            },
            EmotionalState.NERVOUS: {
                VocalCharacteristic.SPEED: 1.4,
                VocalCharacteristic.PITCH: 1.2,
                VocalCharacteristic.VOLUME: 0.6,
                VocalCharacteristic.CLARITY: 0.4
            },
            EmotionalState.CONTEMPT: {
                VocalCharacteristic.SPEED: 0.8,
                VocalCharacteristic.PITCH: 0.9,
                VocalCharacteristic.VOLUME: 0.8,
                VocalCharacteristic.ROUGHNESS: 0.5
            }
        }
    
    def process_dialogue(
        self,
        text: str,
        emotion: EmotionalState,
        power_level: float,
        scene_tension: float
    ) -> DialogueBeat:
        """Process dialogue line into detailed performance instructions"""
        
        # Calculate timing
        timing = self._calculate_timing(text, emotion, power_level)
        
        # Generate phonemes
        phonemes = self._generate_phonemes(text, emotion)
        
        # Generate visemes
        visemes = self._generate_visemes(phonemes, power_level)
        
        # Detect natural pauses
        pauses = self._detect_pauses(text, scene_tension)
        
        return DialogueBeat(
            text=text,
            emotion=emotion,
            power_level=power_level,
            timing=timing,
            phonemes=phonemes,
            visemes=visemes,
            pauses=pauses
        )
    
    def _calculate_timing(
        self,
        text: str,
        emotion: EmotionalState,
        power_level: float
    ) -> Dict[str, float]:
        """Calculate detailed timing for dialogue delivery"""
        
        word_count = len(text.split())
        
        # Base duration
        base_duration = (word_count / self.base_speed) * 60
        
        # Apply emotional modifier
        emotion_speed = self.emotion_modifiers[emotion][VocalCharacteristic.SPEED]
        duration = base_duration * emotion_speed
        
        # Adjust for power level
        # Higher power = more measured speech
        duration *= 1 + (power_level * 0.2)
        
        # Calculate variations
        timing = {
            "total_duration": duration,
            "pre_pause": duration * 0.1,
            "post_pause": duration * 0.15,
            "word_spacing": duration / (word_count + 1)
        }
        
        return timing
    
    def _generate_phonemes(
        self,
        text: str,
        emotion: EmotionalState
    ) -> List[Phoneme]:
        """Convert text to phoneme sequence with timing"""
        
        # This would integrate with a proper text-to-phoneme system
        # For now, using simplified example
        phonemes = []
        words = text.split()
        
        for word in words:
            # Simplified phoneme generation
            for char in word:
                emphasis = 1.0
                if char in 'aeiou':
                    # Emphasize vowels based on emotion
                    emphasis *= self.emotion_modifiers[emotion][VocalCharacteristic.VOLUME]
                
                phonemes.append(Phoneme(
                    sound=char,
                    duration=0.1,  # Base duration
                    emphasis=emphasis
                ))
        
        return phonemes
    
    def _generate_visemes(
        self,
        phonemes: List[Phoneme],
        power_level: float
    ) -> List[Viseme]:
        """Convert phonemes to viseme sequence"""
        
        visemes = []
        
        # Basic phoneme to viseme mapping
        phoneme_map = {
            'a': 'ah',
            'e': 'eh',
            'i': 'ee',
            'o': 'oh',
            'u': 'oo'
        }
        
        for phoneme in phonemes:
            viseme_shape = phoneme_map.get(phoneme.sound.lower(), 'rest')
            
            # Adjust intensity based on power level
            # More powerful characters have more pronounced mouth movements
            intensity = phoneme.emphasis * (0.7 + power_level * 0.3)
            
            visemes.append(Viseme(
                shape=viseme_shape,
                duration=phoneme.duration,
                intensity=intensity
            ))
        
        return visemes
    
    def _detect_pauses(
        self,
        text: str,
        scene_tension: float
    ) -> List[float]:
        """Detect natural pause points in dialogue"""
        
        pauses = []
        
        # Punctuation-based pauses
        for match in re.finditer(r'[,.?!]', text):
            position = match.start() / len(text)
            
            # Pause duration based on punctuation and tension
            if text[match.start()] in '.?!':
                duration = 0.7 * (1 + scene_tension * 0.3)
            else:
                duration = 0.3 * (1 + scene_tension * 0.2)
            
            pauses.append(duration)
        
        return pauses

# Example usage
if __name__ == "__main__":
    engine = VoiceEngine()
    
    # Example dialogue processing
    dialogue = "I didn't think you'd come here today."
    beat = engine.process_dialogue(
        text=dialogue,
        emotion=EmotionalState.NERVOUS,
        power_level=0.3,
        scene_tension=0.7
    )
