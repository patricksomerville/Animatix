"""
Scene mechanics and turn detection for Animatix
Handles scene analysis and dramatic structure
"""
import numpy as np
from typing import Dict, List, Optional
import logging
from dataclasses import dataclass
from pipeline import AnimaticPipeline
from config import AnimatixConfig, EmotionalState

logger = logging.getLogger(__name__)

@dataclass
class SceneBeat:
    """Represents a dramatic beat in the scene"""
    line_number: int
    power_shift: float
    subtext_score: float
    camera_pressure: float
    emotional_state: EmotionalState

class SceneTurnAnalyzer:
    """Analyzes scene structure and detects dramatic turns"""
    
    def __init__(self, script: List[dict], pipeline: Optional[AnimaticPipeline] = None):
        """Initialize analyzer with script and optional pipeline instance"""
        self.script = script
        self.pipeline = pipeline or AnimaticPipeline()
        self.config = AnimatixConfig.get_instance()
        self.beats: List[SceneBeat] = []
        self._parse_beats()
        
    def _parse_beats(self) -> None:
        """Parse script into dramatic beats with error handling"""
        try:
            self.beats = []
            for line in self.script:
                try:
                    beat = SceneBeat(
                        line_number=line.get('number', 0),
                        power_shift=self._calc_power_delta(line),
                        subtext_score=getattr(line, 'subtext_complexity', 0.0),
                        camera_pressure=0.0,
                        emotional_state=self._detect_emotion(line)
                    )
                    self.beats.append(beat)
                except Exception as e:
                    logger.warning(f"Failed to parse beat: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Failed to parse script: {e}")
            raise RuntimeError(f"Script parsing failed: {e}")

    def detect_turn(self) -> Optional[SceneBeat]:
        """Find dramatic turn in scene based on power dynamics"""
        if not self.beats:
            logger.warning("No beats to analyze")
            return None
            
        try:
            # Calculate cumulative power shifts
            shifts = [b.power_shift for b in self.beats]
            cum_shifts = np.cumsum(shifts)
            
            # Find turn point using configured threshold
            threshold = self.config.power_dynamics.dominance_threshold
            max_shift = max(cum_shifts)
            
            if max_shift == 0:
                logger.info("No significant power shifts detected")
                return None
                
            turn_point = np.argmax(cum_shifts >= threshold * max_shift)
            return self.beats[turn_point]
            
        except Exception as e:
            logger.error(f"Turn detection failed: {e}")
            return None

    def _calc_power_delta(self, line: dict) -> float:
        """Calculate power shift using pipeline's analysis"""
        try:
            doc = self.pipeline.nlp(line.get('text', ''))
            
            # Get power shifts per character
            shifts = {}
            for sent in doc.sents:
                for char in self.pipeline.characters.values():
                    shift = self.pipeline._calculate_power_shift(sent, char.name)
                    shifts[char.name] = shifts.get(char.name, 0.0) + shift
            
            # Return net power change
            return sum(shifts.values())
            
        except Exception as e:
            logger.warning(f"Power calculation failed: {e}")
            return 0.0
            
    def _detect_emotion(self, line: dict) -> EmotionalState:
        """Detect dominant emotion in line"""
        try:
            doc = self.pipeline.nlp(line.get('text', ''))
            emotion = self.pipeline._calculate_emotion(doc)
            return EmotionalState(emotion)
        except ValueError:
            logger.warning(f"Unknown emotion detected, defaulting to NEUTRAL")
            return EmotionalState.NEUTRAL
        except Exception as e:
            logger.warning(f"Emotion detection failed: {e}")
            return EmotionalState.NEUTRAL