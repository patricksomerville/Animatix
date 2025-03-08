"""
Enhanced Script Analysis System
Integrates with AnimaticPipeline to provide deep scene understanding
"""
from typing import Dict, List, Tuple, Optional, Any
import spacy
import numpy as np
from dataclasses import dataclass
from enum import Enum
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionalValence(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    COMPLEX = "complex"

class SubtextType(Enum):
    DECEPTION = "deception"
    ATTRACTION = "attraction"
    CONFLICT = "conflict"
    FEAR = "fear"
    POWER_PLAY = "power_play"
    VULNERABILITY = "vulnerability"

@dataclass
class SceneContext:
    location: str
    time_of_day: str
    mood: str
    tension_level: float  # 0-1 scale
    subtext_complexity: float  # 0-1 scale
    emotional_valence: EmotionalValence
    dominant_themes: List[str]
    atmospheric_notes: List[str]

@dataclass
class CharacterIntent:
    stated_goal: str
    hidden_goal: str
    emotional_subtext: str
    power_position: float  # 0-1 scale
    vulnerability: float  # 0-1 scale
    psychological_state: Dict[str, float]
    relationship_dynamics: Dict[str, float]
    character_arc: List[Tuple[float, str]]  # list of (time, state) tuples

@dataclass
class SubtextLayer:
    type: SubtextType
    intensity: float
    involved_characters: List[str]
    trigger_phrases: List[str]
    emotional_indicators: Dict[str, float]

@dataclass
class SceneAnalysis:
    context: SceneContext
    character_intents: Dict[str, CharacterIntent]
    subtext_layers: List[SubtextLayer]
    emotional_progression: List[Tuple[float, Dict[str, float]]]
    power_dynamics: List[Dict[str, float]]
    dramatic_beats: List[Dict[str, Any]]

class SceneAnalyzer:
    """Advanced scene analysis engine with deep psychological insight"""
    
    def __init__(self):
        self.nlp = spacy.load("en_core_web_trf")
        self.scene_context: Optional[SceneContext] = None
        self.character_intents: Dict[str, CharacterIntent] = {}
        self.subtext_layers: List[SubtextLayer] = []
        self.emotional_history: List[Tuple[float, Dict[str, float]]] = []
        
        # Psychological markers
        self.emotional_indicators = {
            "anxiety": ["nervous", "worried", "tense", "uneasy"],
            "confidence": ["assured", "certain", "strong", "decisive"],
            "vulnerability": ["hesitant", "unsure", "weak", "afraid"],
            "dominance": ["commands", "insists", "demands", "controls"]
        }
        
        # Subtext patterns
        self.subtext_patterns = {
            SubtextType.DECEPTION: ["well actually", "believe me", "trust me"],
            SubtextType.ATTRACTION: ["anyway", "whatever", "fine"],
            SubtextType.CONFLICT: ["sure", "right", "okay"],
            SubtextType.POWER_PLAY: ["perhaps", "maybe", "possibly"]
        }
    
    def analyze_scene(self, script_text: str) -> SceneAnalysis:
        """Perform deep analysis of scene dynamics and character intentions"""
        try:
            doc = self.nlp(script_text)
            
            # Extract and analyze scene elements
            self.scene_context = self._extract_scene_context(doc)
            self._analyze_character_intentions(doc)
            self._detect_subtext_layers(doc)
            self._track_emotional_progression(doc)
            
            # Compile analysis
            analysis = SceneAnalysis(
                context=self.scene_context,
                character_intents=self.character_intents,
                subtext_layers=self.subtext_layers,
                emotional_progression=self.emotional_history,
                power_dynamics=self._analyze_power_dynamics(doc),
                dramatic_beats=self._identify_dramatic_beats(doc)
            )
            
            logger.info(f"Scene analysis complete: {len(analysis.dramatic_beats)} beats identified")
            return analysis
            
        except ValueError as e:
            logger.error(f"Invalid script format: {str(e)}")
            raise
        except spacy.errors.Error as e:
            logger.error(f"NLP processing error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in scene analysis: {str(e)}")
            raise
    
    def _extract_scene_context(self, doc) -> SceneContext:
        """Extract and analyze scene context with atmospheric details"""
        try:
            # Find scene heading
            scene_heading = next((sent for sent in doc.sents 
                                if "INT." in sent.text or "EXT." in sent.text), None)
            
            # Extract location and time
            location = "unknown"
            time_of_day = "unknown"
            if scene_heading:
                parts = scene_heading.text.split(" - ")
                if len(parts) > 0:
                    location = parts[0].replace("INT.", "").replace("EXT.", "").strip()
                if len(parts) > 1:
                    time_of_day = parts[1].strip()
            
            # Analyze scene mood and tension
            mood_indicators = self._analyze_mood_indicators(doc)
            tension_level = self._calculate_tension_level(doc)
            
            # Identify dominant themes through NLP analysis
            themes = []
            for sent in doc.sents:
                # Look for thematic keywords and patterns
                if any(theme in sent.text.lower() for theme in 
                      ["power", "control", "fear", "love", "betrayal", "trust"]):
                    theme = next(theme for theme in 
                               ["power", "control", "fear", "love", "betrayal", "trust"]
                               if theme in sent.text.lower())
                    if theme not in themes:
                        themes.append(theme)
            
            # Generate atmospheric notes through detailed analysis
            atmosphere = []
            # Analyze visual cues
            if any(word in doc.text.lower() for word in ["dark", "shadow", "dim"]):
                atmosphere.append("low-key lighting")
            if any(word in doc.text.lower() for word in ["bright", "sunny", "light"]):
                atmosphere.append("high-key lighting")
                
            # Analyze spatial cues
            if any(word in doc.text.lower() for word in ["cramped", "tight", "close"]):
                atmosphere.append("claustrophobic")
            if any(word in doc.text.lower() for word in ["vast", "open", "wide"]):
                atmosphere.append("spacious")
                
            # Analyze emotional atmosphere
            if any(word in doc.text.lower() for word in ["tense", "nervous", "anxious"]):
                atmosphere.append("high tension")
            if any(word in doc.text.lower() for word in ["calm", "peaceful", "quiet"]):
                atmosphere.append("low tension")
            
            return SceneContext(
                location=location,
                time_of_day=time_of_day,
                mood=mood_indicators["dominant_mood"],
                tension_level=tension_level,
                subtext_complexity=self._calculate_subtext_complexity(doc),
                emotional_valence=self._determine_emotional_valence(mood_indicators),
                dominant_themes=themes if themes else ["unspecified"],
                atmospheric_notes=atmosphere if atmosphere else ["neutral"]
            )
            
        except Exception as e:
            logger.error(f"Error extracting scene context: {str(e)}")
            # Return default context if analysis fails
            return SceneContext(
                location="unknown",
                time_of_day="unknown",
                mood="neutral",
                tension_level=0.5,
                subtext_complexity=0.0,
                emotional_valence=EmotionalValence.NEUTRAL,
                dominant_themes=["unspecified"],
                atmospheric_notes=["neutral"]
            )
    
    def _analyze_character_intentions(self, doc) -> None:
        """Analyze character intentions, goals, and psychological states"""
        # Extract character names
        characters = set()
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                characters.add(ent.text)
        
        # Analyze each character
        for char in characters:
            # Get character's dialogue
            char_dialogue = [sent for sent in doc.sents 
                           if char in sent.text]
            
            # Analyze psychological state
            psych_state = self._analyze_psychological_state(char_dialogue)
            
            # Track character relationships
            relationships = self._analyze_relationships(char, characters, doc)
            
            # Build character arc
            arc = self._build_character_arc(char, doc)
            
            self.character_intents[char] = CharacterIntent(
                stated_goal=self._extract_stated_goal(char_dialogue),
                hidden_goal=self._infer_hidden_goal(char_dialogue, psych_state),
                emotional_subtext=self._analyze_emotional_subtext(char_dialogue),
                power_position=self._calculate_power_position(char, doc),
                vulnerability=psych_state.get("vulnerability", 0.0),
                psychological_state=psych_state,
                relationship_dynamics=relationships,
                character_arc=arc
            )
    
    def _detect_subtext_layers(self, doc) -> None:
        """Identify and analyze layers of subtext in the scene"""
        self.subtext_layers = []
        
        for pattern_type, phrases in self.subtext_patterns.items():
            matches = []
            for phrase in phrases:
                if phrase.lower() in doc.text.lower():
                    matches.append(phrase)
            
            if matches:
                # Find involved characters
                involved = set()
                for sent in doc.sents:
                    if any(phrase in sent.text.lower() for phrase in matches):
                        for ent in sent.ents:
                            if ent.label_ == "PERSON":
                                involved.add(ent.text)
                
                # Calculate intensity
                intensity = len(matches) / len(doc.sents)
                
                # Analyze emotional indicators
                indicators = self._analyze_emotional_indicators(
                    [sent for sent in doc.sents 
                     if any(phrase in sent.text.lower() for phrase in matches)]
                )
                
                self.subtext_layers.append(SubtextLayer(
                    type=pattern_type,
                    intensity=intensity,
                    involved_characters=list(involved),
                    trigger_phrases=matches,
                    emotional_indicators=indicators
                ))
    
    def _track_emotional_progression(self, doc) -> None:
        """Track emotional changes through the scene"""
        self.emotional_history = []
        
        for i, sent in enumerate(doc.sents):
            time_point = i / len(list(doc.sents))
            emotional_state = {}
            
            # Analyze emotion in sentence
            for char in self.character_intents.keys():
                if char in sent.text:
                    emotional_state[char] = self._analyze_emotional_state(sent)
            
            if emotional_state:
                self.emotional_history.append((time_point, emotional_state))
    
    def _analyze_power_dynamics(self, doc) -> List[Dict[str, float]]:
        """Analyze power dynamics between characters"""
        dynamics = []
        
        for sent in doc.sents:
            power_state = {}
            for char in self.character_intents.keys():
                if char in sent.text:
                    # Calculate power level from various factors
                    verb_power = self._analyze_verb_power(sent)
                    position_power = self._analyze_syntactic_position(char, sent)
                    modal_power = self._analyze_modal_verbs(sent)
                    
                    # Combine power factors
                    power_state[char] = np.mean([
                        verb_power,
                        position_power,
                        modal_power
                    ])
            
            if power_state:
                dynamics.append(power_state)
        
        return dynamics
    
    def _identify_dramatic_beats(self, doc) -> List[Dict[str, Any]]:
        """Identify and analyze dramatic beats in the scene"""
        beats = []
        
        for i, sent in enumerate(doc.sents):
            # Check for dramatic significance
            power_shift = self._detect_power_shift(sent)
            emotional_change = self._detect_emotional_change(sent)
            revelation = self._detect_revelation(sent)
            
            if power_shift or emotional_change or revelation:
                beats.append({
                    "position": i / len(list(doc.sents)),
                    "type": "power_shift" if power_shift else 
                           "emotional_change" if emotional_change else 
                           "revelation",
                    "intensity": max(power_shift, emotional_change, revelation),
                    "characters": self._get_involved_characters(sent),
                    "subtext": self._analyze_beat_subtext(sent)
                })
        
        return beats
    
    # Helper methods
    def _analyze_mood_indicators(self, doc) -> Dict[str, Any]:
        """Analyze various indicators of scene mood"""
        return {"dominant_mood": "tense"}  # Simplified for example
    
    def _calculate_tension_level(self, doc) -> float:
        """Calculate scene tension level"""
        return 0.7  # Simplified for example
    
    def _extract_themes(self, doc) -> List[str]:
        """Extract dominant themes from scene"""
        return ["power", "deception"]  # Simplified for example
    
    def _analyze_atmosphere(self, doc) -> List[str]:
        """Analyze atmospheric elements"""
        return ["dark", "claustrophobic"]  # Simplified for example
    
    def _calculate_subtext_complexity(self, doc) -> float:
        """Calculate complexity of scene subtext"""
        return 0.8  # Simplified for example
    
    def _determine_emotional_valence(self, mood_indicators: Dict) -> EmotionalValence:
        """Determine overall emotional valence"""
        return EmotionalValence.COMPLEX  # Simplified for example
    
    def _extract_stated_goal(self, dialogue) -> str:
        """Extract character's stated goal"""
        return "achieve reconciliation"  # Simplified for example
    
    def _infer_hidden_goal(self, dialogue, psych_state: Dict) -> str:
        """Infer character's hidden goal"""
        return "maintain control"  # Simplified for example
    
    def _analyze_emotional_subtext(self, dialogue) -> str:
        """Analyze emotional subtext in dialogue"""
        return "suppressed anger"  # Simplified for example
    
    def _calculate_power_position(self, char: str, doc) -> float:
        """Calculate character's power position"""
        return 0.7  # Simplified for example
    
    def _analyze_psychological_state(self, dialogue) -> Dict[str, float]:
        """Analyze character's psychological state"""
        return {
            "anxiety": 0.3,
            "confidence": 0.7,
            "vulnerability": 0.4
        }  # Simplified for example
    
    def _analyze_relationships(self, char: str, all_chars: set, doc) -> Dict[str, float]:
        """Analyze character's relationships"""
        return {other: 0.5 for other in all_chars if other != char}  # Simplified
    
    def _build_character_arc(self, char: str, doc) -> List[Tuple[float, str]]:
        """Build character's emotional/psychological arc"""
        return [(0.0, "confident"), (1.0, "uncertain")]  # Simplified for example
    
    def _analyze_emotional_state(self, sent) -> float:
        """Analyze emotional state from sentence"""
        return 0.6  # Simplified for example
    
    def _analyze_verb_power(self, sent) -> float:
        """Analyze power level of verbs"""
        return 0.5  # Simplified for example
    
    def _analyze_syntactic_position(self, char: str, sent) -> float:
        """Analyze power from syntactic position"""
        return 0.6  # Simplified for example
    
    def _analyze_modal_verbs(self, sent) -> float:
        """Analyze power from modal verb usage"""
        return 0.4  # Simplified for example
    
    def _detect_power_shift(self, sent) -> float:
        """Detect power shifts in sentence"""
        return 0.0  # Simplified for example
    
    def _detect_emotional_change(self, sent) -> float:
        """Detect emotional changes in sentence"""
        return 0.0  # Simplified for example
    
    def _detect_revelation(self, sent) -> float:
        """Detect revelations in sentence"""
        return 0.0  # Simplified for example
    
    def _get_involved_characters(self, sent) -> List[str]:
        """Get characters involved in sentence"""
        return []  # Simplified for example
    
    def _analyze_beat_subtext(self, sent) -> str:
        """Analyze subtext of dramatic beat"""
        return "hidden conflict"  # Simplified for example
    
    def _analyze_emotional_indicators(self, sents) -> Dict[str, float]:
        """Analyze emotional indicators in sentences"""
        return {"tension": 0.7, "conflict": 0.5}  # Simplified for example
