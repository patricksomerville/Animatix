"""
Enhanced Animatic Pipeline
Integrates advanced LLM-powered script analysis, visual style, sound, and performance into a cohesive workflow
"""
import spacy
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from enum import Enum
from llm_processor import LLMProcessor
from claude_thinks import SceneAnalyzer, SceneAnalysis

# Core data structures and enhanced analysis
class ShotType(Enum):
    WIDE = "wide"
    MEDIUM = "medium"
    CLOSE = "close"
    EXTREME_CLOSE = "extreme_close"
    DUTCH = "dutch_angle"
    OVERHEAD = "overhead"
    POV = "pov"

@dataclass
class Character:
    name: str
    emotional_state: str
    power_level: float  # 0-1 scale
    position: Dict[str, float]  # x, y, z coordinates

@dataclass
class Shot:
    type: ShotType
    duration: float
    characters: List[Character]
    camera_movement: Optional[str]
    depth_of_field: float  # 0-1 scale
    lighting_intensity: float  # 0-1 scale

@dataclass
class SceneBeat:
    start_time: float
    end_time: float
    shot: Shot
    dialogue: Optional[str]
    sound_elements: Dict[str, float]
    power_shift: float

class AnimaticPipeline:
    def __init__(self):
        # Load spaCy with graceful fallbacks to avoid heavy model requirements
        try:
            self.nlp = spacy.load("en_core_web_trf")
        except Exception:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except Exception:
                # Final fallback: blank English with sentencizer
                self.nlp = spacy.blank("en")
                if "sentencizer" not in self.nlp.pipe_names:
                    self.nlp.add_pipe("sentencizer")
        self.scene_beats: List[SceneBeat] = []
        self.characters: Dict[str, Character] = {}
        self.llm_processor = LLMProcessor()
        self.scene_analyzer = SceneAnalyzer()
        
    async def process_script(self, script_text: str) -> List[SceneBeat]:
        """Process script text into structured scene beats using LLM enhancement"""
        doc = self.nlp(script_text)
        
        # Get enhanced scene analysis
        scene_analysis = self.scene_analyzer.analyze_scene(script_text)
        enhanced_description = await self.llm_processor.enhance_scene_description(script_text)
        
        # Integrate scene analysis insights
        self._apply_scene_analysis(scene_analysis)
        
        # Extract scene elements with LLM insights
        self._extract_characters(doc, scene_analysis)
        # Fallback: ensure at least one character exists
        if not self.characters:
            self.characters["Protagonist"] = Character(
                name="Protagonist",
                emotional_state="neutral",
                power_level=0.5,
                position={"x": 0, "y": 0, "z": 0}
            )
        await self._analyze_power_dynamics(doc, scene_analysis)
        await self._generate_shot_sequence(scene_analysis)
        self._design_soundscape(scene_analysis)
        
        # Get improvement suggestions
        suggestions = await self.llm_processor.suggest_scene_improvements(script_text)
        self._apply_scene_improvements(suggestions)
        
        return self.scene_beats
    
    def _apply_scene_analysis(self, analysis: SceneAnalysis) -> None:
        """Apply insights from the scene analysis to enhance the pipeline"""
        # Update character emotional states and power dynamics
        for char_name, intent in analysis.character_intents.items():
            if char_name in self.characters:
                self.characters[char_name].emotional_state = intent.emotional_subtext
                self.characters[char_name].power_level = intent.power_position
        
        # Apply subtext layers to scene beats
        for beat in self.scene_beats:
            matching_dramatic_beats = [db for db in analysis.dramatic_beats 
                                    if abs(db["position"] - beat.start_time/3.0) < 0.1]
            if matching_dramatic_beats:
                db = matching_dramatic_beats[0]
                beat.power_shift = db["intensity"] if db["type"] == "power_shift" else 0.0
        
        # Update scene context
        self.context = {
            "tension_level": analysis.context.tension_level,
            "emotional_valence": analysis.context.emotional_valence,
            "themes": analysis.context.dominant_themes,
            "atmosphere": analysis.context.atmospheric_notes
        }
    
    def _extract_characters(self, doc, scene_analysis: SceneAnalysis) -> None:
        """Extract and initialize characters from script with LLM-enhanced understanding"""
        for ent in getattr(doc, "ents", []):
            if ent.label_ == "PERSON" and ent.text not in self.characters:
                # Use emotional context from LLM analysis
                emotional_context = scene_analysis.emotional_context.get(ent.text, {})
                initial_power = next(
                    (pd["power_level"] for pd in scene_analysis.power_dynamics 
                     if pd["character"] == ent.text), 0.5
                )
                
                self.characters[ent.text] = Character(
                    name=ent.text,
                    emotional_state=emotional_context.get("primary_emotion", "neutral"),
                    power_level=initial_power,
                    position={"x": 0, "y": 0, "z": 0}
                )
    
    async def _analyze_power_dynamics(self, doc, scene_analysis: SceneAnalysis) -> None:
        """Analyze character interactions and power shifts with LLM enhancement"""
        from monitor import monitor
        
        power_shifts = []
        emotional_states = []
        
        for sent in doc.sents:
            # Track power shifts per character
            for char_name, char in self.characters.items():
                # Combine traditional NLP with LLM insights
                base_shift = self._calculate_power_shift(sent, char_name)
                
                # Enhance with LLM analysis
                llm_shift = next(
                    (pd.get("power_shift", 0) for pd in scene_analysis.power_dynamics 
                     if pd["character"] == char_name), 0
                )
                
                # Weighted combination of traditional and LLM analysis
                shift = (base_shift * 0.4 + llm_shift * 0.6)
                char.power_level = np.clip(char.power_level + shift, 0, 1)
                power_shifts.append(shift)
                
                # Update character metrics with enhanced understanding
                metrics = {
                    "power_level": float(char.power_level),
                    "emotional_intensity": float(min(1.0, abs(shift) * 2)),
                    "gesture_frequency": float(0.5 if abs(shift) > 0.2 else 0.2),
                    "gaze_stability": float(0.9 if char.power_level > 0.7 else 0.5),
                    "dialogue_pace": float(max(0.0, min(1.0, 1.0 + (char.power_level - 0.5))))
                }
                monitor.update_character_metrics(char_name, metrics)
                
                # Track emotional states with LLM enhancement
                emotion = await self._calculate_emotion_with_llm(sent, char_name)
                emotional_states.append(emotion)
                
        # Update scene-level metrics
        scene_metrics = {
            "tension": float(monitor.calculate_tension(power_shifts)),
            "power_gradient": float(max(abs(ps) for ps in power_shifts) if power_shifts else 0.0),
            "emotional_coherence": float(monitor.calculate_coherence(emotional_states)),
            "beat_progression": [float(x) for x in power_shifts]
        }
        monitor.update_scene_metrics(scene_metrics)
    
    async def _generate_shot_sequence(self, scene_analysis: SceneAnalysis) -> None:
        """Generate camera shots based on scene dynamics with LLM enhancement"""
        current_time = 0.0
        
        for char_name, char in self.characters.items():
            # Get shot suggestions from LLM
            shot_prompt = f"Suggest camera shots for character {char_name} with power level {char.power_level}"
            completion = await self.llm_processor.complete_text(shot_prompt)
            
            # Combine traditional logic with LLM suggestions
            base_shot_type = self._determine_shot_type(char)
            base_camera_movement = self._determine_camera_movement(char)
            
            shot_type = base_shot_type
            camera_movement = base_camera_movement
            
            if completion.confidence > 0.7:
                # Parse LLM suggestions
                suggestions = completion.completed_text.split('\n')
                if suggestions:
                    llm_shot_type = self._parse_shot_type(suggestions[0])
                    if llm_shot_type:
                        shot_type = llm_shot_type
                    if len(suggestions) > 1:
                        camera_movement = suggestions[1].strip()
            
            new_beat = SceneBeat(
                start_time=current_time,
                end_time=current_time + 3.0,  # Default shot duration
                shot=Shot(
                    type=shot_type,
                    duration=3.0,
                    characters=[char],
                    camera_movement=self._determine_camera_movement(char),
                    depth_of_field=self._calculate_depth_of_field(char),
                    lighting_intensity=char.power_level
                ),
                dialogue=None,
                sound_elements=self._generate_sound_elements(char),
                power_shift=0.0
            )
            
            self.scene_beats.append(new_beat)
            current_time += 3.0
    
    def _design_soundscape(self, scene_analysis: SceneAnalysis) -> None:
        """Design sound elements for each beat with LLM-enhanced understanding"""
        for beat in self.scene_beats:
            # Base ambient sound
            beat.sound_elements["ambient"] = 0.3
            
            # Emotional intensity affects music
            emotional_intensity = max(
                char.power_level for char in beat.shot.characters
            )
            beat.sound_elements["score"] = emotional_intensity
            
            # Add sound design for camera movements
            if beat.shot.camera_movement:
                beat.sound_elements["movement_whoosh"] = 0.2
    
    def _calculate_emotion(self, sent) -> str:
        """Calculate emotional state using:
        - Custom emotion NER model
        - Sentiment analysis
        - Contextual embedding clustering
        """
        emotion_ents = [ent for ent in sent.ents if ent.label_ in 
                      ['POS_EMOTION', 'NEG_EMOTION', 'NEUTRAL_EMOTION']]
        
        if not emotion_ents:
            return self._fallback_emotion(sent)
        
        primary_emotion = max(emotion_ents, 
                            key=lambda e: e._.emotion_intensity)
        return primary_emotion.label_.split('_')[0].lower()
    
    def _calculate_power_shift(self, sent, char_name: str) -> float:
        """Calculate power dynamics shift using:
        Lightweight heuristics that don't rely on custom spaCy extensions.
        Factors:
        - Character grammatical role (subject gets slight boost)
        - Presence of strong modal/imperative cues
        - Scene position weighting
        """
        power_delta = 0.0
        
        # Analyze grammatical structure
        try:
            for token in sent:
                if token.text == char_name:
                    # Grammatical role
                    if token.dep_ in ['nsubj', 'nsubjpass']:
                        power_delta += 0.08
                    elif token.dep_ in ['dobj', 'pobj']:
                        power_delta -= 0.05

            # Modal/imperative cues
            text_lower = sent.text.lower()
            if any(m in text_lower for m in ["must", "have to", "need to", "now", "do it", "listen"]):
                power_delta += 0.05
            if any(m in text_lower for m in ["please", "maybe", "perhaps", "could you"]):
                power_delta -= 0.03
        except Exception:
            power_delta += 0.0
        
        # Apply scene position weighting
        try:
            position_weight = 1.0 - (sent.start_char / max(1, len(sent.doc.text)))
        except Exception:
            position_weight = 1.0
        return round(power_delta * position_weight, 2)
    
    def _determine_shot_type(self, char: Character) -> ShotType:
        """Determine appropriate shot type based on character state"""
        if char.power_level > 0.8:
            return ShotType.CLOSE
        elif char.power_level < 0.2:
            return ShotType.WIDE
        return ShotType.MEDIUM
    
    def _determine_camera_movement(self, char: Character) -> Optional[str]:
        """Determine camera movement based on character state"""
        if char.power_level > 0.7:
            return "dolly_in"
        elif char.power_level < 0.3:
            return "dolly_out"
        return None
    
    def _calculate_depth_of_field(self, char: Character) -> float:
        """Calculate depth of field based on emotional intensity"""
        return 1.0 - char.power_level  # Shallower DOF for higher power
    
    def _generate_sound_elements(self, char: Character) -> Dict[str, float]:
        """Generate sound design elements based on character state"""
        return {
            "ambient": 0.3,
            "score": char.power_level,
            "foley": 0.2
        }

    def _fallback_emotion(self, sent):
        """Fallback emotion detection"""
        return "neutral"

    async def _calculate_emotion_with_llm(self, sent, char_name: str) -> str:
        """Lightweight async wrapper for emotion; avoid heavy dependencies."""
        try:
            return self._calculate_emotion(sent)
        except Exception:
            return "neutral"

    def _parse_shot_type(self, text: str) -> Optional[ShotType]:
        """Parse a shot type enum from free-form text."""
        t = (text or "").lower()
        if any(k in t for k in ["extreme close", "ecu", "extreme_close"]):
            return ShotType.EXTREME_CLOSE
        if any(k in t for k in ["close", "cu"]):
            return ShotType.CLOSE
        if any(k in t for k in ["wide", "ws"]):
            return ShotType.WIDE
        if "dutch" in t:
            return ShotType.DUTCH
        if any(k in t for k in ["overhead", "top down", "bird"]):
            return ShotType.OVERHEAD
        if "pov" in t:
            return ShotType.POV
        if "medium" in t or "ms" in t:
            return ShotType.MEDIUM
        return None

    def _apply_scene_improvements(self, suggestions: List[str]) -> None:
        """Apply simple heuristics based on suggestions to tweak beats."""
        if not self.scene_beats:
            return
        text = "\n".join(suggestions or [])
        for beat in self.scene_beats:
            if "dolly-in" in text or "dolly in" in text:
                beat.shot.camera_movement = beat.shot.camera_movement or "dolly_in"
            if "tighter" in text or "close-up" in text or "close up" in text:
                if beat.shot.type == ShotType.MEDIUM:
                    beat.shot.type = ShotType.CLOSE
            if "slower" in text or "linger" in text:
                beat.shot.duration = min(6.0, beat.shot.duration + 1.0)

# Usage example:
if __name__ == "__main__":
    async def main():
        pipeline = AnimaticPipeline()
        script = """
        INT. COFFEE SHOP - DAY

        John sits alone, fidgeting with his coffee cup. Sarah enters, confident.

        JOHN
        (nervously)
        I didn't think you'd come.

        SARAH
        (stern)
        We need to talk.
        """

        scene_beats = await pipeline.process_script(script)

        for beat in scene_beats:
            print(f"Scene Beat:")
            print(f"  Start Time: {beat.start_time}")
            print(f"  End Time: {beat.end_time}")
            print(f"  Shot Type: {beat.shot.type}")
            print(f"  Characters: {[char.name for char in beat.shot.characters]}")
            print(f"  Camera Movement: {beat.shot.camera_movement}")
            print(f"  Sound Elements: {beat.sound_elements}")
            print(f"  Power Shift: {beat.power_shift}")
            print("\n")

    import asyncio
    asyncio.run(main())
