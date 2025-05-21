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
        self.nlp = spacy.load("en_core_web_trf")
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
        for ent in doc.ents:
            if ent.label_ == "CHARACTER" and ent.text not in self.characters:
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
        from monitor import monitor, PerformanceMetrics, SceneMetrics
        
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
                metrics = PerformanceMetrics(
                    power_level=char.power_level,
                    emotional_intensity=abs(shift) * 2,
                    gesture_frequency=0.5 if abs(shift) > 0.2 else 0.2,
                    gaze_stability=0.9 if char.power_level > 0.7 else 0.5,
                    dialogue_pace=1.0 + (char.power_level - 0.5)
                )
                monitor.update_character_metrics(char_name, metrics)
                
                # Track emotional states with LLM enhancement
                emotion = await self._calculate_emotion_with_llm(sent, char_name)
                emotional_states.append(emotion)
                
        # Update scene-level metrics
        scene_metrics = SceneMetrics(
            tension=monitor.calculate_tension(power_shifts),
            power_gradient=max(abs(ps) for ps in power_shifts) if power_shifts else 0,
            emotional_coherence=monitor.calculate_coherence(emotional_states),
            beat_progression=power_shifts
        )
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
        - Verb transitivity patterns
        - Sentence mood (imperative/declarative)
        - Semantic role labeling
        """
        power_delta = 0.0
        
        # Analyze grammatical structure
        for token in sent:
            if token.text == char_name:
                # Check grammatical role
                if token.dep_ in ['nsubj', 'nsubjpass']:
                    power_delta += 0.1 if 'subj' in token.dep_ else -0.05
                
                # Check verb transitivity
                if token.head.pos_ == 'VERB':
                    if token.head._.power_verb_type == 'dominance':
                        power_delta += 0.15
                    elif token.head._.power_verb_type == 'submission':
                        power_delta -= 0.1
        
        # Check sentence type        
        if sent._.sentiment['polarity'] > 0.3:
            power_delta += 0.07 * sent._.sentiment['intensity']
        
        # Apply scene position weighting
        position_weight = 1.0 - (sent.start_char / len(sent.doc.text))
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
