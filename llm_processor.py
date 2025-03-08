"""
LLM Processor module for enhanced AI capabilities in Animatix
"""
import os
from typing import Dict, List, Optional, Union
from pydantic import BaseModel
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import tiktoken

# Load environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

class SceneAnalysis(BaseModel):
    """Scene analysis results from LLM processing"""
    emotional_context: Dict[str, float]
    power_dynamics: List[Dict[str, Union[str, float]]]
    suggested_improvements: List[str]
    scene_coherence: float

class TextCompletion(BaseModel):
    """Text completion results"""
    completed_text: str
    alternatives: List[str]
    confidence: float

class LLMProcessor:
    """Handles all LLM-related processing for scene enhancement"""
    
    def __init__(self):
        """Initialize LLM processor with necessary models"""
        self.sentiment_analyzer = pipeline("sentiment-analysis")
        self.text_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def analyze_scene(self, scene_text: str) -> SceneAnalysis:
        """
        Analyze scene text using GPT-4 for deep understanding
        """
        prompt = f"""Analyze the following scene, focusing on:
        1. Emotional context and undertones
        2. Power dynamics between characters
        3. Potential improvements for visual and dramatic impact
        4. Overall scene coherence
        
        Scene:
        {scene_text}
        
        Provide a structured analysis that can be used for visual direction.
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert scene analyst focusing on emotional depth, character dynamics, and visual storytelling."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        # Process the response into structured data
        analysis = response.choices[0].message.content
        # Convert the analysis into structured data...
        
        return SceneAnalysis(
            emotional_context={},  # Extracted from analysis
            power_dynamics=[],     # Extracted from analysis
            suggested_improvements=[],  # Extracted from analysis
            scene_coherence=0.0    # Calculated from analysis
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def complete_text(self, partial_text: str, context: Optional[str] = None) -> TextCompletion:
        """
        Provide intelligent text completion based on context
        """
        system_prompt = """You are an expert storyteller and scene writer.
        Complete the text in a way that enhances the narrative while maintaining
        consistency with the established tone and style."""
        
        context_msg = f"Context: {context}\n" if context else ""
        prompt = f"{context_msg}Complete the following text:\n{partial_text}"
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=500,
            n=3  # Generate 3 alternatives
        )
        
        completions = [choice.message.content for choice in response.choices]
        
        return TextCompletion(
            completed_text=completions[0],
            alternatives=completions[1:],
            confidence=0.9  # Can be calculated based on model's confidence
        )
    
    async def enhance_scene_description(self, scene_text: str) -> str:
        """
        Enhance scene description with more vivid and cinematic details
        """
        prompt = f"""Enhance the following scene description to be more cinematic
        and visually compelling, while maintaining the original narrative intent:
        
        {scene_text}
        
        Focus on:
        1. Visual composition and framing
        2. Lighting and atmosphere
        3. Character positioning and movement
        4. Emotional undertones through visual elements
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert cinematographer and visual storyteller."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    
    def get_token_count(self, text: str) -> int:
        """Get the number of tokens in a text string"""
        return len(self.encoding.encode(text))
    
    def get_text_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text using sentence transformers"""
        return self.text_embedder.encode(text).tolist()
    
    async def suggest_scene_improvements(self, scene_text: str) -> List[str]:
        """
        Suggest improvements for scene visualization and impact
        """
        prompt = f"""Analyze this scene and suggest specific improvements
        for its visual adaptation and dramatic impact:
        
        {scene_text}
        
        Focus on:
        1. Camera angles and movements
        2. Lighting and color schemes
        3. Sound design opportunities
        4. Character blocking and staging
        5. Emotional emphasis points
        """
        
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert film director and visual storyteller."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.8,
            max_tokens=800
        )
        
        # Process suggestions into a structured list
        suggestions_text = response.choices[0].message.content
        suggestions = [s.strip() for s in suggestions_text.split('\n') if s.strip()]
        return suggestions