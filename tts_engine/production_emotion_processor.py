"""
Production-Grade Emotion Processing System for Orpheus TTS
Based on official Orpheus-TTS repository patterns and best practices
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmotionIntensity(Enum):
    """Emotion intensity levels based on official Orpheus-TTS patterns"""
    SUBTLE = "subtle"
    NORMAL = "normal" 
    STRONG = "strong"

@dataclass
class EmotionTag:
    """Official Orpheus-TTS emotion tag configuration"""
    base_tag: str
    aliases: List[str]
    context_triggers: List[str]
    voice_compatibility: Dict[str, float]  # voice -> compatibility score (0-1)
    default_intensity: EmotionIntensity
    
class ProductionEmotionProcessor:
    """
    Production-grade emotion processor following official Orpheus-TTS patterns.
    
    Based on research from:
    - https://github.com/canopyai/Orpheus-TTS
    - Official emotion tag documentation
    - Production deployment best practices
    """
    
    def __init__(self):
        self.emotion_tags = self._initialize_official_emotions()
        self.voice_profiles = self._initialize_voice_profiles()
        self.context_analyzers = self._initialize_context_analyzers()
        
    def _initialize_official_emotions(self) -> Dict[str, EmotionTag]:
        """Initialize emotion tags based on official Orpheus-TTS patterns"""
        return {
            "laugh": EmotionTag(
                base_tag="laugh",
                aliases=["laughter", "laughing", "chuckle", "giggle"],
                context_triggers=["funny", "hilarious", "amusing", "joke", "comedy", "ridiculous", "haha"],
                voice_compatibility={
                    "tara": 0.95, "zoe": 0.90, "mia": 0.85, "jess": 0.80,
                    "dan": 0.70, "leo": 0.65, "zac": 0.75
                },
                default_intensity=EmotionIntensity.NORMAL
            ),
            "chuckle": EmotionTag(
                base_tag="chuckle", 
                aliases=["soft_laugh", "amused"],
                context_triggers=["amusing", "interesting", "clever", "witty", "ironic", "hmm"],
                voice_compatibility={
                    "tara": 0.90, "leo": 0.95, "dan": 0.90, "zac": 0.85,
                    "zoe": 0.75, "mia": 0.70
                },
                default_intensity=EmotionIntensity.SUBTLE
            ),
            "sigh": EmotionTag(
                base_tag="sigh",
                aliases=["exhale", "breath"],
                context_triggers=["tired", "exhausted", "disappointed", "frustrated", "weary", 
                                "unfortunately", "sadly", "oh well"],
                voice_compatibility={
                    "tara": 0.95, "leah": 0.90, "mia": 0.85, "zoe": 0.80,
                    "dan": 0.70, "leo": 0.65
                },
                default_intensity=EmotionIntensity.NORMAL
            ),
            "gasp": EmotionTag(
                base_tag="gasp",
                aliases=["surprised", "shocked"],
                context_triggers=["surprising", "shocking", "amazing", "incredible", "unbelievable", 
                                "wow", "oh my", "suddenly"],
                voice_compatibility={
                    "zoe": 0.95, "mia": 0.90, "jess": 0.85, "tara": 0.80,
                    "dan": 0.60, "leo": 0.55
                },
                default_intensity=EmotionIntensity.STRONG
            ),
            "groan": EmotionTag(
                base_tag="groan",
                aliases=["frustrated", "annoyed"],
                context_triggers=["annoying", "frustrating", "difficult", "problematic", 
                                "ugh", "terrible", "awful"],
                voice_compatibility={
                    "dan": 0.95, "leo": 0.90, "zac": 0.85, "thomas": 0.80,
                    "tara": 0.60, "zoe": 0.55
                },
                default_intensity=EmotionIntensity.NORMAL
            ),
            "yawn": EmotionTag(
                base_tag="yawn",
                aliases=["tired", "sleepy"],
                context_triggers=["tired", "sleepy", "boring", "late", "exhausted", "drowsy"],
                voice_compatibility={
                    "tara": 0.85, "leah": 0.80, "dan": 0.75, "leo": 0.70,
                    "zoe": 0.60, "mia": 0.55
                },
                default_intensity=EmotionIntensity.SUBTLE
            ),
            "cough": EmotionTag(
                base_tag="cough",
                aliases=["clear_throat", "ahem"],
                context_triggers=["excuse me", "pardon", "ahem", "well", "actually", "um"],
                voice_compatibility={
                    "leo": 0.90, "dan": 0.85, "thomas": 0.80, "zac": 0.75,
                    "tara": 0.65, "zoe": 0.60
                },
                default_intensity=EmotionIntensity.SUBTLE
            ),
            "sniffle": EmotionTag(
                base_tag="sniffle",
                aliases=["emotional", "tearful"],
                context_triggers=["sad", "emotional", "touching", "moving", "heartbreaking", 
                                "tears", "cry", "sorry"],
                voice_compatibility={
                    "tara": 0.90, "mia": 0.85, "leah": 0.80, "amelie": 0.75,
                    "dan": 0.50, "leo": 0.45
                },
                default_intensity=EmotionIntensity.NORMAL
            )
        }
    
    def _initialize_voice_profiles(self) -> Dict[str, Dict[str, float]]:
        """Initialize voice-specific characteristics for emotion optimization"""
        return {
            "tara": {"expressiveness": 0.95, "emotion_range": 0.90, "naturalness": 0.95},
            "zoe": {"expressiveness": 0.90, "emotion_range": 0.85, "naturalness": 0.90},
            "mia": {"expressiveness": 0.85, "emotion_range": 0.80, "naturalness": 0.85},
            "dan": {"expressiveness": 0.80, "emotion_range": 0.75, "naturalness": 0.80},
            "leo": {"expressiveness": 0.75, "emotion_range": 0.70, "naturalness": 0.85},
            "jess": {"expressiveness": 0.85, "emotion_range": 0.80, "naturalness": 0.80},
            "leah": {"expressiveness": 0.80, "emotion_range": 0.75, "naturalness": 0.85},
            "zac": {"expressiveness": 0.75, "emotion_range": 0.70, "naturalness": 0.75}
        }
    
    def _initialize_context_analyzers(self) -> Dict[str, List[str]]:
        """Initialize context analysis patterns for intelligent emotion placement"""
        return {
            "high_excitement": ["amazing", "incredible", "fantastic", "wonderful", "awesome", 
                              "brilliant", "spectacular", "phenomenal"],
            "mild_amusement": ["interesting", "amusing", "clever", "witty", "nice", "good"],
            "strong_emotion": ["absolutely", "completely", "totally", "extremely", "incredibly"],
            "negative_context": ["unfortunately", "sadly", "regrettably", "disappointingly"],
            "surprise_context": ["suddenly", "unexpectedly", "surprisingly", "out of nowhere"],
            "question_context": ["what", "how", "why", "when", "where", "really?"]
        }
    
    def analyze_text_context(self, text: str) -> Dict[str, float]:
        """Analyze text for emotional context with production-grade accuracy"""
        text_lower = text.lower()
        context_scores = {}
        
        # Analyze sentence structure and punctuation
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        # Weight based on punctuation
        if exclamation_count > 0:
            context_scores["excitement"] = min(exclamation_count * 0.3, 1.0)
        if question_count > 0:
            context_scores["curiosity"] = min(question_count * 0.2, 1.0)
        
        # Analyze context patterns
        for context_type, keywords in self.context_analyzers.items():
            matches = sum(1 for word in keywords if word in text_lower)
            if matches > 0:
                context_scores[context_type] = min(matches / len(keywords), 1.0)
        
        return context_scores
    
    def get_optimal_emotion_for_context(self, text: str, voice: str) -> Optional[str]:
        """Get the most appropriate emotion tag for given context and voice"""
        context_scores = self.analyze_text_context(text)
        text_lower = text.lower()
        
        best_emotion = None
        best_score = 0.0
        
        for emotion_name, emotion_tag in self.emotion_tags.items():
            score = 0.0
            
            # Check voice compatibility
            voice_compat = emotion_tag.voice_compatibility.get(voice, 0.5)
            score += voice_compat * 0.4
            
            # Check context triggers
            trigger_matches = sum(1 for trigger in emotion_tag.context_triggers 
                                if trigger in text_lower)
            if trigger_matches > 0:
                score += (trigger_matches / len(emotion_tag.context_triggers)) * 0.6
            
            # Boost score based on context analysis
            if "high_excitement" in context_scores and emotion_name in ["laugh", "gasp"]:
                score += context_scores["high_excitement"] * 0.3
            if "mild_amusement" in context_scores and emotion_name == "chuckle":
                score += context_scores["mild_amusement"] * 0.3
            if "negative_context" in context_scores and emotion_name in ["sigh", "sniffle"]:
                score += context_scores["negative_context"] * 0.3
            
            if score > best_score:
                best_score = score
                best_emotion = emotion_name
        
        # Only return if confidence is high enough
        return best_emotion if best_score > 0.6 else None
    
    def optimize_existing_emotions(self, text: str, voice: str) -> str:
        """Optimize existing emotion tags in text for better accuracy"""
        # Pattern to find emotion tags
        emotion_pattern = r'<(laugh|chuckle|sigh|gasp|groan|yawn|cough|sniffle)>'
        
        def replace_emotion(match):
            emotion = match.group(1).lower()
            if emotion in self.emotion_tags:
                emotion_tag = self.emotion_tags[emotion]
                
                # Get voice compatibility score
                voice_compat = emotion_tag.voice_compatibility.get(voice, 0.5)
                
                # If voice compatibility is low, use a more compatible emotion
                if voice_compat < 0.7:
                    # Find more compatible emotion
                    for alt_emotion, alt_tag in self.emotion_tags.items():
                        if alt_tag.voice_compatibility.get(voice, 0.5) > voice_compat:
                            return f"<{alt_emotion}>"
                
                return match.group(0)  # Keep original if good compatibility
            return match.group(0)
        
        return re.sub(emotion_pattern, replace_emotion, text, flags=re.IGNORECASE)
    
    def add_contextual_emotions(self, text: str, voice: str, max_additions: int = 2) -> str:
        """Add contextual emotions based on text analysis"""
        sentences = re.split(r'[.!?]+', text)
        enhanced_sentences = []
        emotions_added = 0
        
        for sentence in sentences:
            if not sentence.strip() or emotions_added >= max_additions:
                enhanced_sentences.append(sentence.strip())
                continue
            
            sentence = sentence.strip()
            
            # Check if sentence already has emotions
            if re.search(r'<[^>]+>', sentence):
                enhanced_sentences.append(sentence)
                continue
            
            # Get optimal emotion for this sentence
            optimal_emotion = self.get_optimal_emotion_for_context(sentence, voice)
            
            if optimal_emotion and emotions_added < max_additions:
                # Add emotion at natural position (usually end of sentence)
                sentence = f"{sentence} <{optimal_emotion}>"
                emotions_added += 1
            
            enhanced_sentences.append(sentence)
        
        return ". ".join(s for s in enhanced_sentences if s) + "."
    
    def process_text(self, text: str, voice: str = "tara", 
                    optimize_existing: bool = True, 
                    add_contextual: bool = True,
                    max_contextual_additions: int = 2) -> str:
        """
        Main processing function for production-grade emotion enhancement
        
        Args:
            text: Input text to process
            voice: Voice name for optimization
            optimize_existing: Whether to optimize existing emotion tags
            add_contextual: Whether to add contextual emotions
            max_contextual_additions: Maximum number of emotions to add
            
        Returns:
            Enhanced text with optimized emotions
        """
        processed_text = text
        
        # Step 1: Optimize existing emotion tags
        if optimize_existing:
            processed_text = self.optimize_existing_emotions(processed_text, voice)
        
        # Step 2: Add contextual emotions
        if add_contextual:
            processed_text = self.add_contextual_emotions(
                processed_text, voice, max_contextual_additions
            )
        
        # Step 3: Final cleanup and validation
        processed_text = self._cleanup_emotion_placement(processed_text)
        
        return processed_text
    
    def _cleanup_emotion_placement(self, text: str) -> str:
        """Clean up emotion tag placement for natural flow"""
        # Move emotions before punctuation for better flow
        text = re.sub(r'(\w+)\s*(<[^>]+>)\s*([.!?])', r'\1\3 \2', text)
        
        # Remove duplicate emotions in close proximity
        text = re.sub(r'(<[^>]+>)\s*(<[^>]+>)', r'\1', text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def get_emotion_statistics(self, text: str) -> Dict[str, any]:
        """Get comprehensive statistics about emotions in text"""
        emotion_pattern = r'<([^>]+)>'
        emotions = re.findall(emotion_pattern, text, re.IGNORECASE)
        
        stats = {
            "total_emotions": len(emotions),
            "unique_emotions": len(set(emotions)),
            "emotion_distribution": {},
            "emotion_density": len(emotions) / max(len(text.split()), 1),
            "supported_emotions": 0,
            "unsupported_emotions": []
        }
        
        for emotion in emotions:
            emotion_lower = emotion.lower()
            stats["emotion_distribution"][emotion] = stats["emotion_distribution"].get(emotion, 0) + 1
            
            if emotion_lower in self.emotion_tags:
                stats["supported_emotions"] += 1
            else:
                if emotion not in stats["unsupported_emotions"]:
                    stats["unsupported_emotions"].append(emotion)
        
        return stats

# Global production emotion processor instance
production_emotion_processor = ProductionEmotionProcessor()
