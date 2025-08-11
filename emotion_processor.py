"""
Enhanced Emotion Processing System for Orpheus TTS
Provides intelligent emotion tag processing, context awareness, and intensity control
"""

import re
import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class EmotionConfig:
    """Configuration for emotion processing"""
    tag: str
    intensity_levels: List[str]
    context_words: List[str]
    duration_multiplier: float
    voice_compatibility: List[str]  # Which voices work best with this emotion

class EmotionProcessor:
    """Advanced emotion processing for more natural and accurate TTS output"""
    
    def __init__(self):
        self.emotion_configs = self._initialize_emotion_configs()
        self.context_patterns = self._initialize_context_patterns()
        
    def _initialize_emotion_configs(self) -> Dict[str, EmotionConfig]:
        """Initialize comprehensive emotion configurations"""
        return {
            "laugh": EmotionConfig(
                tag="laugh",
                intensity_levels=["<chuckle>", "<laugh>", "<hearty_laugh>"],
                context_words=["funny", "hilarious", "amusing", "joke", "comedy", "ridiculous"],
                duration_multiplier=1.2,
                voice_compatibility=["tara", "zoe", "mia", "jess", "dan"]
            ),
            "chuckle": EmotionConfig(
                tag="chuckle", 
                intensity_levels=["<soft_chuckle>", "<chuckle>", "<amused_chuckle>"],
                context_words=["amusing", "interesting", "clever", "witty", "ironic"],
                duration_multiplier=0.8,
                voice_compatibility=["tara", "leo", "dan", "zac"]
            ),
            "sigh": EmotionConfig(
                tag="sigh",
                intensity_levels=["<soft_sigh>", "<sigh>", "<deep_sigh>"],
                context_words=["tired", "exhausted", "disappointed", "frustrated", "weary", "unfortunately"],
                duration_multiplier=1.5,
                voice_compatibility=["tara", "leah", "mia", "zoe"]
            ),
            "gasp": EmotionConfig(
                tag="gasp",
                intensity_levels=["<soft_gasp>", "<gasp>", "<shocked_gasp>"],
                context_words=["surprising", "shocking", "amazing", "incredible", "unbelievable", "wow"],
                duration_multiplier=0.6,
                voice_compatibility=["zoe", "mia", "jess", "tara"]
            ),
            "groan": EmotionConfig(
                tag="groan",
                intensity_levels=["<soft_groan>", "<groan>", "<frustrated_groan>"],
                context_words=["annoying", "frustrating", "difficult", "problematic", "ugh", "terrible"],
                duration_multiplier=1.3,
                voice_compatibility=["dan", "leo", "zac", "thomas"]
            ),
            "yawn": EmotionConfig(
                tag="yawn",
                intensity_levels=["<tired_yawn>", "<yawn>", "<sleepy_yawn>"],
                context_words=["tired", "sleepy", "boring", "late", "exhausted", "drowsy"],
                duration_multiplier=1.8,
                voice_compatibility=["tara", "leah", "dan", "leo"]
            ),
            "cough": EmotionConfig(
                tag="cough",
                intensity_levels=["<soft_cough>", "<cough>", "<clearing_throat>"],
                context_words=["excuse me", "pardon", "ahem", "well", "actually"],
                duration_multiplier=0.5,
                voice_compatibility=["leo", "dan", "thomas", "zac"]
            ),
            "sniffle": EmotionConfig(
                tag="sniffle",
                intensity_levels=["<soft_sniffle>", "<sniffle>", "<emotional_sniffle>"],
                context_words=["sad", "emotional", "touching", "moving", "heartbreaking", "tears"],
                duration_multiplier=0.7,
                voice_compatibility=["tara", "mia", "leah", "amelie"]
            )
        }
    
    def _initialize_context_patterns(self) -> Dict[str, List[str]]:
        """Initialize context-based emotion enhancement patterns"""
        return {
            "excitement": ["amazing", "incredible", "fantastic", "wonderful", "awesome", "brilliant"],
            "sadness": ["unfortunately", "sadly", "regrettably", "disappointingly", "tragically"],
            "surprise": ["suddenly", "unexpectedly", "surprisingly", "shockingly", "amazingly"],
            "frustration": ["however", "but", "unfortunately", "annoyingly", "frustratingly"],
            "contentment": ["peacefully", "calmly", "serenely", "gently", "softly"],
            "urgency": ["quickly", "immediately", "urgently", "rapidly", "hastily"]
        }
    
    def analyze_text_context(self, text: str) -> Dict[str, float]:
        """Analyze text to determine emotional context and intensity"""
        text_lower = text.lower()
        context_scores = {}
        
        for context, words in self.context_patterns.items():
            score = sum(1 for word in words if word in text_lower)
            if score > 0:
                context_scores[context] = min(score / len(words), 1.0)
        
        return context_scores
    
    def enhance_emotion_tags(self, text: str, voice: str = "tara") -> str:
        """Enhance emotion tags based on context and voice compatibility"""
        enhanced_text = text
        context_scores = self.analyze_text_context(text)
        
        # Find all emotion tags in the text
        emotion_pattern = r'<(laugh|chuckle|sigh|gasp|groan|yawn|cough|sniffle)>'
        matches = re.finditer(emotion_pattern, text, re.IGNORECASE)
        
        replacements = []
        for match in matches:
            emotion_type = match.group(1).lower()
            if emotion_type in self.emotion_configs:
                config = self.emotion_configs[emotion_type]
                
                # Choose intensity based on context
                intensity_level = self._choose_intensity_level(config, context_scores)
                
                # Add voice-specific adjustments
                enhanced_tag = self._apply_voice_adjustments(intensity_level, voice, config)
                
                replacements.append((match.span(), enhanced_tag))
        
        # Apply replacements in reverse order to maintain positions
        for (start, end), replacement in reversed(replacements):
            enhanced_text = enhanced_text[:start] + replacement + enhanced_text[end:]
        
        return enhanced_text
    
    def _choose_intensity_level(self, config: EmotionConfig, context_scores: Dict[str, float]) -> str:
        """Choose appropriate intensity level based on context"""
        base_intensity = 1  # Default to middle intensity
        
        # Adjust intensity based on context
        if "excitement" in context_scores:
            base_intensity += context_scores["excitement"]
        if "urgency" in context_scores:
            base_intensity += context_scores["urgency"] * 0.5
        if "sadness" in context_scores and config.tag in ["sigh", "sniffle"]:
            base_intensity += context_scores["sadness"]
        if "surprise" in context_scores and config.tag == "gasp":
            base_intensity += context_scores["surprise"]
        
        # Clamp to valid range
        intensity_index = max(0, min(len(config.intensity_levels) - 1, int(base_intensity)))
        return config.intensity_levels[intensity_index]
    
    def _apply_voice_adjustments(self, emotion_tag: str, voice: str, config: EmotionConfig) -> str:
        """Apply voice-specific adjustments to emotion tags"""
        # If voice is not compatible, use a softer version
        if voice not in config.voice_compatibility:
            # Use the softest intensity level for incompatible voices
            return config.intensity_levels[0]
        
        return emotion_tag
    
    def add_contextual_emotions(self, text: str, voice: str = "tara") -> str:
        """Intelligently add emotion tags based on text content"""
        enhanced_text = text
        context_scores = self.analyze_text_context(text)
        
        # Add emotions based on context (but don't overdo it)
        sentences = re.split(r'[.!?]+', text)
        enhanced_sentences = []
        
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            sentence_enhanced = sentence.strip()
            sentence_lower = sentence.lower()
            
            # Add contextual emotions sparingly (max 1 per sentence)
            emotion_added = False
            
            # Check for surprise context
            if not emotion_added and "surprise" in context_scores and context_scores["surprise"] > 0.3:
                if any(word in sentence_lower for word in ["amazing", "incredible", "wow", "unbelievable"]):
                    sentence_enhanced = sentence_enhanced + " <gasp>"
                    emotion_added = True
            
            # Check for amusement context
            if not emotion_added and any(word in sentence_lower for word in ["funny", "hilarious", "amusing"]):
                if random.random() < 0.4:  # 40% chance to add laugh
                    sentence_enhanced = sentence_enhanced + " <chuckle>"
                    emotion_added = True
            
            # Check for frustration context
            if not emotion_added and "frustration" in context_scores and context_scores["frustration"] > 0.2:
                if any(word in sentence_lower for word in ["unfortunately", "however", "but"]):
                    sentence_enhanced = sentence_enhanced + " <sigh>"
                    emotion_added = True
            
            enhanced_sentences.append(sentence_enhanced)
        
        return ". ".join(enhanced_sentences) + "." if enhanced_sentences else text
    
    def optimize_emotion_placement(self, text: str) -> str:
        """Optimize the placement of emotion tags for better flow"""
        # Move emotions to more natural positions
        # Before punctuation is often more natural than after
        
        # Pattern: word <emotion> punctuation -> word punctuation <emotion>
        text = re.sub(r'(\w+)\s*(<[^>]+>)\s*([.!?])', r'\1\3 \2', text)
        
        # Pattern: <emotion> at start of sentence -> move after first few words
        text = re.sub(r'^(<[^>]+>)\s*(\w+\s+\w+)', r'\2 \1', text, flags=re.MULTILINE)
        
        return text
    
    def process_text(self, text: str, voice: str = "tara", add_contextual: bool = True) -> str:
        """Main processing function - enhance existing emotions and optionally add contextual ones"""
        processed_text = text
        
        # Step 1: Enhance existing emotion tags
        processed_text = self.enhance_emotion_tags(processed_text, voice)
        
        # Step 2: Add contextual emotions if requested
        if add_contextual:
            processed_text = self.add_contextual_emotions(processed_text, voice)
        
        # Step 3: Optimize emotion placement
        processed_text = self.optimize_emotion_placement(processed_text)
        
        return processed_text
    
    def get_emotion_stats(self, text: str) -> Dict[str, int]:
        """Get statistics about emotions in the text"""
        emotion_pattern = r'<([^>]+)>'
        emotions = re.findall(emotion_pattern, text, re.IGNORECASE)
        
        stats = {}
        for emotion in emotions:
            stats[emotion] = stats.get(emotion, 0) + 1
        
        return stats

# Global emotion processor instance
emotion_processor = EmotionProcessor()
