"""
TTS Engine package for Orpheus text-to-speech system.

This package contains the core components for audio generation:
- inference.py: Token generation and API handling
- speechpipe.py: Audio conversion pipeline with streaming support
"""

# Make key components available at package level
from .inference import (
    generate_speech_from_api,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES,
    list_available_voices,
    generate_tokens_from_api  # Add this for direct access
)

# Make streaming functions available
from .speechpipe import (
    generate_audio_stream,  # NEW: For streaming audio generation
    convert_to_audio,
    turn_token_into_id,
    create_wav_header  # NEW: For streaming WAV headers
)

# Import production emotion processor with fallback
try:
    from .production_emotion_processor import production_emotion_processor
    print("✅ Production-grade emotion processing available")
except ImportError:
    try:
        from .emotion_processor import emotion_processor as production_emotion_processor
        print("✅ Enhanced emotion processing available (fallback)")
    except ImportError as e:
        print(f"⚠️ Emotion processor not available: {e}")
        production_emotion_processor = None

__all__ = [
    'generate_speech_from_api', 
    'AVAILABLE_VOICES', 
    'DEFAULT_VOICE', 
    'VOICE_TO_LANGUAGE', 
    'AVAILABLE_LANGUAGES', 
    'list_available_voices',
    'generate_tokens_from_api',
    'generate_audio_stream',
    'convert_to_audio',
    'turn_token_into_id',
    'create_wav_header'
]

if production_emotion_processor is not None:
    __all__.append('production_emotion_processor')
