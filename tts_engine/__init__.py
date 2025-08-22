from .inference import (
    generate_speech_from_api,
    AVAILABLE_VOICES,
    DEFAULT_VOICE,
    VOICE_TO_LANGUAGE,
    AVAILABLE_LANGUAGES,
    list_available_voices
)

# Import production emotion processor with fallback
try:
    from .production_emotion_processor import production_emotion_processor
    print(" Production-grade emotion processing available")
except ImportError:
    try:
        from .emotion_processor import emotion_processor as production_emotion_processor
        print(" Enhanced emotion processing available (fallback)")
    except ImportError as e:
        print(f" Emotion processor not available: {e}")
        production_emotion_processor = None

try:
    __all__ = ['generate_speech_from_api', 'AVAILABLE_VOICES', 'DEFAULT_VOICE', 
               'VOICE_TO_LANGUAGE', 'AVAILABLE_LANGUAGES', 'list_available_voices', 
               'production_emotion_processor']
except ImportError:
    __all__ = ['generate_speech_from_api', 'AVAILABLE_VOICES', 'DEFAULT_VOICE', 
               'VOICE_TO_LANGUAGE', 'AVAILABLE_LANGUAGES', 'list_available_voices']
