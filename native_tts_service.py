"""
Native TTS Service for Orpheus-FastAPI
Loads and runs the Orpheus TTS model directly, similar to how Vocalis loads Whisper/SmolVLM
"""

import logging
import time
import wave
import os
from typing import Optional, Generator
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NativeTTSService:
    """
    Native TTS service that loads the Orpheus model directly.
    Similar to how Vocalis loads Whisper and SmolVLM models.
    """
    
    def __init__(self):
        """Initialize the service with empty model references."""
        self.model = None
        self.initialized = False
        self.fallback_mode = False
        self.model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
        self.device = None
        
    def initialize(self):
        """
        Initialize the Orpheus TTS model, downloading it if necessary.
        This will be called on server startup.
        
        Returns:
            bool: Whether initialization was successful
        """
        if self.initialized:
            logger.info("Orpheus TTS model already initialized")
            return True
        
        try:
            # Determine device (use CUDA if available)
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            logger.info(f"Using device for Orpheus TTS model: {self.device}")
            
            logger.info(f"Loading Orpheus TTS model {self.model_name} (this may take a while on first run)...")
            
            # Import and initialize the Orpheus model
            from orpheus_tts import OrpheusModel
            
            # This call will trigger the download if the model isn't cached locally
            self.model = OrpheusModel(
                model_name=self.model_name,
                max_model_len=2048
            )
            
            self.initialized = True
            logger.info("Orpheus TTS model loaded successfully")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to import orpheus-speech package: {e}")
            logger.error("Please install with: pip install orpheus-speech")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Orpheus TTS model: {e}")
            return False
    
    def generate_speech(self, text: str, voice: str = "tara", output_file: Optional[str] = None) -> bool:
        """
        Generate speech from text using the native Orpheus model or fallback to API.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (default: tara)
            output_file: Path to save the audio file
            
        Returns:
            bool: Whether generation was successful
        """
        # If in fallback mode, use the original API-based approach
        if self.fallback_mode:
            logger.info("Using fallback API-based TTS generation")
            return self._generate_speech_fallback(text, voice, output_file)
        
        if not self.initialized:
            logger.error("Orpheus TTS model not initialized")
            return False
        
        try:
            start_time = time.time()
            logger.info(f"Generating speech for: {text[:50]}{'...' if len(text) > 50 else ''}")
            logger.info(f"Using voice: {voice}")
            
            # Generate speech tokens using the native model
            syn_tokens = self.model.generate_speech(
                prompt=text,
                voice=voice
            )
            
            # Write audio to file
            if output_file:
                with wave.open(output_file, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(24000)
                    
                    total_frames = 0
                    chunk_counter = 0
                    
                    for audio_chunk in syn_tokens:
                        chunk_counter += 1
                        frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                        total_frames += frame_count
                        wf.writeframes(audio_chunk)
                    
                    duration = total_frames / wf.getframerate()
                    
                end_time = time.time()
                generation_time = end_time - start_time
                
                logger.info(f"Generated {duration:.2f}s of audio in {generation_time:.2f}s")
                logger.info(f"Audio saved to {output_file}")
                
                return True
            else:
                logger.error("No output file specified")
                return False
                
        except Exception as e:
            logger.error(f"Speech generation failed: {e}")
            return False
    
    def _generate_speech_fallback(self, text: str, voice: str = "tara", output_file: Optional[str] = None) -> bool:
        """
        Fallback method using the original API-based TTS generation.
        This imports and uses the existing tts_engine when native model fails.
        """
        try:
            logger.info(f"Fallback TTS generation for: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # Import the original TTS engine
            from tts_engine import generate_speech_from_api
            
            # Use the original API-based generation
            generate_speech_from_api(
                prompt=text,
                voice=voice,
                output_file=output_file,
                use_batching=len(text) > 1000,
                max_batch_chars=1000
            )
            
            logger.info(f"Fallback TTS generation completed, saved to: {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Fallback TTS generation failed: {e}")
            return False

# Global service instance
native_tts_service = NativeTTSService()
