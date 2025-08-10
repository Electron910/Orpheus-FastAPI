#!/usr/bin/env python3
"""
FIXED Hybrid TTS Application - Uses correct model name
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import os
import struct
import asyncio
import multiprocessing
from typing import Optional, Generator
from dotenv import load_dotenv
import uvicorn
from contextlib import asynccontextmanager

# Global model instance
orpheus_engine = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await initialize_orpheus_model()
    yield
    # Shutdown
    pass

app = FastAPI(title="Orpheus TTS Hybrid API", version="2.0.0", lifespan=lifespan)

class TTSRequest(BaseModel):
    prompt: str
    voice: Optional[str] = "tara"
    temperature: Optional[float] = 0.4
    top_p: Optional[float] = 0.9
    max_tokens: Optional[int] = 2000
    repetition_penalty: Optional[float] = 1.1

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV file header for streaming audio"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b'data', data_size
    )
    return header

async def initialize_orpheus_model():
    """Initialize the Orpheus TTS model asynchronously"""
    global orpheus_engine
    
    if orpheus_engine is not None:
        return True
        
    try:
        print("🎤 Initializing Original Orpheus TTS Model...")
        from orpheus_tts import OrpheusModel
        
        # Use the CORRECT model name
        model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        orpheus_engine = await loop.run_in_executor(
            None, 
            lambda: OrpheusModel(model_name=model_name)
        )
        
        print("✅ Orpheus TTS Model initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize Orpheus TTS model: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API documentation"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Orpheus TTS API - WORKING</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f0f8ff; }}
            .status {{ background: #d4edda; padding: 15px; border-radius: 5px; margin: 10px 0; }}
            .endpoint {{ background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }}
            .method {{ color: #007acc; font-weight: bold; }}
            .test-link {{ background: #007acc; color: white; padding: 10px 20px; text-decoration: none; border-radius: 5px; display: inline-block; margin: 5px; }}
        </style>
    </head>
    <body>
        <h1>🎤 Orpheus TTS API - WORKING!</h1>
        
        <div class="status">
            <h3>✅ Status: OPERATIONAL</h3>
            <p><strong>Model:</strong> canopylabs/orpheus-tts-0.1-finetune-prod</p>
            <p><strong>Your URL:</strong> https://561pjq4x4ud1px-5005.proxy.runpod.net/</p>
        </div>
        
        <h2>🎵 Quick Tests:</h2>
        <a href="/tts?prompt=Hello%20world,%20this%20is%20working%20Orpheus%20TTS" class="test-link">Test TTS</a>
        <a href="/health" class="test-link">Health Check</a>
        
        <h2>📋 API Endpoints:</h2>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/tts</strong><br>
            Generate TTS audio with query parameters<br>
            Example: <code>/tts?prompt=Hello%20world&voice=tara</code>
        </div>
        
        <div class="endpoint">
            <span class="method">POST</span> <strong>/generate_speech</strong><br>
            Generate TTS audio with JSON payload<br>
            Body: <code>{{"prompt": "text", "voice": "tara"}}</code>
        </div>
        
        <div class="endpoint">
            <span class="method">GET</span> <strong>/health</strong><br>
            Health check and model status
        </div>
        
        <h2>🔗 Integration:</h2>
        <p>Use this URL in your Vocalis frontend:</p>
        <code>https://561pjq4x4ud1px-5005.proxy.runpod.net/tts</code>
    </body>
    </html>
    """

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if orpheus_engine is None:
        return {"status": "error", "message": "Orpheus TTS model not initialized"}
    return {
        "status": "ok", 
        "model": "canopylabs/orpheus-tts-0.1-finetune-prod",
        "engine": "original-orpheus",
        "version": "2.0.0",
        "url": "https://561pjq4x4ud1px-5005.proxy.runpod.net/"
    }

async def generate_audio_stream(prompt: str, voice: str = "tara", temperature: float = 0.4, 
                               top_p: float = 0.9, max_tokens: int = 2000, 
                               repetition_penalty: float = 1.1) -> Generator[bytes, None, None]:
    """Generate streaming audio from text"""
    if orpheus_engine is None:
        raise HTTPException(status_code=500, detail="Orpheus TTS model not initialized")
    
    try:
        print(f"🗣️ Generating TTS for: '{prompt[:50]}...' with voice: {voice}")
        
        # Send WAV header first
        yield create_wav_header()
        
        # Generate speech tokens using original Orpheus
        loop = asyncio.get_event_loop()
        syn_tokens = await loop.run_in_executor(
            None,
            lambda: orpheus_engine.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p
            )
        )
        
        chunk_count = 0
        for chunk in syn_tokens:
            if chunk is not None and len(chunk) > 0:
                chunk_count += 1
                print(f"📦 Streaming chunk {chunk_count}, size: {len(chunk)} bytes")
                yield chunk
                
        print(f"✅ Finished streaming {chunk_count} audio chunks")
        
    except Exception as e:
        print(f"❌ Error during audio generation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/generate_speech")
async def generate_speech_endpoint(request: TTSRequest):
    """Main TTS generation endpoint (POST)"""
    return StreamingResponse(
        generate_audio_stream(
            prompt=request.prompt,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            repetition_penalty=request.repetition_penalty
        ),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )

@app.get("/tts")
async def tts_get_endpoint(
    prompt: str,
    voice: str = "tara",
    temperature: float = 0.4,
    top_p: float = 0.9,
    max_tokens: int = 2000,
    repetition_penalty: float = 1.1
):
    """TTS generation endpoint (GET) for easy testing"""
    return StreamingResponse(
        generate_audio_stream(
            prompt=prompt,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            repetition_penalty=repetition_penalty
        ),
        media_type="audio/wav",
        headers={"Content-Disposition": "attachment; filename=speech.wav"}
    )

def main():
    """Main function with proper multiprocessing setup"""
    # Set multiprocessing method for VLLM compatibility
    multiprocessing.set_start_method('spawn', force=True)
    
    host = "0.0.0.0"
    port = 5005
    
    print(f"🚀 Starting FIXED Orpheus TTS Server on {host}:{port}")
    print("🔧 Using CORRECT model: canopylabs/orpheus-tts-0.1-finetune-prod")
    print("🌐 Your URL: https://561pjq4x4ud1px-5005.proxy.runpod.net/")
    
    # Run with uvicorn
    uvicorn.run(
        "fixed_hybrid_app:app",
        host=host,
        port=port,
        reload=False,
        workers=1
    )

if __name__ == '__main__':
    main()
