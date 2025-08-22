from snac import SNAC
import numpy as np
import torch
import asyncio
import threading
import queue
import time
import os
import sys
import io
import wave
import struct

# Helper to detect if running in Uvicorn's reloader (same as in inference.py)
def is_reloader_process():
    """Check if the current process is a uvicorn reloader"""
    return (sys.argv[0].endswith('_continuation.py') or 
            os.environ.get('UVICORN_STARTED') == 'true')

# Set a flag to avoid repeat messages
IS_RELOADER = is_reloader_process()

# Try to enable torch.compile if PyTorch 2.0+ is available
TORCH_COMPILE_AVAILABLE = False
try:
    if hasattr(torch, 'compile'):
        TORCH_COMPILE_AVAILABLE = True
        if not IS_RELOADER:
            print("PyTorch 2.0+ detected, torch.compile is available")
except:
    pass

# Try to enable CUDA graphs if available
CUDA_GRAPHS_AVAILABLE = False
try:
    if torch.cuda.is_available() and hasattr(torch.cuda, 'make_graphed_callables'):
        CUDA_GRAPHS_AVAILABLE = True
        if not IS_RELOADER:
            print("CUDA graphs support is available")
except:
    pass

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

# Check if CUDA is available and set device accordingly
snac_device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
if not IS_RELOADER:
    print(f"Using device: {snac_device}")
model = model.to(snac_device)

# Disable torch.compile as it requires Triton which isn't installed
# We'll use regular PyTorch optimization techniques instead
if not IS_RELOADER:
    print("Using standard PyTorch optimizations (torch.compile disabled)")

# Prepare CUDA streams for parallel processing if available
cuda_stream = None
if snac_device == "cuda":
    cuda_stream = torch.cuda.Stream()
    if not IS_RELOADER:
        print("Using CUDA stream for parallel processing")

def convert_to_audio(multiframe, count):
    """
    Optimized version of convert_to_audio that eliminates inefficient tensor operations
    and reduces CPU-GPU transfers for much faster inference on high-end GPUs.
    """
    if len(multiframe) < 7:
        return None
  
    num_frames = len(multiframe) // 7
    frame = multiframe[:num_frames*7]
    
    # Pre-allocate tensors instead of incrementally building them
    codes_0 = torch.zeros(num_frames, dtype=torch.int32, device=snac_device)
    codes_1 = torch.zeros(num_frames * 2, dtype=torch.int32, device=snac_device)
    codes_2 = torch.zeros(num_frames * 4, dtype=torch.int32, device=snac_device)
    
    # Use vectorized operations where possible
    frame_tensor = torch.tensor(frame, dtype=torch.int32, device=snac_device)
    
    # Direct indexing is much faster than concatenation in a loop
    for j in range(num_frames):
        idx = j * 7
        
        # Code 0 - single value per frame
        codes_0[j] = frame_tensor[idx]
        
        # Code 1 - two values per frame
        codes_1[j*2] = frame_tensor[idx+1]
        codes_1[j*2+1] = frame_tensor[idx+4]
        
        # Code 2 - four values per frame
        codes_2[j*4] = frame_tensor[idx+2]
        codes_2[j*4+1] = frame_tensor[idx+3]
        codes_2[j*4+2] = frame_tensor[idx+5]
        codes_2[j*4+3] = frame_tensor[idx+6]
    
    # Reshape codes into expected format
    codes = [
        codes_0.unsqueeze(0), 
        codes_1.unsqueeze(0), 
        codes_2.unsqueeze(0)
    ]
    
    # Check tokens are in valid range
    if (torch.any(codes[0] < 0) or torch.any(codes[0] > 4096) or 
        torch.any(codes[1] < 0) or torch.any(codes[1] > 4096) or 
        torch.any(codes[2] < 0) or torch.any(codes[2] > 4096)):
        return None

    # Use CUDA stream for parallel processing if available
    stream_ctx = torch.cuda.stream(cuda_stream) if cuda_stream is not None else torch.no_grad()
    
    with stream_ctx, torch.inference_mode():
        # Decode the audio
        audio_hat = model.decode(codes)
        
        # Extract the relevant slice and efficiently convert to bytes
        # Keep data on GPU as long as possible
        audio_slice = audio_hat[:, :, 2048:4096]
        
        # Process on GPU if possible, with minimal data transfer
        if snac_device == "cuda":
            # Scale directly on GPU
            audio_int16_tensor = (audio_slice * 32767).to(torch.int16)
            # Only transfer the final result to CPU
            audio_bytes = audio_int16_tensor.cpu().numpy().tobytes()
        else:
            # For non-CUDA devices, fall back to the original approach
            detached_audio = audio_slice.detach().cpu()
            audio_np = detached_audio.numpy()
            audio_int16 = (audio_np * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
    return audio_bytes

# Define the custom token prefix
CUSTOM_TOKEN_PREFIX = "<custom_token_"

# Use a single global cache for token processing
token_id_cache = {}
MAX_CACHE_SIZE = 10000  # Increased cache size for better performance

def turn_token_into_id(token_string, index):
    """
    Optimized token-to-ID conversion with caching.
    This is the definitive implementation used by both inference.py and speechpipe.py.
    
    Args:
        token_string: The token string to convert
        index: Position index used for token offset calculation
        
    Returns:
        int: Token ID if valid, None otherwise
    """
    # Check cache first (significant speedup for repeated tokens)
    cache_key = (token_string, index % 7)
    if cache_key in token_id_cache:
        return token_id_cache[cache_key]
        
    # Early rejection for obvious non-matches
    if CUSTOM_TOKEN_PREFIX not in token_string:
        return None
        
    # Process token
    token_string = token_string.strip()
    last_token_start = token_string.rfind(CUSTOM_TOKEN_PREFIX)
    
    if last_token_start == -1:
        return None
    
    last_token = token_string[last_token_start:]
    
    if not (last_token.startswith(CUSTOM_TOKEN_PREFIX) and last_token.endswith(">")):
        return None
        
    try:
        number_str = last_token[14:-1]
        token_id = int(number_str) - 10 - ((index % 7) * 4096)
        
        # Cache the result if it's valid
        if len(token_id_cache) < MAX_CACHE_SIZE:
            token_id_cache[cache_key] = token_id
            
        return token_id
    except (ValueError, IndexError):
        return None

def create_wav_header(sample_rate=24000, channels=1, bits_per_sample=16):
    """Create a WAV file header for streaming"""
    # We'll set data length to maximum initially, then correct it if needed
    data_length = 0xFFFFFFFF - 36  # Maximum size minus header
    
    header = bytearray()
    
    # RIFF header
    header.extend(b'RIFF')
    header.extend(struct.pack('<I', data_length + 36))  # File size - 8
    header.extend(b'WAVE')
    
    # fmt subchunk
    header.extend(b'fmt ')
    header.extend(struct.pack('<I', 16))  # Subchunk size
    header.extend(struct.pack('<H', 1))   # Audio format (PCM)
    header.extend(struct.pack('<H', channels))
    header.extend(struct.pack('<I', sample_rate))
    header.extend(struct.pack('<I', sample_rate * channels * bits_per_sample // 8))  # Byte rate
    header.extend(struct.pack('<H', channels * bits_per_sample // 8))  # Block align
    header.extend(struct.pack('<H', bits_per_sample))
    
    # data subchunk header
    header.extend(b'data')
    header.extend(struct.pack('<I', data_length))
    
    return bytes(header)

async def generate_audio_stream(token_gen):
    """
    NEW STREAMING FUNCTION: Generate audio chunks as they are ready
    This replaces the blocking tokens_decoder approach
    """
    buffer = []
    count = 0
    
    # Track if first chunk has been processed  
    first_chunk_processed = False
    
    # Use different thresholds for first chunk vs. subsequent chunks
    min_frames_first = 7  # Just one chunk (7 tokens) for first audio
    min_frames_subsequent = 28  # Standard minimum after first audio
    process_every_n = 7  # Process every 7 tokens
    
    # Yield WAV header first
    wav_header = create_wav_header()
    yield wav_header
    print("Sent WAV header for streaming")
    
    start_time = time.time()
    token_count = 0
    last_log_time = start_time
    
    async for token_sim in token_gen:
        token_count += 1
        
        # Use the unified turn_token_into_id
        token = turn_token_into_id(token_sim, count)
        
        if token is not None and token > 0:
            buffer.append(token)
            count += 1
            
            # Log throughput periodically
            current_time = time.time()
            if current_time - last_log_time > 5.0:  # Every 5 seconds
                elapsed = current_time - last_log_time
                if elapsed > 0:
                    recent_tokens = token_count
                    tokens_per_sec = recent_tokens / elapsed
                    print(f"Token processing rate: {tokens_per_sec:.1f} tokens/second")
                last_log_time = current_time
                token_count = 0
            
            # Different processing logic based on whether first chunk has been processed
            if not first_chunk_processed:
                # Process first chunk as soon as possible for minimal latency
                if count >= min_frames_first:
                    buffer_to_proc = buffer[-min_frames_first:]
                    
                    print(f"Processing first audio chunk with {len(buffer_to_proc)} tokens for low latency")
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        first_chunk_processed = True
                        yield audio_samples
            else:
                # For subsequent chunks, use standard processing with proper batching
                if count % process_every_n == 0 and count >= min_frames_subsequent:
                    # Use standard processing logic  
                    if len(buffer) >= 49:  # Ideal frames
                        buffer_to_proc = buffer[-49:]
                    elif len(buffer) >= min_frames_subsequent:
                        buffer_to_proc = buffer[-min_frames_subsequent:]
                    else:
                        continue
                    
                    # Debug output
                    if count % 28 == 0:
                        print(f"Processing buffer with {len(buffer_to_proc)} tokens, total collected: {len(buffer)}")
                    
                    # Process the tokens
                    audio_samples = convert_to_audio(buffer_to_proc, count)
                    if audio_samples is not None:
                        yield audio_samples
    
    # Process remaining complete frames at the end
    if len(buffer) >= 49:
        buffer_to_proc = buffer[-49:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
    elif len(buffer) >= min_frames_subsequent:
        buffer_to_proc = buffer[-min_frames_subsequent:]
        audio_samples = convert_to_audio(buffer_to_proc, count)
        if audio_samples is not None:
            yield audio_samples
    elif len(buffer) >= process_every_n:
        # Pad final partial frame
        last_token = buffer[-1]
        padding_needed = min_frames_subsequent - len(buffer)
        padding = [last_token] * padding_needed
        padded_buffer = buffer + padding
        
        print(f"Processing final partial frame: {len(buffer)} tokens + {padding_needed} padding")
        audio_samples = convert_to_audio(padded_buffer, count)
        if audio_samples is not None:
            yield audio_samples

# Legacy compatibility functions
async def tokens_decoder(token_gen):
    """Legacy compatibility - redirects to streaming function"""
    async for chunk in generate_audio_stream(token_gen):
        if len(chunk) > 44:  # Skip WAV header for legacy compatibility
            yield chunk[44:] if chunk.startswith(b'RIFF') else chunk

def tokens_decoder_sync(syn_token_gen, output_file=None):
    """
    Optimized synchronous decoder with optional file output
    Now supports streaming while optionally saving to file
    """
    # Convert sync generator to async
    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    # Setup file writing if requested
    wav_file = None
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        wav_file = wave.open(output_file, "wb")
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(24000)
    
    audio_segments = []
    
    async def process_audio():
        header_written = False
        async for audio_chunk in generate_audio_stream(async_token_gen()):
            if not header_written and audio_chunk.startswith(b'RIFF'):
                # Skip header for return value but keep for file
                header_written = True
                if wav_file:
                    # Write just the audio data part to wav file
                    continue
            else:
                # This is audio data
                audio_segments.append(audio_chunk)
                if wav_file:
                    wav_file.writeframes(audio_chunk)
    
    # Run the async processing
    asyncio.run(process_audio())
    
    # Close file if opened
    if wav_file:
        wav_file.close()
        print(f"Audio saved to {output_file}")
    
    return audio_segments
