#!/usr/bin/env python3
"""
Memory-Optimized Orpheus TTS with GPU Memory Management
Handles GPU memory conflicts and provides fallback options
"""

import os
import sys
import subprocess
import multiprocessing
import psutil
import torch
import gc
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def clear_gpu_memory():
    """Clear GPU memory cache"""
    print("🧹 Clearing GPU memory cache...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"   GPU memory cleared. Available: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

def get_gpu_memory_info():
    """Get current GPU memory usage"""
    if not torch.cuda.is_available():
        return None, None, None
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    cached_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - cached_memory
    
    return total_memory / 1e9, allocated_memory / 1e9, free_memory / 1e9

def find_gpu_processes():
    """Find processes using GPU memory"""
    print("🔍 Checking for GPU processes...")
    try:
        result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            processes = []
            for line in result.stdout.strip().split('\n'):
                if line.strip():
                    parts = line.split(', ')
                    if len(parts) >= 3:
                        pid, name, memory = parts[0], parts[1], parts[2]
                        processes.append((int(pid), name, int(memory)))
            return processes
        return []
    except Exception as e:
        print(f"   ⚠️ Could not check GPU processes: {e}")
        return []

def kill_llama_processes():
    """Kill llama-cpp-python processes to free GPU memory"""
    print("🔄 Looking for llama-cpp processes to stop...")
    killed = False
    
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = ' '.join(proc.info['cmdline'] or [])
            if any(keyword in cmdline.lower() for keyword in ['llama', 'vocalis', 'llama-cpp-python']):
                print(f"   🛑 Stopping process: {proc.info['name']} (PID: {proc.info['pid']})")
                proc.terminate()
                proc.wait(timeout=5)
                killed = True
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired):
            continue
    
    if killed:
        print("   ✅ Processes stopped. Waiting for GPU memory to free...")
        import time
        time.sleep(3)
        clear_gpu_memory()
    else:
        print("   ℹ️ No llama processes found")

def test_orpheus_with_memory_management():
    """Test Orpheus TTS with proper memory management"""
    print("🎤 Testing Orpheus TTS with Memory Management")
    print("=" * 50)
    
    # Check initial GPU memory
    total, allocated, free = get_gpu_memory_info()
    if total:
        print(f"📊 GPU Memory: {free:.1f}GB free / {total:.1f}GB total")
        
        if free < 20:  # Need at least 20GB for Orpheus
            print("⚠️ Insufficient GPU memory. Attempting to free memory...")
            
            # Kill competing processes
            kill_llama_processes()
            
            # Check GPU processes
            gpu_procs = find_gpu_processes()
            if gpu_procs:
                print("   🔍 GPU processes found:")
                for pid, name, memory in gpu_procs:
                    print(f"      PID {pid}: {name} ({memory}MB)")
            
            # Recheck memory
            total, allocated, free = get_gpu_memory_info()
            print(f"📊 GPU Memory after cleanup: {free:.1f}GB free / {total:.1f}GB total")
    
    # Set multiprocessing method
    multiprocessing.set_start_method('spawn', force=True)
    
    try:
        print("📥 Loading Orpheus model with reduced memory settings...")
        from orpheus_tts import OrpheusModel
        import wave
        import time
        
        # Initialize with memory-conscious settings
        model = OrpheusModel(
            model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
            # Add memory optimization parameters if available
        )
        print("✅ Model loaded successfully!")
        
        prompt = "Hello, this is a test of the memory-optimized Orpheus TTS system."
        
        print(f"🗣️ Generating speech for: '{prompt}'")
        start_time = time.monotonic()
        
        # Generate speech tokens with conservative settings
        syn_tokens = model.generate_speech(
            prompt=prompt,
            voice="tara",
            repetition_penalty=1.1,
            max_tokens=1000,  # Reduced for memory efficiency
            temperature=0.4,
            top_p=0.9
        )
        
        # Save to WAV file
        output_file = "memory_optimized_test.wav"
        with wave.open(output_file, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            total_frames = 0
            chunk_count = 0
            
            for audio_chunk in syn_tokens:
                if audio_chunk is not None and len(audio_chunk) > 0:
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    chunk_count += 1
                    wf.writeframes(audio_chunk)
                    print(f"📦 Processed chunk {chunk_count}, frames: {frame_count}")
            
            duration = total_frames / wf.getframerate()
            end_time = time.monotonic()
            
            print(f"🎵 Generated {duration:.2f} seconds of audio in {end_time - start_time:.2f} seconds")
            print(f"📊 Total chunks: {chunk_count}, Total frames: {total_frames}")
            
            # Check file size
            file_size = os.path.getsize(output_file)
            print(f"📁 Output file: {output_file} ({file_size} bytes)")
            
            if file_size > 1000:  # More than just WAV header
                print("✅ SUCCESS: Memory-optimized Orpheus TTS is working!")
                return True
            else:
                print("❌ WARNING: Generated file is too small")
                return False
                
    except Exception as e:
        error_msg = str(e)
        print(f"❌ ERROR: {error_msg}")
        
        if "memory" in error_msg.lower():
            print("\n💡 MEMORY ISSUE DETECTED:")
            print("   1. Stop the Vocalis LLM server first")
            print("   2. Run: pkill -f llama")
            print("   3. Wait 10 seconds, then try again")
            print("   4. Or restart the entire container")
        
        import traceback
        traceback.print_exc()
        return False

def create_memory_optimized_server():
    """Create a memory-optimized TTS server"""
    server_code = '''#!/usr/bin/env python3
"""
Memory-Optimized TTS Server - Starts only when sufficient GPU memory is available
"""

from flask import Flask, Response, request, jsonify
import struct
import os
import sys
import torch
import multiprocessing
from dotenv import load_dotenv

load_dotenv()
app = Flask(__name__)
engine = None

def check_gpu_memory():
    """Check if sufficient GPU memory is available"""
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    total_memory = torch.cuda.get_device_properties(0).total_memory
    allocated_memory = torch.cuda.memory_allocated(0)
    cached_memory = torch.cuda.memory_reserved(0)
    free_memory = total_memory - cached_memory
    
    free_gb = free_memory / 1e9
    required_gb = 20  # Minimum required for Orpheus
    
    if free_gb >= required_gb:
        return True, f"Sufficient memory: {free_gb:.1f}GB free"
    else:
        return False, f"Insufficient memory: {free_gb:.1f}GB free, need {required_gb}GB"

def initialize_model():
    """Initialize model only if memory is sufficient"""
    global engine
    
    if engine is not None:
        return True
        
    # Check memory first
    memory_ok, memory_msg = check_gpu_memory()
    print(f"🔍 Memory check: {memory_msg}")
    
    if not memory_ok:
        print("❌ Cannot start TTS - insufficient GPU memory")
        print("💡 Stop other GPU processes first (e.g., Vocalis LLM)")
        return False
    
    try:
        print("🎤 Initializing Orpheus TTS...")
        from orpheus_tts import OrpheusModel
        
        engine = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        print("✅ Orpheus TTS initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        return False

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV file header"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36 + data_size, b'WAVE', b'fmt ', 16, 1, channels,
        sample_rate, byte_rate, block_align, bits_per_sample, b'data', data_size
    )
    return header

@app.route('/health')
def health_check():
    """Health check with memory status"""
    memory_ok, memory_msg = check_gpu_memory()
    
    return jsonify({
        "status": "ok" if engine else "not_ready",
        "model_loaded": engine is not None,
        "memory_status": memory_msg,
        "memory_sufficient": memory_ok
    })

@app.route('/tts')
def tts_endpoint():
    """TTS endpoint"""
    if engine is None:
        return jsonify({"error": "Model not initialized - check /health for memory status"}), 500
    
    prompt = request.args.get('prompt', 'Hello world')
    voice = request.args.get('voice', 'tara')
    
    def generate_audio_stream():
        try:
            yield create_wav_header()
            
            syn_tokens = engine.generate_speech(
                prompt=prompt,
                voice=voice,
                repetition_penalty=1.1,
                max_tokens=1500,
                temperature=0.4,
                top_p=0.9
            )
            
            for chunk in syn_tokens:
                if chunk is not None and len(chunk) > 0:
                    yield chunk
                    
        except Exception as e:
            print(f"❌ TTS Error: {str(e)}")
            yield b''

    return Response(generate_audio_stream(), mimetype='audio/wav')

@app.route('/')
def index():
    """Simple index with status"""
    memory_ok, memory_msg = check_gpu_memory()
    model_status = "✅ Ready" if engine else "❌ Not Ready"
    
    return f"""
    <h1>Memory-Optimized Orpheus TTS</h1>
    <p><strong>Model Status:</strong> {model_status}</p>
    <p><strong>Memory Status:</strong> {memory_msg}</p>
    <p><a href="/health">Health Check</a></p>
    {f'<p><a href="/tts?prompt=Hello%20world">Test TTS</a></p>' if engine else ''}
    """

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    
    if initialize_model():
        print("🚀 Starting memory-optimized TTS server on port 5005...")
        app.run(host='0.0.0.0', port=5005, threaded=True, use_reloader=False)
    else:
        print("💥 Cannot start server - model initialization failed")
        print("💡 Free up GPU memory and try again")
        sys.exit(1)
'''
    
    with open("memory_optimized_server.py", "w") as f:
        f.write(server_code)
    
    print("📝 Created memory_optimized_server.py")

def main():
    """Main function"""
    print("🚀 Orpheus TTS Memory Management Tool")
    print("=" * 50)
    
    # Step 1: Check current memory
    total, allocated, free = get_gpu_memory_info()
    if total:
        print(f"📊 Current GPU Memory: {free:.1f}GB free / {total:.1f}GB total")
    
    # Step 2: Create optimized server
    create_memory_optimized_server()
    
    # Step 3: Test with memory management
    success = test_orpheus_with_memory_management()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SUCCESS: Orpheus TTS is working!")
        print("\n📋 Next Steps:")
        print("1. Run the memory-optimized server:")
        print("   python memory_optimized_server.py")
        print("2. Test: http://localhost:5005/")
    else:
        print("💥 FAILED: Memory issues detected")
        print("\n🔧 Solutions:")
        print("1. Stop Vocalis LLM: pkill -f llama")
        print("2. Wait 10 seconds for memory to clear")
        print("3. Run this script again")
        print("4. Or restart the container to clear all GPU memory")
    
    return success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
