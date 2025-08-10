#!/usr/bin/env python3
"""
Simple GPU Memory Cleanup Script
Stops competing processes and frees GPU memory for Orpheus TTS
"""

import os
import sys
import subprocess
import time
import psutil

def kill_llama_processes():
    """Kill all llama-cpp and related processes"""
    print("🔄 Stopping LLM processes to free GPU memory...")
    
    killed_processes = []
    
    # Kill by process name patterns
    patterns = ['llama-cpp-python', 'llama', 'vocalis']
    
    for pattern in patterns:
        try:
            result = subprocess.run(['pkill', '-f', pattern], capture_output=True)
            if result.returncode == 0:
                print(f"   ✅ Stopped processes matching: {pattern}")
                killed_processes.append(pattern)
        except Exception as e:
            print(f"   ⚠️ Could not kill {pattern}: {e}")
    
    # Also kill by port (if llama-cpp is running on port 1234)
    try:
        result = subprocess.run(['lsof', '-ti:1234'], capture_output=True, text=True)
        if result.stdout.strip():
            pids = result.stdout.strip().split('\n')
            for pid in pids:
                if pid.strip():
                    subprocess.run(['kill', '-9', pid.strip()])
                    print(f"   ✅ Killed process on port 1234: PID {pid}")
                    killed_processes.append(f"PID-{pid}")
    except Exception as e:
        print(f"   ℹ️ No processes found on port 1234")
    
    if killed_processes:
        print(f"   🎯 Stopped: {', '.join(killed_processes)}")
        print("   ⏳ Waiting 5 seconds for GPU memory to clear...")
        time.sleep(5)
    else:
        print("   ℹ️ No LLM processes found to stop")
    
    return len(killed_processes) > 0

def clear_gpu_cache():
    """Clear GPU memory cache"""
    print("🧹 Clearing GPU memory cache...")
    
    try:
        # Clear PyTorch cache
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print("   ✅ PyTorch GPU cache cleared")
        else:
            print("   ⚠️ CUDA not available")
    except ImportError:
        print("   ℹ️ PyTorch not available for cache clearing")
    
    # Force garbage collection
    import gc
    gc.collect()
    print("   ✅ Python garbage collection completed")

def check_gpu_memory():
    """Check current GPU memory usage"""
    print("📊 Checking GPU memory status...")
    
    try:
        # Use nvidia-smi to check memory
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split('\n')[0]
            total, used, free = map(int, line.split(', '))
            
            total_gb = total / 1024
            used_gb = used / 1024
            free_gb = free / 1024
            
            print(f"   📈 Total: {total_gb:.1f}GB")
            print(f"   📊 Used:  {used_gb:.1f}GB")
            print(f"   📉 Free:  {free_gb:.1f}GB")
            
            if free_gb >= 20:
                print("   ✅ Sufficient memory for Orpheus TTS!")
                return True
            else:
                print(f"   ❌ Insufficient memory. Need ~20GB, have {free_gb:.1f}GB")
                return False
        else:
            print("   ⚠️ Could not read GPU memory status")
            return False
            
    except Exception as e:
        print(f"   ❌ Error checking GPU memory: {e}")
        return False

def main():
    """Main cleanup function"""
    print("🚀 GPU Memory Cleanup for Orpheus TTS")
    print("=" * 40)
    
    # Step 1: Check initial memory
    print("\n1️⃣ Initial Memory Check:")
    initial_memory_ok = check_gpu_memory()
    
    if initial_memory_ok:
        print("✅ GPU memory is already sufficient!")
        return True
    
    # Step 2: Kill competing processes
    print("\n2️⃣ Stopping Competing Processes:")
    processes_killed = kill_llama_processes()
    
    # Step 3: Clear caches
    print("\n3️⃣ Clearing GPU Cache:")
    clear_gpu_cache()
    
    # Step 4: Check final memory
    print("\n4️⃣ Final Memory Check:")
    final_memory_ok = check_gpu_memory()
    
    # Summary
    print("\n" + "=" * 40)
    print("📋 CLEANUP SUMMARY:")
    
    if final_memory_ok:
        print("🎉 SUCCESS: GPU memory freed!")
        print("\n📋 Next Steps:")
        print("1. Run Orpheus TTS test:")
        print("   python test_original_orpheus_fixed.py")
        print("\n2. Or start the TTS server:")
        print("   python memory_optimized_tts.py")
        return True
    else:
        print("💥 FAILED: Still insufficient GPU memory")
        print("\n🔧 Manual Solutions:")
        print("1. Restart the entire container:")
        print("   (This will clear all GPU memory)")
        print("\n2. Or check for other GPU processes:")
        print("   nvidia-smi")
        print("   pkill -f python")
        return False

if __name__ == '__main__':
    success = main()
    print(f"\n{'✅ Ready for Orpheus TTS!' if success else '❌ Manual intervention needed'}")
    sys.exit(0 if success else 1)
