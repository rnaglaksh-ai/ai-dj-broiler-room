import time
import numpy as np

def stress_test_audio_engine(num_stems=8, buffer_size=512, iterations=1000):
    print(f"--- Starting Stress Test: {num_stems} Stems @ {buffer_size} samples ---")
    latencies = []

    # Simulate 8 stems of audio data
    stems = [np.random.rand(buffer_size).astype(np.float32) for _ in range(num_stems)]
    
    for i in range(iterations):
        start_time = time.perf_counter()
        
        # SIMULATE PROCESSING (The "Work"):
        # 1. Summing stems
        mixed_audio = sum(stems)
        # 2. Applying fake EQ/FX (floating point math)
        processed = np.tanh(mixed_audio * 1.2) 
        
        end_time = time.perf_counter()
        latencies.append(end_time - start_time)

    avg_latency = np.mean(latencies) * 1000 # convert to ms
    max_latency = np.max(latencies) * 1000
    
    # THRESHOLD CHECK: At 44.1kHz, a 512 buffer must be processed in < 11.6ms
    print(f"Average Latency: {avg_latency:.4f} ms")
    print(f"Worst-Case Spike: {max_latency:.4f} ms")
    
    if max_latency > 10.0:
        print("CRITICAL: Buffer underrun risk detected! Optimize your DSP.")
    else:
        print("PASS: Audio engine is stable.")

if __name__ == "__main__":
    try:
        stress_test_audio_engine()
    except ImportError:
        print("Error: NumPy is required. Please install it (pip install numpy).")
    except Exception as e:
        print(f"An error occurred: {e}")
