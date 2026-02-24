
import os
import time
import subprocess
import hashlib
from pathlib import Path
import soundfile as sf
import numpy as np

# --- CONFIGURATION (Ndot Standard) ---
# Adjust these paths to match your actual production setup
LIBRARY_DIR = "./stems_cache/uploads"    # Scanning uploads for now, or a dedicated music folder
CACHE_DIR = "./stems_cache/stems"        # Where stems live
MODEL_VOCAL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt" 
MODEL_RHYTHM = "hdemucs_mmi"

def ensure_directories():
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR, exist_ok=True)
    if not os.path.exists(LIBRARY_DIR):
        os.makedirs(LIBRARY_DIR, exist_ok=True)

def get_analyzed_track_ids():
    # Returns a set of track_ids (folder names) already in the cache
    return {d for d in os.listdir(CACHE_DIR) if os.path.isdir(os.path.join(CACHE_DIR, d))}

def calculate_file_hash(filepath):
    """Generate a consistent hash for audio files to detect duplicates/renames."""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        # Read first 8MB (header + start) to be fast but reasonably unique
        buf = f.read(8 * 1024 * 1024)
        hasher.update(buf)
    return hasher.hexdigest()[:12]

def normalize_stem(file_path, target_lufs=-14.0):
    """
    Apply Gain Normalization to a specific stem file.
    Simple Peak implementation for now (approx -6dB peak for safety).
    """
    try:
        data, sr = sf.read(file_path)
        peak = np.max(np.abs(data))
        if peak > 0:
            # Target ~ -6dB peak for safety (approx matches -14LUFS for dense stems)
            target_peak = 0.5 
            gain = target_peak / peak
            # Don't boost noise floor
            if gain < 1.0 or (gain > 1.0 and gain < 4.0):
                data = data * gain
                sf.write(file_path, data, sr)
                print(f"    [NORM] Normalized {Path(file_path).name} (Gain: {gain:.2f})")
    except Exception as e:
        print(f"    [!] Norm failed for {file_path}: {e}")

def pre_analyze_library():
    print("\n[ NDOT ] ANTIGRAVITY PRE-ANALYSIS DAEMON STARTING...")
    print(f"[ CONF ] Library: {LIBRARY_DIR}")
    print(f"[ CONF ] Cache:   {CACHE_DIR}")
    
    ensure_directories()
    
    # Analyze Loop
    while True:
        analyzed_ids = get_analyzed_track_ids()
        
        # Scan for files in LIBRARY_DIR (which serves as our "Inbox")
        # In a real app, this might scan a recursive music folder.
        # Here we scan the uploads folder where the app drops files.
        files_to_process = []
        for f in os.listdir(LIBRARY_DIR):
            if f.endswith(('.mp3', '.wav', '.flac', '.m4a')):
                # In our server, files are named {track_id}_{filename}
                # We can extract track_id from filename or re-hash.
                # Let's assume standard format matches server.py
                parts = f.split('_', 1)
                if len(parts) > 1:
                    track_id = parts[0]
                    if len(track_id) == 12 and track_id not in analyzed_ids:
                        files_to_process.append((track_id, os.path.join(LIBRARY_DIR, f)))

        if not files_to_process:
            print("[ SLEEP ] Waiting for new DNA...", end='\r')
            time.sleep(5)
            continue
            
        print(f"\n[ ::: ] Found {len(files_to_process)} new tracks for Ndot analysis.")

        for track_id, input_path in files_to_process:
            output_subfolder = os.path.join(CACHE_DIR, track_id)
            os.makedirs(output_subfolder, exist_ok=True)
            
            print(f"[ .... ] Analyzing DNA: {Path(input_path).name}...")
            
            start_time = time.time()
            
            # 1. BS-Roformer (Vocals)
            # Using audio-separator CLI
            print("    [ 1/2 ] Extracting Vocals (BS-Roformer)...")
            try:
                # We need to run inside the python env. 
                # Assuming 'audio-separator' is in path (pip installed)
                subprocess.run([
                    "audio-separator", 
                    input_path,
                    "--model_filename", MODEL_VOCAL,
                    "--output_dir", output_subfolder,
                    "--output_format", "WAV",
                    "--normalization_threshold", "0.9" 
                ], check=True, stdout=subprocess.DEVNULL)
                
                # Rename the output to standard 'vocals.wav' / 'other.wav'
                # audio-separator filenames are complex, need to find them
                for f in os.listdir(output_subfolder):
                    if "Vocals" in f and f.endswith(".wav"):
                        os.replace(os.path.join(output_subfolder, f), os.path.join(output_subfolder, f"vocals.wav"))
                    elif "Instrumental" in f and f.endswith(".wav"):
                         # This instrumental acts as source for next step
                         os.replace(os.path.join(output_subfolder, f), os.path.join(output_subfolder, f"temp_inst.wav"))
                
            except Exception as e:
                print(f"[RED_ALERT] Vocal extraction failed: {e}")
                continue

            # 2. HT-Demucs (Drums/Bass)
            # We run this on the 'temp_inst.wav' to split it further
            print("    [ 2/2 ] Splitting Rhythm (HT-Demucs)...")
            temp_inst = os.path.join(output_subfolder, "temp_inst.wav")
            if os.path.exists(temp_inst):
                try:
                    # Run Demucs on the instrumental
                    # demucs -n hdemucs_mmi -o {output_subfolder} {temp_inst}
                    # Demucs creates subfolders by model name, need to manage that
                    subprocess.run([
                        "demucs",
                        "-n", "hdemucs_mmi",
                        "-o", output_subfolder,
                        "--jobs", "2", 
                        temp_inst
                    ], check=True, stdout=subprocess.DEVNULL)
                    
                    # Move files from demucs subfolder to root of track folder
                    demucs_out = os.path.join(output_subfolder, "hdemucs_mmi", "temp_inst")
                    if os.path.exists(demucs_out):
                        for stem in ["drums.wav", "bass.wav", "other.wav"]:
                             src = os.path.join(demucs_out, stem)
                             dst = os.path.join(output_subfolder, stem)
                             if os.path.exists(src):
                                 os.replace(src, dst)
                    
                    # Clean up
                    try:
                        import shutil
                        shutil.rmtree(os.path.join(output_subfolder, "hdemucs_mmi"))
                        os.remove(temp_inst)
                    except: pass
                    
                except Exception as e:
                    print(f"[RED_ALERT] Rhythm splitting failed: {e}")

            # 3. Post-Process (Normalization)
            print("    [ NORM ] Applying Fidelity Guard (-14LUFS equivalent)...")
            for stem in ["vocals.wav", "drums.wav", "bass.wav", "other.wav"]:
                p = os.path.join(output_subfolder, stem)
                if os.path.exists(p):
                    normalize_stem(p)
            
            elapsed = time.time() - start_time
            print(f"[ ::.. ] Done! Stems cached in {elapsed:.1f}s")

if __name__ == "__main__":
    pre_analyze_library()
