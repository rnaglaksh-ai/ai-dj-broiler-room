import os
import json
import numpy as np
import librosa
import soundfile as sf
import traceback

def analyze_phrasing(file_path: str):
    """
    Detects DJ-friendly 16/32-bar phrases, build-ups, and drops (chorus).
    Returns timestamps in seconds.
    """
    try:
        # Load audio (downsampled to 22.05kHz mono for speed)
        y, sr = librosa.load(file_path, sr=22050, mono=True)
        
        # 1. Structural segmentation (Onset / Beat Tracking)
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beats, sr=sr)
        
        # 2. Extract Spectral Novelty (Energy drops / explodes)
        # Using Root Mean Square (RMS) energy to detect intense vs quiet sections
        rms = librosa.feature.rms(y=y)[0]
        rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        
        # Smooth the RMS to find macroscopic structure (windows of ~4 seconds)
        window_length = int(sr / 512 * 4) # 4 seconds approx in frames
        if window_length % 2 == 0:
            window_length += 1
            
        import scipy.signal
        rms_smoothed = scipy.signal.savgol_filter(rms, window_length, 3)
        
        # 3. Detect "Drops" (Sudden, massive spikes in sustained energy)
        # Calculate the derivative of the smoothed energy
        energy_diff = np.diff(rms_smoothed)
        
        # Find peaks in energy increases
        peaks, _ = scipy.signal.find_peaks(energy_diff, distance=int(sr/512 * 16), prominence=np.std(energy_diff)*2)
        
        drops = []
        for p in peaks:
            time_sec = rms_times[p]
            # Snap to nearest beat
            closest_beat = beat_times[np.argmin(np.abs(beat_times - time_sec))]
            drops.append(closest_beat)
            
        # 4. Detect Breakdowns (Sudden, massive dips in energy)
        valleys, _ = scipy.signal.find_peaks(-energy_diff, distance=int(sr/512 * 16), prominence=np.std(energy_diff)*2)
        
        breakdowns = []
        for v in valleys:
            time_sec = rms_times[v]
            # Snap to nearest beat
            closest_beat = beat_times[np.argmin(np.abs(beat_times - time_sec))]
            breakdowns.append(closest_beat)
            
        # 5. Outro logic (Gradual or sudden loss of energy at the end of the track)
        track_duration = len(y) / sr
        outro_candidates = [b for b in breakdowns if b > track_duration * 0.75]
        outro_start = outro_candidates[0] if outro_candidates else track_duration - 30 # Default 30s to end if no breakdown found
        
        # 6. Intro logic (When does the main energy kick in?)
        intro_candidates = [d for d in drops if d < track_duration * 0.3]
        intro_end = intro_candidates[0] if intro_candidates else 30 # Default 30s if no early drop found
        
        return {
            "tempo": float(tempo[0] if isinstance(tempo, np.ndarray) else tempo),
            "musicStartTime": float(beat_times[0]) if len(beat_times) > 0 else 0.0,
            "introEndTime": float(intro_end),
            "drops": [float(d) for d in drops],
            "breakdowns": [float(b) for b in breakdowns],
            "outroStartTime": float(outro_start),
            "duration": float(track_duration)
        }
        
    except Exception as e:
        print(f"Error in phrase analysis: {e}")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Internal simple test
    import sys
    if len(sys.argv) > 1:
        res = analyze_phrasing(sys.argv[1])
        print(json.dumps(res, indent=2))
        
