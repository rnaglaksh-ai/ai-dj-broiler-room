
import os
import sys
import glob
import subprocess
import asyncio
from pathlib import Path
from stem_extractor import DJStemExtractor

def download_audio(url: str, output_dir: str):
    """Download audio from YouTube using yt-dlp"""
    print(f"[*] Downloading {url}...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Template: Title [ID].wav
    out_tmpl = os.path.join(output_dir, "%(title)s [%(id)s].%(ext)s")
    
    cmd = [
        "python", "-m", "yt_dlp",
        "-x", "--audio-format", "wav",
        "--audio-quality", "0",
        "--ffmpeg-location", "ffmpeg-master-latest-win64-gpl/bin",
        "-o", out_tmpl,
        url
    ]
    
    subprocess.run(cmd, check=True)
    
    # Find the downloaded file (most recent WAV in dir)
    files = list(glob.glob(os.path.join(output_dir, "*.wav")))
    files.sort(key=os.path.getmtime)
    if not files:
        raise FileNotFoundError("Download failed, no WAV found")
    
    return files[-1]

async def process_track(url: str):
    base_dir = "stems_cache"
    upload_dir = os.path.join(base_dir, "uploads")
    stems_dir = os.path.join(base_dir, "stems")
    
    # 1. Download
    try:
        wav_path = download_audio(url, upload_dir)
        print(f"[+] Downloaded: {wav_path}")
    except Exception as e:
        print(f"[!] Download error: {e}")
        return

    # 2. Initialize Extractor
    print("[*] Initializing Dual-Engine Extractor...")
    extractor = DJStemExtractor()
    
    track_name = Path(wav_path).stem
    track_id = track_name.split("[")[-1].replace("]", "") if "[" in track_name else "yt_import"
    out_dir = os.path.join(stems_dir, track_id)

    # 3. Lite Separation (MDX-Net)
    print(f"[*] Starting Lite Separation (MDX-Net) for {track_name}...")
    try:
        lite_paths = await asyncio.to_thread(extractor.separate_lite, wav_path, out_dir)
        print(f"[+] Lite Stems Ready: {list(lite_paths.keys())}")
    except Exception as e:
        print(f"[!] Lite engine failed: {e}")

    # 4. HQ Separation (HT-Demucs)
    print(f"[*] Starting HQ Separation (hdemucs_mmi)...")
    try:
        hq_paths = await asyncio.to_thread(extractor._separate_demucs, wav_path, out_dir, shifts=2)
        print(f"[+] HQ Stems Ready: {list(hq_paths.keys())}")
    except Exception as e:
        print(f"[!] HQ engine failed: {e}")

    print("\n" + "="*50)
    print(f"DONE! Track processed: {track_name}")
    print(f"Audio File: {wav_path}")
    print(f"Stems Directory: {out_dir}")
    print("="*50)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python download_and_extract.py <youtube_url>")
        sys.exit(1)
        
    url = sys.argv[1]
    asyncio.run(process_track(url))
