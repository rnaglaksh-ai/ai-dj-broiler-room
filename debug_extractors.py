
import inspect
import sys
import os

# Add local ffmpeg to PATH
ffmpeg_bin = os.path.abspath("ffmpeg-master-latest-win64-gpl/bin")
if os.path.exists(ffmpeg_bin) and ffmpeg_bin not in os.environ["PATH"]:
    print(f"[*] Adding local ffmpeg to PATH: {ffmpeg_bin}")
    os.environ["PATH"] += os.pathsep + ffmpeg_bin

# Check Audio Separator (Lite)
try:
    from audio_separator.separator import Separator
    print("[*] Checking Audio Separator models...")
    sep = Separator()
    print("Separator initialized successfully.")
    
    # Check if download_model_list exists
    if hasattr(sep, 'download_model_list'):
        print(f"Has download_model_list: Yes")
        # List default models?
    else:
        print(f"Has download_model_list: No")
        
    print(f"Supported models: {list(sep.list_supported_models()) if hasattr(sep, 'list_supported_models') else 'Unknown'}")
    
    # Check if there is a known models list?
    # Inspect load_model signature
    if hasattr(sep, 'load_model'):
        sig = inspect.signature(sep.load_model)
        print(f"Separator.load_model signature: {sig}")
    else:
        print("Separator has no load_model method")

except ImportError as e:
    print(f"[!] audio-separator not importable: {e}")
except Exception as e:
    print(f"[!] audio-separator init failed: {e}")

print("-" * 30)

# Check Demucs (HQ)
try:
    from demucs.api import Separator as DemucsSeparator
    print("[*] Checking Demucs API...")
    try:
        sig = inspect.signature(DemucsSeparator.separate_tensor)
        print(f"DemucsSeparator.separate_tensor signature: {sig}")
    except AttributeError:
        # Maybe it's not a method?
        print("DemucsSeparator has no separate_tensor method?")
        print(dir(DemucsSeparator))
except ImportError as e:
    print(f"[!] demucs not importable: {e}")
