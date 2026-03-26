"""
Genere 5 fichiers audio passes dans le ResUNet NON entraine (poids aleatoires).
Utilise le CPU uniquement pour eviter les GPU hangs ROCm.
Sortie : output/unet_untrained/
"""
import os, sys, subprocess, tempfile
import numpy as np
import soundfile as sf
import torch

sys.stdout.reconfigure(encoding="utf-8")

# Forcer CPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from train import SpectralResUNet, SR

DEVICE  = torch.device("cpu")
OUT_DIR = os.path.join("output", "unet_untrained")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    ("Dataset Radio (2)/01__kwahmah_atc001.flac",   "atc01"),
    ("Dataset Radio (2)/03_kwahmah_atc003.flac",    "atc03"),
    ("Dataset Radio (2)/05__kwahmah_atc005.flac",   "atc05"),
    ("dataset/val/raw/synth_val_00000.wav",          "synth_00000"),
    ("dataset/val/raw/synth_val_00050.wav",          "synth_00050"),
]

def load_wav(path, max_s=10):
    tmp = os.path.join(tempfile.gettempdir(), "_unet_tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-t", str(max_s),
                    "-ar", str(SR), "-ac", "1", tmp],
                   capture_output=True, check=True)
    data, _ = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return data

print("Chargement ResUNet (poids ALEATOIRES, CPU)...")
model = SpectralResUNet().to(DEVICE)
model.eval()

for src_path, name in FILES:
    if not os.path.exists(src_path):
        print(f"  SKIP {src_path}")
        continue

    print(f"  {name} ...")
    raw = load_wav(src_path)

    with torch.no_grad():
        x   = torch.tensor(raw).unsqueeze(0).unsqueeze(0)
        out = model(x).squeeze().numpy().astype(np.float32)

    if np.max(np.abs(out)) > 0:
        out = out / np.max(np.abs(out)) * 0.9

    sf.write(os.path.join(OUT_DIR, f"{name}_src.wav"),  raw, SR)
    sf.write(os.path.join(OUT_DIR, f"{name}_unet.wav"), out, SR)
    print(f"    -> {name}_src.wav / {name}_unet.wav")

print(f"\nTermine. Fichiers dans : {OUT_DIR}")
