"""Test SGMSE+ (diffusion-based) sur les fichiers radio réels."""

import torch
import torchaudio
import numpy as np
import os
import sys

sys.path.insert(0, "/home/maxime/Bureau/speechbrain-develop")

from speechbrain.inference.enhancement import SGMSEEnhancement
from pesq import pesq
from pystoi import stoi as stoi_fn

RADIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset Radio (2)")
OUTPUT_DIR = os.path.join(RADIO_DIR, "sgmse_enhanced")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paires noisy/clean pour métriques
PAIRS = [
    ("flac/02__kwahmah_atc002.flac", "Test1/02__kwahmah_atc002_clean.wav"),
    ("flac/03_kwahmah_atc003.flac", "Test1/03_kwahmah_atc003_clean.wav"),
    ("flac/05__kwahmah_atc005.flac", "Test1/05__kwahmah_atc005_clean.wav"),
    ("flac/06_kwahmah_atc006.flac", "Test1/06_kwahmah_atc006_clean.wav"),
]

TARGET_SR = 16000


def load_mono_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    return wav, TARGET_SR


def compute_metrics(clean_np, enhanced_np, sr):
    ml = min(len(clean_np), len(enhanced_np))
    clean_np = clean_np[:ml]
    enhanced_np = enhanced_np[:ml]
    try:
        p = pesq(sr, clean_np, enhanced_np, "wb")
    except Exception:
        p = float("nan")
    try:
        s = stoi_fn(clean_np, enhanced_np, sr)
    except Exception:
        s = float("nan")
    return p, s


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\nChargement du modèle SGMSE+ (speechbrain/sgmse-voicebank)...")
    enhancer = SGMSEEnhancement.from_hparams(
        source="speechbrain/sgmse-voicebank",
        savedir="pretrained_models/sgmse-voicebank",
        run_opts={"device": device},
    )
    print("Modèle chargé.\n")

    print(f"{'Fichier':<45s} {'PESQ_before':>11s} {'PESQ_after':>11s} {'STOI_before':>11s} {'STOI_after':>11s}")
    print("-" * 95)

    for noisy_rel, clean_rel in PAIRS:
        noisy_path = os.path.join(RADIO_DIR, noisy_rel)
        clean_path = os.path.join(RADIO_DIR, clean_rel)

        if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
            print(f"  SKIP: {noisy_rel}")
            continue

        name = os.path.splitext(os.path.basename(noisy_path))[0]

        # Charger
        noisy_wav, sr = load_mono_16k(noisy_path)
        clean_wav, _ = load_mono_16k(clean_path)

        noisy_np = noisy_wav.squeeze().numpy()
        clean_np = clean_wav.squeeze().numpy()

        # Métriques avant
        pesq_before, stoi_before = compute_metrics(clean_np, noisy_np, sr)

        # Enhancement SGMSE+
        with torch.no_grad():
            enhanced = enhancer.enhance_batch(noisy_wav.to(device))
        enhanced_np = enhanced.squeeze().cpu().numpy()

        # Sauvegarder
        out_path = os.path.join(OUTPUT_DIR, f"{name}_sgmse.wav")
        torchaudio.save(out_path, torch.tensor(enhanced_np).unsqueeze(0), sr)

        # Métriques après
        pesq_after, stoi_after = compute_metrics(clean_np, enhanced_np, sr)

        print(f"{name:<45s} {pesq_before:11.3f} {pesq_after:11.3f} {stoi_before:11.3f} {stoi_after:11.3f}")

    print(f"\nFichiers enhanced sauvegardés dans: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
