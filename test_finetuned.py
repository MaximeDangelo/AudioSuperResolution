#!/usr/bin/env python3
"""
Test du modèle MetricGAN+ fine-tuné sur les vrais fichiers radio ATC.
Compare: original (noisy) vs modèle pré-entraîné vs modèle fine-tuné.
"""

import sys
import os
import torch
import torchaudio
import numpy as np
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from pesq import pesq
from pystoi import stoi as stoi_fn
from speechbrain.processing.features import spectral_magnitude

# Chemins
HPARAMS = "hparams/train_radio.yaml"
CKPT_DIR = "results/MetricGAN_radio/4234/save"
DATASET_RADIO = Path("/home/maxime/Bureau/speechbrain-develop/Dataset Radio (2)")
OUTPUT_DIR = Path("results/test_radio")
TARGET_SR = 16000

# Paires de test (noisy, clean)
TEST_PAIRS = [
    ("flac/02__kwahmah_atc002.flac", "Test1/02__kwahmah_atc002_clean.wav"),
    ("flac/03_kwahmah_atc003.flac", "Test1/03_kwahmah_atc003_clean.wav"),
    ("flac/05__kwahmah_atc005.flac", "Test1/05__kwahmah_atc005_clean.wav"),
    ("flac/06_kwahmah_atc006.flac", "Test1/06_kwahmah_atc006_clean.wav"),
]


def load_mono(path, target_sr=TARGET_SR):
    wav, sr = torchaudio.load(str(path))
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def enhance_with_finetuned(wav, hparams, generator, device="cpu"):
    """Applique le modèle fine-tuné."""
    wav = wav.to(device)
    with torch.no_grad():
        feats = hparams["compute_STFT"](wav)
        feats = spectral_magnitude(feats, power=0.5)
        feats = torch.log1p(feats)
        lens = torch.ones(1, device=device)
        mask = generator(feats, lengths=lens)
        mask = mask.clamp(min=hparams["min_mask"]).squeeze(2)
        enhanced_spec = torch.mul(mask, feats)
        enhanced_wav = hparams["resynth"](torch.expm1(enhanced_spec), wav)
    return enhanced_wav.cpu()


def enhance_with_pretrained(wav, device="cpu"):
    """Applique le modèle pré-entraîné (baseline)."""
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_metricgan",
    )
    enhancer.to(device)
    with torch.no_grad():
        enhanced = enhancer.enhance_batch(wav.to(device), torch.ones(1))
    return enhanced.cpu()


def compute_metrics(clean_np, test_np, sr=TARGET_SR):
    """Calcule PESQ et STOI."""
    ml = min(len(clean_np), len(test_np))
    clean_np, test_np = clean_np[:ml], test_np[:ml]
    try:
        p = pesq(sr, clean_np, test_np, "wb")
    except Exception:
        p = float("nan")
    try:
        s = stoi_fn(clean_np, test_np, sr)
    except Exception:
        s = float("nan")
    return p, s


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Charger hparams
    with open(HPARAMS, encoding="utf-8") as f:
        hparams = load_hyperpyyaml(f)

    # Trouver le meilleur checkpoint
    ckpts = sorted(Path(CKPT_DIR).iterdir())
    best_ckpt = ckpts[-1]  # Le dernier sauvegardé (meilleur PESQ)
    print(f"Checkpoint: {best_ckpt.name}")

    # Charger le générateur fine-tuné
    generator = hparams["models"]["generator"]
    ckpt_path = best_ckpt / "generator.ckpt"
    generator.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    generator.eval()
    print("Modèle fine-tuné chargé.\n")

    device = "cpu"

    # Charger le modèle pré-entraîné (baseline)
    print("Chargement du modèle pré-entraîné (baseline)...")
    from speechbrain.inference.enhancement import SpectralMaskEnhancement
    pretrained = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_metricgan",
    )
    print("OK\n")

    print(f"{'Fichier':<40s} | {'':^30s} | {'':^30s} | {'':^30s}")
    print(f"{'':40s} | {'NOISY (original)':^30s} | {'PRÉ-ENTRAÎNÉ':^30s} | {'FINE-TUNÉ':^30s}")
    print(f"{'':40s} | {'PESQ':>8s} {'STOI':>8s} {'':12s} | {'PESQ':>8s} {'STOI':>8s} {'':12s} | {'PESQ':>8s} {'STOI':>8s} {'':12s}")
    print("-" * 135)

    all_results = []

    for noisy_rel, clean_rel in TEST_PAIRS:
        noisy_path = DATASET_RADIO / noisy_rel
        clean_path = DATASET_RADIO / clean_rel
        if not noisy_path.exists() or not clean_path.exists():
            print(f"  SKIP: {noisy_rel}")
            continue

        name = Path(noisy_rel).stem
        noisy_wav = load_mono(noisy_path)
        clean_wav = load_mono(clean_path)

        # Aligner longueurs
        ml = min(noisy_wav.shape[1], clean_wav.shape[1])
        noisy_wav = noisy_wav[:, :ml]
        clean_wav = clean_wav[:, :ml]

        clean_np = clean_wav.squeeze().numpy()
        noisy_np = noisy_wav.squeeze().numpy()

        # Métriques noisy
        p_noisy, s_noisy = compute_metrics(clean_np, noisy_np)

        # Enhancement pré-entraîné
        with torch.no_grad():
            enh_pre = pretrained.enhance_batch(noisy_wav, torch.ones(1))
        enh_pre_np = enh_pre.squeeze().cpu().numpy()
        p_pre, s_pre = compute_metrics(clean_np, enh_pre_np)

        # Enhancement fine-tuné
        enh_ft = enhance_with_finetuned(noisy_wav, hparams, generator, device)
        enh_ft_np = enh_ft.squeeze().numpy()
        p_ft, s_ft = compute_metrics(clean_np, enh_ft_np)

        print(f"  {name:<38s} | {p_noisy:8.3f} {s_noisy:8.3f} {'':12s} | {p_pre:8.3f} {s_pre:8.3f} {'':12s} | {p_ft:8.3f} {s_ft:8.3f} {'':12s}")

        # Sauvegarder les fichiers enhanced
        torchaudio.save(str(OUTPUT_DIR / f"{name}_finetuned.wav"), enh_ft, TARGET_SR)
        torchaudio.save(str(OUTPUT_DIR / f"{name}_pretrained.wav"), enh_pre.cpu(), TARGET_SR)

        all_results.append({
            "name": name,
            "pesq_noisy": p_noisy, "stoi_noisy": s_noisy,
            "pesq_pre": p_pre, "stoi_pre": s_pre,
            "pesq_ft": p_ft, "stoi_ft": s_ft,
        })

    # Moyennes
    if all_results:
        print("-" * 135)
        avg = lambda key: np.mean([r[key] for r in all_results])
        print(f"  {'MOYENNE':<38s} | {avg('pesq_noisy'):8.3f} {avg('stoi_noisy'):8.3f} {'':12s} | {avg('pesq_pre'):8.3f} {avg('stoi_pre'):8.3f} {'':12s} | {avg('pesq_ft'):8.3f} {avg('stoi_ft'):8.3f} {'':12s}")
        print()
        print(f"  Amélioration PESQ fine-tuné vs noisy:       +{avg('pesq_ft') - avg('pesq_noisy'):.3f}")
        print(f"  Amélioration PESQ fine-tuné vs pré-entraîné: +{avg('pesq_ft') - avg('pesq_pre'):.3f}")

    print(f"\nFichiers enhanced sauvegardés dans: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
