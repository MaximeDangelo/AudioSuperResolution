"""Test du MetricGAN+ fine-tuné sur les vrais fichiers radio ATC.
Chargement direct du modèle (sans YAML) pour éviter les conflits torchvision."""

import sys
import os
import torch
import torchaudio
import numpy as np

sys.path.insert(0, "/home/maxime/Bureau/speechbrain-develop")

from pesq import pesq
from pystoi import stoi as stoi_fn
from speechbrain.lobes.models.MetricGAN import EnhancementGenerator
from speechbrain.processing.features import STFT, ISTFT, spectral_magnitude
from speechbrain.processing.signal_processing import resynthesize

RADIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset Radio (2)")
OUTPUT_DIR = os.path.join(RADIO_DIR, "metricgan_finetuned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_DIR = "/home/maxime/Bureau/AudioSuperResolution/results/MetricGAN_radio/4234/save/CKPT+2026-03-25+11-11-53+00"

TARGET_SR = 16000

PAIRS = [
    ("flac/02__kwahmah_atc002.flac", "Test1/02__kwahmah_atc002_clean.wav"),
    ("flac/03_kwahmah_atc003.flac", "Test1/03_kwahmah_atc003_clean.wav"),
    ("flac/05__kwahmah_atc005.flac", "Test1/05__kwahmah_atc005_clean.wav"),
    ("flac/06_kwahmah_atc006.flac", "Test1/06_kwahmah_atc006_clean.wav"),
]

NOISY_ONLY = [
    "04-voicebot_pilotes-pendant-un-vol.mp3",
    "09_fumesKMDT1-App-Dep-West-Feb-02-2023-1800Z.mp3",
    "10_hercky69EINN2-Feb-08-2023-0630Z.mp3",
]


def load_mono_16k(path, max_seconds=30):
    info = torchaudio.info(str(path))
    sr_orig = info.sample_rate
    max_frames = int(max_seconds * sr_orig)
    num_frames = min(info.num_frames, max_frames)
    wav, sr = torchaudio.load(str(path), num_frames=num_frames)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    return wav


def compute_metrics(clean_np, enhanced_np, sr):
    ml = min(len(clean_np), len(enhanced_np))
    c, e = clean_np[:ml], enhanced_np[:ml]
    try:
        p = pesq(sr, c, e, "wb")
    except Exception:
        p = float("nan")
    try:
        s = stoi_fn(c, e, sr)
    except Exception:
        s = float("nan")
    return p, s


def enhance(noisy_wav, generator, compute_stft, compute_istft, device,
            chunk_seconds=4.0, overlap_seconds=0.5):
    """Enhance par chunks de chunk_seconds avec overlap pour éviter les artefacts."""
    sr = TARGET_SR
    chunk_len = int(chunk_seconds * sr)
    overlap_len = int(overlap_seconds * sr)
    hop = chunk_len - overlap_len

    noisy_np = noisy_wav.squeeze().numpy()
    total_len = len(noisy_np)

    if total_len <= chunk_len:
        # Petit fichier : traitement direct
        return _enhance_chunk(noisy_wav, generator, compute_stft, compute_istft, device)

    # Traitement par chunks avec overlap-add
    output = np.zeros(total_len, dtype=np.float32)
    weight = np.zeros(total_len, dtype=np.float32)

    pos = 0
    while pos < total_len:
        end = min(pos + chunk_len, total_len)
        chunk = torch.tensor(noisy_np[pos:end], dtype=torch.float32).unsqueeze(0)
        enh_chunk = _enhance_chunk(chunk, generator, compute_stft, compute_istft, device)

        # Fenêtre de fondu pour l'overlap
        clen = len(enh_chunk)
        win = np.ones(clen, dtype=np.float32)
        if pos > 0 and overlap_len > 0:
            fade_len = min(overlap_len, clen)
            win[:fade_len] = np.linspace(0, 1, fade_len)
        if end < total_len and overlap_len > 0:
            fade_len = min(overlap_len, clen)
            win[-fade_len:] = np.linspace(1, 0, fade_len)

        output[pos:pos + clen] += enh_chunk * win
        weight[pos:pos + clen] += win
        pos += hop

    weight = np.maximum(weight, 1e-8)
    return output / weight


def _enhance_chunk(noisy_wav, generator, compute_stft, compute_istft, device):
    with torch.no_grad():
        noisy_gpu = noisy_wav.to(device)
        stft_out = compute_stft(noisy_gpu)
        noisy_spec = spectral_magnitude(stft_out, power=0.5)
        noisy_spec = torch.log1p(noisy_spec)

        lengths = torch.tensor([1.0], device=device)
        mask = generator(noisy_spec, lengths=lengths)
        mask = mask.clamp(min=0.05)
        predict_spec = torch.mul(mask, noisy_spec)

        predict_spec_linear = torch.expm1(predict_spec)
        enhanced_wav = resynthesize(
            predict_spec_linear, noisy_gpu,
            stft=compute_stft, istft=compute_istft, normalize_wavs=False
        )
    return enhanced_wav.squeeze().cpu().numpy()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Charger le modèle directement
    print("\nChargement du modèle MetricGAN+ fine-tuné (PESQ=2.91)...")
    generator = EnhancementGenerator()
    generator.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "generator.ckpt"), map_location=device
    ))
    generator.to(device).eval()

    compute_stft = STFT(sample_rate=TARGET_SR, win_length=32, hop_length=16, n_fft=512,
                        window_fn=torch.hamming_window)
    compute_istft = ISTFT(sample_rate=TARGET_SR, win_length=32, hop_length=16,
                          window_fn=torch.hamming_window)
    print("OK\n")

    # === Paires avec référence clean ===
    print("=" * 90)
    print("  PAIRES AVEC RÉFÉRENCE CLEAN")
    print("=" * 90)
    print(f"{'Fichier':<40s} {'PESQ_bef':>9s} {'PESQ_aft':>9s} {'STOI_bef':>9s} {'STOI_aft':>9s}")
    print("-" * 90)

    for noisy_rel, clean_rel in PAIRS:
        noisy_path = os.path.join(RADIO_DIR, noisy_rel)
        clean_path = os.path.join(RADIO_DIR, clean_rel)
        if not os.path.exists(noisy_path) or not os.path.exists(clean_path):
            continue

        name = os.path.splitext(os.path.basename(noisy_path))[0]
        noisy_wav = load_mono_16k(noisy_path)
        clean_wav = load_mono_16k(clean_path)

        noisy_np = noisy_wav.squeeze().numpy()
        clean_np = clean_wav.squeeze().numpy()

        pesq_bef, stoi_bef = compute_metrics(clean_np, noisy_np, TARGET_SR)
        enhanced_np = enhance(noisy_wav, generator, compute_stft, compute_istft, device)
        pesq_aft, stoi_aft = compute_metrics(clean_np, enhanced_np, TARGET_SR)

        out_path = os.path.join(OUTPUT_DIR, f"{name}_metricgan_ft.wav")
        torchaudio.save(out_path, torch.tensor(enhanced_np).unsqueeze(0), TARGET_SR)

        print(f"{name:<40s} {pesq_bef:9.3f} {pesq_aft:9.3f} {stoi_bef:9.3f} {stoi_aft:9.3f}")

    # === Fichiers sans référence clean ===
    print(f"\n{'=' * 90}")
    print("  FICHIERS SANS RÉFÉRENCE (test qualitatif)")
    print("=" * 90)

    for fname in NOISY_ONLY:
        noisy_path = os.path.join(RADIO_DIR, fname)
        if not os.path.exists(noisy_path):
            continue

        name = os.path.splitext(fname)[0]
        noisy_wav = load_mono_16k(noisy_path)
        enhanced_np = enhance(noisy_wav, generator, compute_stft, compute_istft, device)

        out_path = os.path.join(OUTPUT_DIR, f"{name}_metricgan_ft.wav")
        torchaudio.save(out_path, torch.tensor(enhanced_np).unsqueeze(0), TARGET_SR)

        rms_bef = np.sqrt(np.mean(noisy_wav.squeeze().numpy() ** 2))
        rms_aft = np.sqrt(np.mean(enhanced_np ** 2))
        print(f"  {name}: sauvegardé")
        print(f"    RMS avant={rms_bef:.4f}, après={rms_aft:.4f}")

    print(f"\nTous les fichiers enhanced dans: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
