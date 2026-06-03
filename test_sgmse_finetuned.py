"""
Test SGMSE+ fine-tune sur fichiers radio reels + echantillons de validation.

Charge le checkpoint fine-tune (avec poids EMA) et genere des audios enhanced.
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

sys.stdout.reconfigure(encoding="utf-8")

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "sgmse-voicebank")
CHECKPOINT_PATH = os.path.join(SCRIPT_DIR, "checkpoints", "sgmse", "best_model.pt")
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
RADIO_DIR = os.path.join(SCRIPT_DIR, "Dataset Radio (2)")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "runs", "sgmse_test_v3")

os.makedirs(OUTPUT_DIR, exist_ok=True)

SR_MODEL = 16000
N_FFT = 510
HOP_LENGTH = 128

# Fichiers radio reels a tester
RADIO_FILES = [
    ("flac/05__kwahmah_atc005.flac", "Test1/05__kwahmah_atc005_clean.wav"),
    ("flac/06_kwahmah_atc006.flac", "Test1/06_kwahmah_atc006_clean.wav"),
    ("flac/07-wahmah_heathrow-air-traffic-control.flac",
     "Test1/07-wahmah_heathrow-air-traffic-control_clean.wav"),
    ("flac/08__kwahmah_hong-kong-air-traffic-control.flac",
     "Test1/08__kwahmah_hong-kong-air-traffic-control_clean.wav"),
]

# Echantillons de validation a tester
VAL_INDICES = [0, 5, 10, 50, 100]

MAX_DURATION_S = 30  # Limiter pour les fichiers longs


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def get_window(device):
    return torch.hann_window(N_FFT).to(device)


def stft(x, window):
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=window, center=True, return_complex=True)


def istft(spec, window, length=None):
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       window=window, center=True, length=length)


def spec_fwd(S, factor=0.15, exponent=0.5):
    if exponent != 1.0:
        mag = S.abs() ** exponent
        ph = S.angle()
        S = mag * torch.exp(1j * ph)
    return S * factor


def spec_back(S, factor=0.15, exponent=0.5):
    S = S / factor
    if exponent != 1.0:
        mag = S.abs() ** (1.0 / exponent)
        ph = S.angle()
        S = mag * torch.exp(1j * ph)
    return S


def pad_spec(spec):
    T = spec.shape[-1]
    num_pad = (64 - T % 64) % 64
    if num_pad > 0:
        real = spec.real
        imag = spec.imag
        real = F.pad(real, (0, num_pad))
        imag = F.pad(imag, (0, num_pad))
        spec = torch.complex(real, imag)
    return spec


def load_model(device):
    """Charge SGMSE+ pre-entraine puis applique les poids EMA du checkpoint fine-tune."""
    from speechbrain.inference.enhancement import SGMSEEnhancement

    print("Chargement SGMSE+ pre-entraine...")
    pretrained = SGMSEEnhancement.from_hparams(
        source="speechbrain/sgmse-voicebank",
        savedir=PRETRAINED_DIR,
    )
    score_model = pretrained.mods.score_model

    print(f"Chargement checkpoint fine-tune: {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"  Epoch: {ckpt['epoch']}, Val loss: {ckpt['val_loss']:.4f}")

    # Charger les poids EMA (meilleurs pour l'inference)
    if "ema_state_dict" in ckpt and "shadow_params" in ckpt["ema_state_dict"]:
        shadow_params = ckpt["ema_state_dict"]["shadow_params"]
        for p, s in zip(score_model.dnn.parameters(), shadow_params):
            p.data.copy_(s.to(device))
        print("  Poids EMA charges dans le DNN")
    else:
        # Fallback : charger les poids normaux
        score_model.load_state_dict(ckpt["model_state_dict"])
        print("  Poids normaux charges (pas d'EMA)")

    score_model = score_model.to(device)
    score_model.eval()
    return score_model


def enhance_audio(score_model, raw_wav, device, window):
    """Enhance un signal audio avec SGMSE+ (sampling complet, 30 pas)."""
    score_model.eval()
    with torch.no_grad():
        raw = raw_wav.unsqueeze(0).to(device)
        norm = torch.clamp(raw.abs().amax(dim=-1, keepdim=True), min=1e-8)
        y = raw / norm

        Y = spec_fwd(stft(y, window)).unsqueeze(1)
        T_orig = Y.shape[-1]
        Yp = pad_spec(Y)

        x_hat = score_model.enhance(Yp)

        Xh = x_hat[:, :, :, :T_orig].squeeze(1)
        Xh = spec_back(Xh)
        enh = istft(Xh, window, length=y.shape[-1]) * norm
        return enh.squeeze(0)


def load_mono_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR_MODEL:
        wav = torchaudio.transforms.Resample(sr, SR_MODEL)(wav)
    return wav.squeeze(0)


def compute_metrics(clean_np, enhanced_np):
    from pesq import pesq
    from pystoi import stoi as stoi_fn

    ml = min(len(clean_np), len(enhanced_np))
    c, e = clean_np[:ml], enhanced_np[:ml]

    try:
        p = pesq(SR_MODEL, c, e, "wb")
    except Exception:
        p = float("nan")
    try:
        s = stoi_fn(c, e, SR_MODEL)
    except Exception:
        s = float("nan")
    return p, s


def main():
    device = get_device()
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()

    score_model = load_model(device)
    window = get_window(device)
    max_samples = int(MAX_DURATION_S * SR_MODEL)

    # === Test sur fichiers radio reels ===
    print("\n" + "=" * 80)
    print("TEST SUR FICHIERS RADIO REELS")
    print("=" * 80)
    print(f"{'Fichier':<35s} {'PESQ_raw':>9s} {'PESQ_enh':>9s} {'STOI_raw':>9s} {'STOI_enh':>9s}")
    print("-" * 80)

    for noisy_rel, clean_rel in RADIO_FILES:
        noisy_path = os.path.join(RADIO_DIR, noisy_rel)
        clean_path = os.path.join(RADIO_DIR, clean_rel)

        if not os.path.exists(noisy_path):
            print(f"  SKIP: {noisy_rel} (fichier introuvable)")
            continue

        name = os.path.splitext(os.path.basename(noisy_path))[0]
        raw_wav = load_mono_16k(noisy_path)

        # Limiter la duree
        if len(raw_wav) > max_samples:
            raw_wav = raw_wav[:max_samples]

        # Sauvegarder le raw
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_raw.wav"),
                 raw_wav.numpy(), SR_MODEL)

        # Enhance
        print(f"  Enhancement de {name}...", end="", flush=True)
        enhanced = enhance_audio(score_model, raw_wav, device, window)
        enhanced_np = enhanced.cpu().numpy()
        print(" OK")

        # Sauvegarder
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_sgmse_ft.wav"),
                 enhanced_np, SR_MODEL)

        # Metriques si clean disponible
        if os.path.exists(clean_path):
            clean_wav = load_mono_16k(clean_path)
            if len(clean_wav) > max_samples:
                clean_wav = clean_wav[:max_samples]
            clean_np = clean_wav.numpy()
            raw_np = raw_wav.numpy()

            pesq_raw, stoi_raw = compute_metrics(clean_np, raw_np)
            pesq_enh, stoi_enh = compute_metrics(clean_np, enhanced_np)

            print(f"{name:<35s} {pesq_raw:9.3f} {pesq_enh:9.3f} {stoi_raw:9.3f} {stoi_enh:9.3f}")
        else:
            print(f"{name:<35s} {'N/A':>9s} {'N/A':>9s} {'N/A':>9s} {'N/A':>9s}")

    # === Test sur echantillons de validation ===
    print("\n" + "=" * 80)
    print("TEST SUR ECHANTILLONS DE VALIDATION")
    print("=" * 80)
    print(f"{'Echantillon':<25s} {'PESQ':>9s} {'STOI':>9s}")
    print("-" * 50)

    val_raw_dir = os.path.join(DATASET_DIR, "val", "raw")
    val_clean_dir = os.path.join(DATASET_DIR, "val", "clean")
    val_files = sorted(glob.glob(os.path.join(val_raw_dir, "*.wav")))

    pesq_vals, stoi_vals = [], []

    for idx in VAL_INDICES:
        if idx >= len(val_files):
            continue

        raw_path = val_files[idx]
        clean_path = os.path.join(val_clean_dir, os.path.basename(raw_path))
        if not os.path.exists(clean_path):
            continue

        name = os.path.splitext(os.path.basename(raw_path))[0]

        raw_wav, _ = sf.read(raw_path)
        clean_wav, _ = sf.read(clean_path)
        raw_t = torch.tensor(raw_wav, dtype=torch.float32)

        # Enhance
        print(f"  Enhancement de {name}...", end="", flush=True)
        enhanced = enhance_audio(score_model, raw_t, device, window)
        enhanced_np = enhanced.cpu().numpy()
        print(" OK")

        # Sauvegarder
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_raw.wav"), raw_wav, SR_MODEL)
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_sgmse_ft.wav"), enhanced_np, SR_MODEL)

        # Metriques
        p, s = compute_metrics(clean_wav.astype(np.float32), enhanced_np)
        print(f"{name:<25s} {p:9.3f} {s:9.3f}")

        if not np.isnan(p):
            pesq_vals.append(p)
        if not np.isnan(s):
            stoi_vals.append(s)

    if pesq_vals:
        print("-" * 50)
        print(f"{'MOYENNE':<25s} {np.mean(pesq_vals):9.3f} {np.mean(stoi_vals):9.3f}")

    print(f"\nFichiers sauvegardes dans: {OUTPUT_DIR}")
    print(f"Total: {len(os.listdir(OUTPUT_DIR))} fichiers")


if __name__ == "__main__":
    main()
