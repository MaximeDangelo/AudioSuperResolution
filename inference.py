"""
Inference : applique le pipeline hybride complet sur des fichiers audio.

Pipeline :
  1. Demucs (debruitage waveform, 100% denoise)
  2. Resample 44.1 kHz
  3. SpectralResUNet (fine-tune radio : denoise spectral + reconstruction HF)
  4. MetricGAN+ (polissage PESQ, SpeechBrain, optionnel)

Note : VoiceFixer a ete retire du pipeline car il degradait
les signaux radio (ecrasement des frequences, artefacts vocaux).

Peut traiter :
- Les fichiers OGG du projet original (../New/DatasetRadioCom/)
- N'importe quel fichier WAV/OGG/MP3
"""
import os
import sys
import subprocess
import tempfile
import shutil
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import matplotlib

sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, resample_poly
from math import gcd

# Metriques audio
from pystoi import stoi as compute_stoi
from mir_eval.separation import bss_eval_sources

try:
    from pesq import pesq as compute_pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False

from train import SpectralResUNet, SR, N_FFT, HOP_LENGTH, WIN_LENGTH, get_device

# === Configuration ===

CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "best_model.pt")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

INPUT_FILES = [
    os.path.join("Dataset Radio (2)", "01__kwahmah_atc001.flac"),
    os.path.join("Dataset Radio (2)", "07-wahmah_heathrow-air-traffic-control.flac"),
    os.path.join("Dataset Radio (2)", "08__kwahmah_hong-kong-air-traffic-control.flac"),
]

# Pipeline : activer/desactiver les etapes
DENOISE_ENGINE = "demucs"  # "demucs" | "sepformer" | "none"
USE_VOICEFIXER = False
USE_FINETUNE = True  # Utilise le modele fine-tune si disponible
USE_METRICGAN = True  # Polissage PESQ (bug speechbrain/huggingface_hub corrige)

# Demucs
DEMUCS_DRY_WET = 0.0  # 100% Demucs, 0% original
DEMUCS_MODEL = "htdemucs"

# SepFormer (alternative a Demucs)
SEPFORMER_MODEL = "speechbrain/sepformer-dns4-16k-enhancement"
SEPFORMER_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "sepformer")

# MetricGAN+ (polissage PESQ)
METRICGAN_MODEL = "speechbrain/metricgan-plus-voicebank"
METRICGAN_SAVEDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pretrained_models", "metricgan")

# VoiceFixer
VOICEFIXER_MODE = 0

# Dry/wet mix pour le debruitage (Demucs ou SepFormer)
DENOISE_DRY_WET = 0.0  # 100% denoise

# Duree max pour test rapide (None = tout le fichier)
MAX_DURATION_S = 120

DEVICE = get_device()


# === Fonctions ===

def load_audio(filepath, target_sr=16000):
    """Charge un fichier audio via ffmpeg, mono au SR cible."""
    tmp_wav = os.path.join(tempfile.gettempdir(), "sr_inference_input.wav")
    cmd = ["ffmpeg", "-y", "-i", filepath]
    if MAX_DURATION_S:
        cmd += ["-t", str(MAX_DURATION_S)]
    cmd += ["-ar", str(target_sr), "-ac", "1", tmp_wav]
    subprocess.run(cmd, capture_output=True, check=True)
    data, sr = sf.read(tmp_wav, dtype="float32")
    os.remove(tmp_wav)
    return data, sr


def denoise_demucs(data, sr):
    """Etape 1 : Demucs debruitage avec dry/wet mix."""
    tmp_in = os.path.join(tempfile.gettempdir(), "demucs_in.wav")
    tmp_out_dir = os.path.join(tempfile.gettempdir(), "demucs_out")
    sf.write(tmp_in, data, sr)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", DEMUCS_MODEL,
        "-o", tmp_out_dir,
        tmp_in,
    ]
    if torch.cuda.is_available():
        cmd += ["-d", "cuda"]
    else:
        cmd += ["-d", "cpu"]

    print("    Demucs en cours...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERREUR Demucs: {result.stderr[:500]}")
        return data

    # Trouver vocals.wav
    vocals_path = None
    for root, dirs, files in os.walk(tmp_out_dir):
        for f in files:
            if f == "vocals.wav":
                vocals_path = os.path.join(root, f)
                break
        if vocals_path:
            break

    if not vocals_path:
        print(f"    ERREUR: vocals.wav introuvable")
        return data

    vocals, _ = sf.read(vocals_path, dtype="float32")
    if vocals.ndim > 1:
        vocals = np.mean(vocals, axis=1)

    min_len = min(len(data), len(vocals))
    data = data[:min_len]
    vocals = vocals[:min_len]

    output = (1.0 - DENOISE_DRY_WET) * data + DENOISE_DRY_WET * vocals

    try:
        os.remove(tmp_in)
        shutil.rmtree(tmp_out_dir, ignore_errors=True)
    except Exception:
        pass

    return output.astype(np.float32)


_sepformer_model = None

def denoise_sepformer(data, sr):
    """Etape 1 (alt) : SepFormer debruitage par Transformers (SpeechBrain, 16kHz)."""
    global _sepformer_model
    from speechbrain.inference.separation import SepformerSeparation

    print("    SepFormer en cours...")
    if _sepformer_model is None:
        _sepformer_model = SepformerSeparation.from_hparams(
            source=SEPFORMER_MODEL,
            savedir=SEPFORMER_SAVEDIR,
            run_opts={"device": str(DEVICE)},
        )

    # SepFormer attend du 16kHz
    if sr != 16000:
        g = gcd(sr, 16000)
        data_16k = resample_poly(data, 16000 // g, sr // g).astype(np.float32)
    else:
        data_16k = data

    # Inference : [1, T] -> [1, T, sources]
    mix = torch.from_numpy(data_16k).float().unsqueeze(0)
    est_sources = _sepformer_model.separate_batch(mix)
    enhanced = est_sources[:, :, 0].squeeze().cpu().numpy()

    # Resample vers le SR original si necessaire
    if sr != 16000:
        g = gcd(16000, sr)
        enhanced = resample_poly(enhanced, sr // g, 16000 // g).astype(np.float32)

    # Dry/wet mix
    min_len = min(len(data), len(enhanced))
    data = data[:min_len]
    enhanced = enhanced[:min_len]
    output = (1.0 - DENOISE_DRY_WET) * data + DENOISE_DRY_WET * enhanced

    return output.astype(np.float32)


def super_resolve_voicefixer(data, sr):
    """Etape 2 : VoiceFixer super-resolution."""
    from voicefixer import VoiceFixer
    vf = VoiceFixer()

    tmp_in = os.path.join(tempfile.gettempdir(), "vf_in.wav")
    tmp_out = os.path.join(tempfile.gettempdir(), "vf_out.wav")
    sf.write(tmp_in, data, sr)

    print(f"    VoiceFixer (mode={VOICEFIXER_MODE}) en cours...")
    vf.restore(input=tmp_in, output=tmp_out, cuda=torch.cuda.is_available(), mode=VOICEFIXER_MODE)

    enhanced, new_sr = sf.read(tmp_out, dtype="float32")
    if enhanced.ndim > 1:
        enhanced = np.mean(enhanced, axis=1)

    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except Exception:
        pass

    return enhanced, new_sr


def enhance_finetune(model, data, device, chunk_size=SR * 3, overlap=SR // 2):
    """Etape 3 : Applique le modele fine-tune par chunks avec overlap-add."""
    model.eval()
    n = len(data)
    output = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)

    fade_len = overlap
    fade_in = np.linspace(0, 1, fade_len, dtype=np.float32)
    fade_out = np.linspace(1, 0, fade_len, dtype=np.float32)

    pos = 0
    with torch.no_grad():
        while pos < n:
            end = min(pos + chunk_size, n)
            chunk = data[pos:end]

            if len(chunk) < chunk_size:
                chunk = np.pad(chunk, (0, chunk_size - len(chunk)))

            x = torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            y = model(x).squeeze().cpu().numpy()
            y = y[:end - pos]

            w = np.ones(len(y), dtype=np.float32)
            if pos > 0:
                actual_fade = min(fade_len, len(y))
                w[:actual_fade] = fade_in[:actual_fade]
            if end < n:
                actual_fade = min(fade_len, len(y))
                w[-actual_fade:] = fade_out[:actual_fade]

            output[pos:end] += y * w
            weight[pos:end] += w

            pos += chunk_size - overlap

    weight = np.maximum(weight, 1e-8)
    return output / weight


_metricgan_model = None

def polish_metricgan(data, sr):
    """Etape finale : MetricGAN+ polissage optimise PESQ (SpeechBrain, 16kHz).

    MetricGAN+ est entraine a optimiser directement le score PESQ.
    Il travaille a 16 kHz en interne. Pour du 44.1 kHz, on resample
    vers 16 kHz, on applique l'enhancement, puis on resample vers le SR
    d'origine. Note : les frequences au-dessus de 8 kHz ne sont pas
    traitees par MetricGAN+ (limite du 16 kHz).
    """
    global _metricgan_model
    from speechbrain.inference.enhancement import SpectralMaskEnhancement

    print("    MetricGAN+ en cours...")
    if _metricgan_model is None:
        # Patch: huggingface_hub >= 0.24 a supprime use_auth_token
        import huggingface_hub
        _orig_download = huggingface_hub.hf_hub_download
        def _patched_download(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return _orig_download(*args, **kwargs)
        huggingface_hub.hf_hub_download = _patched_download
        try:
            _metricgan_model = SpectralMaskEnhancement.from_hparams(
                source=METRICGAN_MODEL,
                savedir=METRICGAN_SAVEDIR,
                run_opts={"device": str(DEVICE)},
            )
        finally:
            huggingface_hub.hf_hub_download = _orig_download

    # MetricGAN+ travaille a 16kHz
    target_sr = 16000
    if sr != target_sr:
        g = gcd(sr, target_sr)
        data_16k = resample_poly(data, target_sr // g, sr // g).astype(np.float32)
    else:
        data_16k = data

    # Inference
    noisy = torch.from_numpy(data_16k).float().unsqueeze(0)
    lengths = torch.tensor([1.0])
    enhanced = _metricgan_model.enhance_batch(noisy, lengths)
    enhanced_np = enhanced.squeeze().cpu().numpy()

    # Resample vers le SR original
    if sr != target_sr:
        g = gcd(target_sr, sr)
        enhanced_np = resample_poly(enhanced_np, sr // g, target_sr // g).astype(np.float32)

    # Aligner la longueur
    min_len = min(len(data), len(enhanced_np))
    return enhanced_np[:min_len]


def compute_kpis(data, sr, label=""):
    """Calcule les KPIs audio de base (sans reference)."""
    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    frame_len = int(sr * 0.02)
    n_frames = len(data) // frame_len
    if n_frames > 0:
        frames = data[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        noise_floor_db = 20 * np.log10(np.percentile(frame_rms, 5) + 1e-10)
    else:
        noise_floor_db = rms_db

    snr_db = rms_db - noise_floor_db

    freqs, psd = welch(data, fs=sr, nperseg=min(2048, len(data)))
    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)

    psd_norm = psd / (np.max(psd) + 1e-10)
    above_3db = freqs[psd_norm > 0.5]
    bandwidth = above_3db[-1] - above_3db[0] if len(above_3db) > 1 else 0

    return {
        "label": label, "sr": sr,
        "rms_db": round(rms_db, 1),
        "snr_db": round(snr_db, 1),
        "spectral_centroid_hz": round(spectral_centroid, 0),
        "bandwidth_hz": round(bandwidth, 0),
    }


def compute_snr(reference, enhanced):
    """Calcule le SNR (Signal-to-Noise Ratio) en dB."""
    noise = enhanced - reference
    power_signal = np.mean(reference ** 2)
    power_noise = np.mean(noise ** 2)
    return 10 * np.log10(power_signal / (power_noise + 1e-10))


def compute_lsd(reference, enhanced, sr, n_fft=2048, hop_length=512):
    """Calcule le LSD (Log-Spectral Distance) en dB.

    Mesure la distance spectrale logarithmique, particulierement
    sensible aux hautes frequences (objectif super-resolution).
    Plus c'est bas, mieux c'est.
    """
    def log_power_spectrum(x):
        stft = np.abs(np.fft.rfft(
            x.reshape(-1, n_fft) * np.hanning(n_fft), axis=1
        )) ** 2
        return np.log10(stft + 1e-10)

    # Decouper en frames
    n_frames = (min(len(reference), len(enhanced)) - n_fft) // hop_length
    if n_frames < 1:
        return 0.0

    ref_frames = np.array([reference[i*hop_length:i*hop_length+n_fft] for i in range(n_frames)])
    enh_frames = np.array([enhanced[i*hop_length:i*hop_length+n_fft] for i in range(n_frames)])

    window = np.hanning(n_fft)
    ref_spec = np.log10(np.abs(np.fft.rfft(ref_frames * window, axis=1)) ** 2 + 1e-10)
    enh_spec = np.log10(np.abs(np.fft.rfft(enh_frames * window, axis=1)) ** 2 + 1e-10)

    lsd = np.sqrt(np.mean((ref_spec - enh_spec) ** 2))
    return lsd


def compute_all_metrics(reference, enhanced, sr_ref, sr_enh):
    """Calcule les 5 metriques de performance.

    Metriques :
    - PESQ : qualite percue (1 a 4.5, ITU-T P.862)
    - STOI : intelligibilite (0 a 1)
    - SDR  : Signal-to-Distortion Ratio (dB)
    - LSD  : Log-Spectral Distance (dB, plus bas = mieux)
    - SNR  : Signal-to-Noise Ratio (dB)

    Les signaux sont resamplees a un SR commun si necessaire.
    PESQ necessite 16 kHz ou 8 kHz.
    """
    # Aligner les longueurs
    min_len = min(len(reference), len(enhanced))
    ref = reference[:min_len].copy()
    enh = enhanced[:min_len].copy()

    metrics = {}

    # SNR
    metrics["SNR (dB)"] = round(compute_snr(ref, enh), 2)

    # LSD
    metrics["LSD (dB)"] = round(compute_lsd(ref, enh, sr_ref), 4)

    # STOI (necessite meme SR, fonctionne a n'importe quel SR)
    try:
        stoi_val = compute_stoi(ref, enh, sr_ref, extended=False)
        metrics["STOI"] = round(stoi_val, 4)
    except Exception as e:
        metrics["STOI"] = f"erreur: {e}"

    # SDR via mir_eval
    try:
        sdr, _, _, _ = bss_eval_sources(
            ref.reshape(1, -1), enh.reshape(1, -1), compute_permutation=False
        )
        metrics["SDR (dB)"] = round(float(sdr[0]), 2)
    except Exception as e:
        metrics["SDR (dB)"] = f"erreur: {e}"

    # PESQ (necessite 16 kHz ou 8 kHz)
    if PESQ_AVAILABLE:
        try:
            pesq_sr = 16000
            if sr_ref != pesq_sr:
                g = gcd(sr_ref, pesq_sr)
                ref_16k = resample_poly(ref, pesq_sr // g, sr_ref // g).astype(np.float32)
                enh_16k = resample_poly(enh, pesq_sr // g, sr_ref // g).astype(np.float32)
            else:
                ref_16k, enh_16k = ref, enh
            pesq_val = compute_pesq(pesq_sr, ref_16k, enh_16k, "wb")
            metrics["PESQ"] = round(pesq_val, 4)
        except Exception as e:
            metrics["PESQ"] = f"erreur: {e}"
    else:
        metrics["PESQ"] = "non disponible (pip install pesq)"

    return metrics


def plot_result(stages, output_path, title=""):
    """Spectrogrammes multi-etapes."""
    n = len(stages)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n))
    if n == 1:
        axes = [axes]

    for ax, (data, sr, label) in zip(axes, stages):
        ax.specgram(data, NFFT=2048, Fs=sr, noverlap=1024, cmap="inferno")
        ax.set_title(label)
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, sr // 2)

    axes[-1].set_xlabel("Temps (s)")
    if title:
        fig.suptitle(title, fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  INFERENCE - Pipeline Hybride Radio Militaire")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Debruitage: {DENOISE_ENGINE} (dry/wet={DENOISE_DRY_WET})")
    print(f"  VoiceFixer: {'ON' if USE_VOICEFIXER else 'OFF'} (mode={VOICEFIXER_MODE})")
    print(f"  MetricGAN+: {'ON' if USE_METRICGAN else 'OFF'}")

    # Charger le modele fine-tune si disponible
    finetune_model = None
    if USE_FINETUNE and os.path.exists(CHECKPOINT):
        finetune_model = SpectralResUNet().to(DEVICE)
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        finetune_model.load_state_dict(ckpt["model_state_dict"])
        finetune_model.eval()
        print(f"  Fine-tune: ON (epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f})")
    else:
        print(f"  Fine-tune: OFF (pas de checkpoint)")

    print()

    for filepath in INPUT_FILES:
        abs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
        if not os.path.exists(abs_path):
            print(f"  ERREUR: {abs_path} introuvable")
            continue

        filename = os.path.basename(filepath)
        base_name = os.path.splitext(filename)[0]
        print(f"Traitement: {filename}")

        # Charger a 16kHz (pour Demucs)
        data, sr = load_audio(abs_path, target_sr=16000)
        print(f"  Duree: {len(data)/sr:.1f}s | SR: {sr}Hz")

        stages = [(data.copy(), sr, f"1. Original ({sr} Hz)")]

        # Etape 1 : Debruitage (Demucs ou SepFormer)
        if DENOISE_ENGINE == "demucs":
            print("  [Demucs] Debruitage...")
            data = denoise_demucs(data, sr)
            stages.append((data.copy(), sr, f"2. Apres Demucs ({sr} Hz)"))
        elif DENOISE_ENGINE == "sepformer":
            print("  [SepFormer] Debruitage...")
            data = denoise_sepformer(data, sr)
            stages.append((data.copy(), sr, f"2. Apres SepFormer ({sr} Hz)"))

        # Etape 2 : VoiceFixer
        current_sr = sr
        if USE_VOICEFIXER:
            print("  [VoiceFixer] Super-resolution...")
            data, current_sr = super_resolve_voicefixer(data, sr)
            stages.append((data.copy(), current_sr, f"3. Apres VoiceFixer ({current_sr} Hz)"))

        # Etape 3 : Fine-tune
        if finetune_model is not None:
            print("  [Fine-tune] Amelioration radio...")
            if current_sr != SR:
                g = gcd(current_sr, SR)
                data = resample_poly(data, SR // g, current_sr // g).astype(np.float32)
                current_sr = SR
            data = enhance_finetune(finetune_model, data, DEVICE)
            stages.append((data.copy(), current_sr, f"4. Apres Fine-tune ({current_sr} Hz)"))

        # Etape 4 : MetricGAN+ polissage
        if USE_METRICGAN:
            print("  [MetricGAN+] Polissage PESQ...")
            data = polish_metricgan(data, current_sr)
            stages.append((data.copy(), current_sr, f"5. Apres MetricGAN+ ({current_sr} Hz)"))

        # Sauvegarder
        out_wav = os.path.join(OUTPUT_DIR, f"{base_name}_enhanced.wav")
        sf.write(out_wav, data, current_sr)
        print(f"  -> {out_wav}")

        # KPIs
        kpi_orig = compute_kpis(stages[0][0], stages[0][1], "original")
        kpi_final = compute_kpis(data, current_sr, "enhanced")
        print(f"\n  {'KPI':<25} {'Original':>12} {'Enhanced':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        for key in ["sr", "rms_db", "snr_db", "spectral_centroid_hz", "bandwidth_hz"]:
            print(f"  {key:<25} {kpi_orig.get(key, ''):>12} {kpi_final.get(key, ''):>12}")

        # Graphiques
        plot_path = os.path.join(OUTPUT_DIR, f"{base_name}_pipeline.png")
        plot_result(stages, plot_path, title=base_name)
        print(f"  -> {plot_path}\n")

    print("Termine !\n")


def evaluate_on_validation(model, device, n_samples=20):
    """Evalue le modele fine-tune sur le dataset de validation.

    Calcule les 5 metriques (PESQ, STOI, SDR, LSD, SNR) sur des paires
    (raw, clean) du dataset de validation, avec et sans le modele.
    """
    import csv as csv_mod
    val_raw_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "val", "raw")
    val_clean_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "val", "clean")

    if not os.path.isdir(val_raw_dir):
        print("  Pas de dataset de validation trouve.")
        return

    raw_files = sorted([f for f in os.listdir(val_raw_dir) if f.endswith(".wav")])
    if not raw_files:
        print("  Pas de fichiers de validation.")
        return

    # Limiter le nombre d'echantillons
    rng = np.random.default_rng(42)
    if len(raw_files) > n_samples:
        indices = rng.choice(len(raw_files), n_samples, replace=False)
        raw_files = [raw_files[i] for i in sorted(indices)]

    print(f"\n{'=' * 60}")
    print(f"  EVALUATION SUR DATASET DE VALIDATION ({len(raw_files)} paires)")
    print(f"{'=' * 60}")
    print(f"  Metriques : PESQ, STOI, SDR, LSD, SNR")
    print()

    all_metrics_before = []
    all_metrics_after = []

    model.eval()
    for i, fname in enumerate(raw_files):
        raw_path = os.path.join(val_raw_dir, fname)
        clean_path = os.path.join(val_clean_dir, fname.replace("raw", "clean") if "raw" in fname else fname)

        # Le nom est identique dans raw/ et clean/
        if not os.path.exists(clean_path):
            clean_path = os.path.join(val_clean_dir, fname)
        if not os.path.exists(clean_path):
            continue

        raw_data, sr_raw = sf.read(raw_path, dtype="float32")
        clean_data, sr_clean = sf.read(clean_path, dtype="float32")

        # Aligner les longueurs
        min_len = min(len(raw_data), len(clean_data))
        raw_data = raw_data[:min_len]
        clean_data = clean_data[:min_len]

        # Appliquer le modele
        with torch.no_grad():
            enhanced = enhance_finetune(model, raw_data, device,
                                        chunk_size=SR * 3, overlap=SR // 2)
        enhanced = enhanced[:min_len]

        # Metriques AVANT (raw vs clean)
        m_before = compute_all_metrics(clean_data, raw_data, sr_raw, sr_raw)
        all_metrics_before.append(m_before)

        # Metriques APRES (enhanced vs clean)
        m_after = compute_all_metrics(clean_data, enhanced, sr_raw, sr_raw)
        all_metrics_after.append(m_after)

        if (i + 1) % 5 == 0 or i == 0:
            print(f"  [{i+1}/{len(raw_files)}] {fname}")

    if not all_metrics_before:
        print("  Aucune paire evaluee.")
        return

    # Moyenner les metriques
    metric_keys = ["SNR (dB)", "LSD (dB)", "STOI", "SDR (dB)", "PESQ"]
    print(f"\n  {'Metrique':<20} {'Avant (raw)':>15} {'Apres (model)':>15} {'Delta':>10}")
    print(f"  {'-'*20} {'-'*15} {'-'*15} {'-'*10}")

    results_csv = []
    for key in metric_keys:
        vals_before = [m[key] for m in all_metrics_before if isinstance(m.get(key), (int, float))]
        vals_after = [m[key] for m in all_metrics_after if isinstance(m.get(key), (int, float))]

        if vals_before and vals_after:
            avg_b = np.mean(vals_before)
            avg_a = np.mean(vals_after)
            delta = avg_a - avg_b
            # Pour LSD, negatif = amelioration
            sign = "+" if delta > 0 else ""
            print(f"  {key:<20} {avg_b:>15.4f} {avg_a:>15.4f} {sign}{delta:>9.4f}")
            results_csv.append({"metric": key, "before": round(avg_b, 4),
                                "after": round(avg_a, 4), "delta": round(delta, 4)})
        else:
            val = all_metrics_before[0].get(key, "N/A") if all_metrics_before else "N/A"
            print(f"  {key:<20} {'N/A':>15} {'N/A':>15} {'N/A':>10}")

    # Sauvegarder les resultats en CSV
    csv_path = os.path.join(OUTPUT_DIR, "evaluation_metrics.csv")
    if results_csv:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=["metric", "before", "after", "delta"])
            writer.writeheader()
            writer.writerows(results_csv)
        print(f"\n  -> Resultats sauvegardes dans {csv_path}")

    print()


if __name__ == "__main__":
    main()

    # Evaluation sur le dataset de validation si le modele est disponible
    if os.path.exists(CHECKPOINT):
        model = SpectralResUNet().to(DEVICE)
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        evaluate_on_validation(model, DEVICE, n_samples=20)
