"""
Pipeline hybride pour amelioration de communications radio militaires.

Etape 1 : Demucs (debruitage waveform, dry/wet mix)
Etape 2 : VoiceFixer (super-resolution + restauration, sortie 44.1 kHz)

Genere des KPIs avant/apres et des spectrogrammes comparatifs.
"""
import os
import sys
import tempfile
import subprocess
import csv
import numpy as np
import soundfile as sf
from scipy.signal import welch
import torch
import matplotlib

sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Configuration ===

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "New", "DatasetRadioCom")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")

# Fichiers a traiter (None = tous les OGG du dossier)
FILES = [
    "Dataset Radio (3).ogg",
]

# Demucs
DEMUCS_DRY_WET = 0.5      # 0 = tout sec (original), 1 = tout wet (denoise)
DEMUCS_MODEL = "htdemucs"  # Modele Demucs a utiliser

# VoiceFixer
VOICEFIXER_MODE = 0        # 0 = normal, 1 = speech, 2 = restauration complete

# Parametres generaux
MAX_DURATION_S = 120       # Duree max par fichier pour tests rapides (None = tout)
OUTPUT_SR = 44100          # SR final (VoiceFixer sort en 44.1 kHz)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# === Utilitaires audio ===

def load_ogg(filepath, max_duration_s=None):
    """Charge un OGG via ffmpeg, retourne (data float32 mono, sr)."""
    tmp_wav = os.path.join(tempfile.gettempdir(), "pipeline_input.wav")
    cmd = ["ffmpeg", "-y", "-i", filepath]
    if max_duration_s:
        cmd += ["-t", str(max_duration_s)]
    cmd += ["-ac", "1", "-ar", "16000", tmp_wav]
    subprocess.run(cmd, capture_output=True, check=True)
    data, sr = sf.read(tmp_wav, dtype="float32")
    os.remove(tmp_wav)
    return data, sr


def compute_kpis(data, sr, label=""):
    """Calcule les KPIs audio standard."""
    # RMS
    rms = np.sqrt(np.mean(data ** 2))
    rms_db = 20 * np.log10(rms + 1e-10)

    # Noise floor (percentile 5 de l'enveloppe RMS par frames)
    frame_len = int(sr * 0.02)
    n_frames = len(data) // frame_len
    if n_frames > 0:
        frames = data[:n_frames * frame_len].reshape(n_frames, frame_len)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        noise_floor_db = 20 * np.log10(np.percentile(frame_rms, 5) + 1e-10)
    else:
        noise_floor_db = rms_db

    # SNR estime
    snr_db = rms_db - noise_floor_db

    # Spectral centroid
    freqs, psd = welch(data, fs=sr, nperseg=min(2048, len(data)))
    spectral_centroid = np.sum(freqs * psd) / (np.sum(psd) + 1e-10)

    # Bandwidth effective (-3dB)
    psd_norm = psd / (np.max(psd) + 1e-10)
    above_3db = freqs[psd_norm > 0.5]
    bandwidth = above_3db[-1] - above_3db[0] if len(above_3db) > 1 else 0

    return {
        "label": label,
        "sr": sr,
        "duration_s": round(len(data) / sr, 2),
        "rms_db": round(rms_db, 1),
        "noise_floor_db": round(noise_floor_db, 1),
        "snr_db": round(snr_db, 1),
        "spectral_centroid_hz": round(spectral_centroid, 0),
        "bandwidth_hz": round(bandwidth, 0),
    }


# === Etape 1 : Demucs ===

def denoise_demucs(data, sr, dry_wet=DEMUCS_DRY_WET):
    """Debruite avec Demucs (Facebook) et applique un dry/wet mix."""
    # Sauver en WAV temporaire
    tmp_in = os.path.join(tempfile.gettempdir(), "demucs_in.wav")
    tmp_out_dir = os.path.join(tempfile.gettempdir(), "demucs_out")
    sf.write(tmp_in, data, sr)

    # Lancer Demucs en mode separation
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", DEMUCS_MODEL,
        "-o", tmp_out_dir,
        tmp_in,
    ]
    if DEVICE == "cuda":
        cmd += ["-d", "cuda"]
    else:
        cmd += ["-d", "cpu"]

    print("    Demucs en cours...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ERREUR Demucs: {result.stderr[:500]}")
        return data

    # Charger le stem vocal
    vocals_path = os.path.join(tmp_out_dir, DEMUCS_MODEL, "demucs_in", "vocals.wav")
    if not os.path.exists(vocals_path):
        # Essayer sans sous-dossier
        for root, dirs, files in os.walk(tmp_out_dir):
            for f in files:
                if f == "vocals.wav":
                    vocals_path = os.path.join(root, f)
                    break

    if not os.path.exists(vocals_path):
        print(f"    ERREUR: vocals.wav introuvable dans {tmp_out_dir}")
        return data

    vocals, _ = sf.read(vocals_path, dtype="float32")

    # Mono si stereo
    if vocals.ndim > 1:
        vocals = np.mean(vocals, axis=1)

    # Ajuster la longueur
    min_len = min(len(data), len(vocals))
    data = data[:min_len]
    vocals = vocals[:min_len]

    # Dry/wet mix
    output = (1.0 - dry_wet) * data + dry_wet * vocals

    # Nettoyage
    try:
        os.remove(tmp_in)
        import shutil
        shutil.rmtree(tmp_out_dir, ignore_errors=True)
    except Exception:
        pass

    return output.astype(np.float32)


# === Etape 2 : VoiceFixer ===

def super_resolve_voicefixer(data, sr, mode=VOICEFIXER_MODE):
    """Applique VoiceFixer pour super-resolution et restauration."""
    from voicefixer import VoiceFixer
    vf = VoiceFixer()

    # VoiceFixer prend un fichier en entree/sortie
    tmp_in = os.path.join(tempfile.gettempdir(), "vf_in.wav")
    tmp_out = os.path.join(tempfile.gettempdir(), "vf_out.wav")

    sf.write(tmp_in, data, sr)

    print(f"    VoiceFixer (mode={mode}) en cours...")
    vf.restore(
        input=tmp_in,
        output=tmp_out,
        cuda=(DEVICE == "cuda"),
        mode=mode,
    )

    # Charger le resultat (44.1 kHz)
    enhanced, new_sr = sf.read(tmp_out, dtype="float32")

    # Mono si stereo
    if enhanced.ndim > 1:
        enhanced = np.mean(enhanced, axis=1)

    # Nettoyage
    try:
        os.remove(tmp_in)
        os.remove(tmp_out)
    except Exception:
        pass

    return enhanced, new_sr


# === Visualisation ===

def plot_comparison(original, denoised, enhanced, sr_orig, sr_enhanced, output_path, title=""):
    """Spectrogrammes comparatifs 3 etapes."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Original
    axes[0].specgram(original, NFFT=2048, Fs=sr_orig, noverlap=1024, cmap="inferno")
    axes[0].set_title(f"1. Original ({sr_orig} Hz)")
    axes[0].set_ylabel("Freq (Hz)")
    axes[0].set_ylim(0, sr_orig // 2)

    # Apres Demucs
    axes[1].specgram(denoised, NFFT=2048, Fs=sr_orig, noverlap=1024, cmap="inferno")
    axes[1].set_title(f"2. Apres Demucs (denoise, {sr_orig} Hz)")
    axes[1].set_ylabel("Freq (Hz)")
    axes[1].set_ylim(0, sr_orig // 2)

    # Apres VoiceFixer
    axes[2].specgram(enhanced, NFFT=2048, Fs=sr_enhanced, noverlap=1024, cmap="inferno")
    axes[2].set_title(f"3. Apres VoiceFixer (super-resolution, {sr_enhanced} Hz)")
    axes[2].set_ylabel("Freq (Hz)")
    axes[2].set_xlabel("Temps (s)")
    axes[2].set_ylim(0, sr_enhanced // 2)

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_psd_comparison(original, denoised, enhanced, sr_orig, sr_enhanced, output_path):
    """PSD comparatif des 3 etapes."""
    fig, ax = plt.subplots(figsize=(12, 5))

    for data, sr, label, color in [
        (original, sr_orig, "Original", "gray"),
        (denoised, sr_orig, "Demucs", "blue"),
        (enhanced, sr_enhanced, "VoiceFixer", "green"),
    ]:
        freqs, psd = welch(data, fs=sr, nperseg=min(4096, len(data)))
        ax.semilogy(freqs, psd, label=label, color=color, alpha=0.8, linewidth=1)

    ax.set_xlabel("Frequence (Hz)")
    ax.set_ylabel("PSD (V^2/Hz)")
    ax.set_title("Densite Spectrale de Puissance - Comparaison")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, sr_enhanced // 2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


# === Pipeline principal ===

def process_file(filepath, filename):
    """Traite un fichier avec le pipeline complet."""
    print(f"\n{'=' * 60}")
    print(f"  Fichier : {filename}")
    print(f"{'=' * 60}")

    # Charger
    print("  [1/4] Chargement...")
    original, sr_orig = load_ogg(filepath, max_duration_s=MAX_DURATION_S)
    print(f"    -> {len(original)/sr_orig:.1f}s, {sr_orig} Hz")

    # KPI original
    kpi_orig = compute_kpis(original, sr_orig, label="original")

    # Etape 1 : Demucs
    print("  [2/4] Debruitage Demucs...")
    denoised = denoise_demucs(original, sr_orig)
    kpi_denoised = compute_kpis(denoised, sr_orig, label="demucs")

    # Etape 2 : VoiceFixer
    print("  [3/4] Super-resolution VoiceFixer...")
    enhanced, sr_enhanced = super_resolve_voicefixer(denoised, sr_orig)
    kpi_enhanced = compute_kpis(enhanced, sr_enhanced, label="voicefixer")

    # Sauvegarder les WAV
    base_name = os.path.splitext(filename)[0]
    out_denoised = os.path.join(OUTPUT_DIR, f"{base_name}_1_demucs.wav")
    out_enhanced = os.path.join(OUTPUT_DIR, f"{base_name}_2_voicefixer.wav")
    sf.write(out_denoised, denoised, sr_orig)
    sf.write(out_enhanced, enhanced, sr_enhanced)

    # Graphiques
    print("  [4/4] Generation des graphiques...")
    plot_comparison(
        original, denoised, enhanced, sr_orig, sr_enhanced,
        os.path.join(OUTPUT_DIR, f"{base_name}_spectrograms.png"),
        title=base_name,
    )
    plot_psd_comparison(
        original, denoised, enhanced, sr_orig, sr_enhanced,
        os.path.join(OUTPUT_DIR, f"{base_name}_psd.png"),
    )

    # Afficher les KPIs
    print(f"\n  {'KPI':<25} {'Original':>12} {'Demucs':>12} {'VoiceFixer':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    for key in ["sr", "rms_db", "noise_floor_db", "snr_db", "spectral_centroid_hz", "bandwidth_hz"]:
        v1 = kpi_orig[key]
        v2 = kpi_denoised[key]
        v3 = kpi_enhanced[key]
        print(f"  {key:<25} {v1:>12} {v2:>12} {v3:>12}")

    return kpi_orig, kpi_denoised, kpi_enhanced


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  PIPELINE HYBRIDE - Communications Radio Militaires")
    print("  Etape 1: Demucs (debruitage)")
    print("  Etape 2: VoiceFixer (super-resolution 44.1 kHz)")
    print("=" * 60)
    print(f"  Device: {DEVICE}")
    print(f"  Demucs dry/wet: {DEMUCS_DRY_WET}")
    print(f"  VoiceFixer mode: {VOICEFIXER_MODE}")
    print(f"  Max duree: {MAX_DURATION_S}s" if MAX_DURATION_S else "  Max duree: illimite")

    # Lister les fichiers
    if FILES:
        filenames = FILES
    else:
        filenames = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".ogg")])

    print(f"  Fichiers: {len(filenames)}\n")

    all_kpis = []
    for filename in filenames:
        filepath = os.path.join(INPUT_DIR, filename)
        if not os.path.exists(filepath):
            print(f"  ERREUR: {filepath} introuvable")
            continue
        kpi_orig, kpi_denoised, kpi_enhanced = process_file(filepath, filename)
        all_kpis.append({
            "filename": filename,
            **{f"orig_{k}": v for k, v in kpi_orig.items() if k != "label"},
            **{f"demucs_{k}": v for k, v in kpi_denoised.items() if k != "label"},
            **{f"vf_{k}": v for k, v in kpi_enhanced.items() if k != "label"},
        })

    # Export CSV
    if all_kpis:
        csv_path = os.path.join(OUTPUT_DIR, "kpi_report.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_kpis[0].keys())
            writer.writeheader()
            writer.writerows(all_kpis)
        print(f"\nRapport CSV: {csv_path}")

    print(f"\nResultats dans: {OUTPUT_DIR}")
    print("Pipeline termine !")


if __name__ == "__main__":
    main()
