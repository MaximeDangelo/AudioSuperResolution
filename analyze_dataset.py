"""
Analyse détaillée des paires clean/noisy du dataset radio.
Génère des spectrogrammes comparatifs et des métriques.
"""

import os
import numpy as np
import torchaudio
import torch
from scipy import signal
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False
    print("[!] pip install matplotlib pour les spectrogrammes visuels")


DATASET_DIR = Path("os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset Radio (2)")")
OUTPUT_DIR = DATASET_DIR / "analysis"


# Paires noisy/clean identifiées
PAIRS = [
    ("flac/02__kwahmah_atc002.flac", "Test1/02__kwahmah_atc002_clean.wav"),
    ("flac/03_kwahmah_atc003.flac", "Test1/03_kwahmah_atc003_clean.wav"),
    ("flac/05__kwahmah_atc005.flac", "Test1/05__kwahmah_atc005_clean.wav"),
    ("flac/06_kwahmah_atc006.flac", "Test1/06_kwahmah_atc006_clean.wav"),
    ("flac/07-wahmah_heathrow-air-traffic-control.flac", "Test1/07-wahmah_heathrow-air-traffic-control_clean.wav"),
    ("flac/08__kwahmah_hong-kong-air-traffic-control.flac", "Test1/08__kwahmah_hong-kong-air-traffic-control_clean.wav"),
]

# Paires avec fichiers clean WAV à la racine
PAIRS_ROOT = [
    ("flac/02__kwahmah_atc002.flac", "02__kwahmah_atc002-Clean.wav"),
    ("flac/03_kwahmah_atc003.flac", "03_kwahmah_atc003-Clean.wav"),
    ("flac/07-wahmah_heathrow-air-traffic-control.flac", "07-wahmah_heathrow-air-traffic-control-Clean.wav"),
]


def load_mono(path, target_sr=16000, max_seconds=30):
    """Charge un fichier audio en mono et resample, limité à max_seconds."""
    info = torchaudio.info(str(path))
    sr_orig = info.sample_rate
    max_frames = int(max_seconds * sr_orig)
    num_frames = min(info.num_frames, max_frames)
    wav, sr = torchaudio.load(str(path), num_frames=num_frames)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav.squeeze().numpy(), target_sr, sr


def analyze_pair(noisy_path, clean_path, name, target_sr=16000):
    """Analyse complète d'une paire noisy/clean."""
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")

    noisy, sr, orig_sr_noisy = load_mono(noisy_path, target_sr)
    clean, _, orig_sr_clean = load_mono(clean_path, target_sr)

    # Aligner les longueurs
    min_len = min(len(noisy), len(clean))
    noisy = noisy[:min_len]
    clean = clean[:min_len]
    duration = min_len / sr

    print(f"\n  Fichier noisy: {noisy_path.name} (sr original: {orig_sr_noisy} Hz)")
    print(f"  Fichier clean: {clean_path.name} (sr original: {orig_sr_clean} Hz)")
    print(f"  Durée: {duration:.1f}s (analysé à {sr} Hz)")

    # --- Métriques temporelles ---
    print(f"\n  --- Domaine temporel ---")
    print(f"  {'':20s} {'Noisy':>12s} {'Clean':>12s} {'Diff':>12s}")

    rms_n = np.sqrt(np.mean(noisy**2))
    rms_c = np.sqrt(np.mean(clean**2))
    print(f"  {'RMS':20s} {rms_n:12.5f} {rms_c:12.5f} {rms_n-rms_c:+12.5f}")

    peak_n = np.max(np.abs(noisy))
    peak_c = np.max(np.abs(clean))
    print(f"  {'Peak':20s} {peak_n:12.5f} {peak_c:12.5f} {peak_n-peak_c:+12.5f}")

    crest_n = 20*np.log10(peak_n / (rms_n + 1e-10))
    crest_c = 20*np.log10(peak_c / (rms_c + 1e-10))
    print(f"  {'Crest factor (dB)':20s} {crest_n:12.2f} {crest_c:12.2f} {crest_n-crest_c:+12.2f}")

    # --- Corrélation ---
    corr = np.corrcoef(noisy, clean)[0, 1]
    print(f"\n  Corrélation noisy/clean: {corr:.4f}")

    # --- Différence signal ---
    diff = noisy - clean
    rms_diff = np.sqrt(np.mean(diff**2))
    snr = 20 * np.log10(rms_c / (rms_diff + 1e-10))
    print(f"  RMS de la différence: {rms_diff:.5f}")
    print(f"  SNR estimé (clean/diff): {snr:.1f} dB")

    # --- Analyse spectrale ---
    print(f"\n  --- Domaine fréquentiel ---")

    nperseg = min(2048, min_len)
    freqs_n, psd_n = signal.welch(noisy, sr, nperseg=nperseg)
    freqs_c, psd_c = signal.welch(clean, sr, nperseg=nperseg)
    freqs_d, psd_d = signal.welch(diff, sr, nperseg=nperseg)

    psd_n_db = 10 * np.log10(psd_n + 1e-12)
    psd_c_db = 10 * np.log10(psd_c + 1e-12)
    psd_d_db = 10 * np.log10(psd_d + 1e-12)

    # Bande passante effective
    for label, psd_db, freqs in [("Noisy", psd_n_db, freqs_n), ("Clean", psd_c_db, freqs_c)]:
        active = freqs[psd_db > psd_db.max() - 20]
        print(f"  Bandwidth {label} (-20dB): {active.min():.0f} - {active.max():.0f} Hz")

    # Énergie par bande
    bands = [(0, 300, "Sub-300Hz"), (300, 1000, "300-1kHz"), (1000, 3000, "1k-3kHz"),
             (3000, 5000, "3k-5kHz"), (5000, 8000, "5k-8kHz")]

    print(f"\n  Énergie par bande (dB):")
    print(f"  {'Bande':15s} {'Noisy':>10s} {'Clean':>10s} {'Diff':>10s}")
    for flo, fhi, label in bands:
        if fhi > sr/2:
            continue
        mask = (freqs_n >= flo) & (freqs_n < fhi)
        if mask.sum() == 0:
            continue
        e_n = 10 * np.log10(np.mean(psd_n[mask]) + 1e-12)
        e_c = 10 * np.log10(np.mean(psd_c[mask]) + 1e-12)
        e_d = 10 * np.log10(np.mean(psd_d[mask]) + 1e-12)
        print(f"  {label:15s} {e_n:10.1f} {e_c:10.1f} {e_d:10.1f}")

    # --- PESQ / STOI ---
    try:
        from pesq import pesq
        mode = "wb" if sr >= 16000 else "nb"
        pesq_score = pesq(sr, clean, noisy, mode)
        print(f"\n  PESQ (noisy vs clean): {pesq_score:.4f}")
    except Exception as e:
        print(f"\n  PESQ error: {e}")

    try:
        from pystoi import stoi as stoi_fn
        stoi_score = stoi_fn(clean, noisy, sr)
        print(f"  STOI (noisy vs clean): {stoi_score:.4f}")
    except Exception as e:
        print(f"  STOI error: {e}")

    # --- Spectrogrammes ---
    if HAS_PLT:
        fig, axes = plt.subplots(4, 1, figsize=(14, 12))
        fig.suptitle(f"Analyse: {name}", fontsize=14, fontweight="bold")

        # Spectrogramme noisy
        axes[0].set_title("Spectrogramme - NOISY")
        f_spec, t_spec, Sxx_n = signal.spectrogram(noisy, sr, nperseg=512, noverlap=384)
        axes[0].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx_n + 1e-12), shading="gouraud", cmap="inferno", vmin=-80, vmax=-20)
        axes[0].set_ylabel("Fréquence (Hz)")
        axes[0].set_ylim(0, min(8000, sr/2))

        # Spectrogramme clean
        axes[1].set_title("Spectrogramme - CLEAN")
        _, _, Sxx_c = signal.spectrogram(clean, sr, nperseg=512, noverlap=384)
        axes[1].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx_c + 1e-12), shading="gouraud", cmap="inferno", vmin=-80, vmax=-20)
        axes[1].set_ylabel("Fréquence (Hz)")
        axes[1].set_ylim(0, min(8000, sr/2))

        # Spectrogramme de la différence
        axes[2].set_title("Spectrogramme - DIFFÉRENCE (noisy - clean)")
        _, _, Sxx_d = signal.spectrogram(diff, sr, nperseg=512, noverlap=384)
        axes[2].pcolormesh(t_spec, f_spec, 10*np.log10(Sxx_d + 1e-12), shading="gouraud", cmap="inferno", vmin=-80, vmax=-20)
        axes[2].set_ylabel("Fréquence (Hz)")
        axes[2].set_ylim(0, min(8000, sr/2))

        # PSD comparée
        axes[3].set_title("Densité spectrale de puissance (PSD)")
        axes[3].plot(freqs_n, psd_n_db, label="Noisy", alpha=0.8)
        axes[3].plot(freqs_c, psd_c_db, label="Clean", alpha=0.8)
        axes[3].plot(freqs_d, psd_d_db, label="Différence", alpha=0.8, linestyle="--")
        axes[3].set_xlabel("Fréquence (Hz)")
        axes[3].set_ylabel("PSD (dB)")
        axes[3].set_xlim(0, min(8000, sr/2))
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)

        plt.tight_layout()
        out_fig = OUTPUT_DIR / f"analysis_{name}.png"
        plt.savefig(out_fig, dpi=150)
        plt.close()
        print(f"\n  Spectrogramme sauvegardé: {out_fig}")

    return {
        "name": name,
        "duration": duration,
        "corr": corr,
        "snr": snr,
        "rms_noisy": rms_n,
        "rms_clean": rms_c,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results = []

    for noisy_rel, clean_rel in PAIRS:
        noisy_path = DATASET_DIR / noisy_rel
        clean_path = DATASET_DIR / clean_rel
        if noisy_path.exists() and clean_path.exists():
            name = noisy_path.stem
            # Limiter à 60s pour les très longs fichiers
            r = analyze_pair(noisy_path, clean_path, name, target_sr=16000)
            results.append(r)

    # Résumé global
    print(f"\n\n{'='*70}")
    print(f"  RÉSUMÉ GLOBAL")
    print(f"{'='*70}")
    print(f"\n  {'Fichier':<45s} {'Corr':>6s} {'SNR(dB)':>8s} {'RMS_n':>8s} {'RMS_c':>8s}")
    for r in results:
        print(f"  {r['name']:<45s} {r['corr']:6.3f} {r['snr']:8.1f} {r['rms_noisy']:8.4f} {r['rms_clean']:8.4f}")

    avg_corr = np.mean([r["corr"] for r in results])
    avg_snr = np.mean([r["snr"] for r in results])
    print(f"\n  Corrélation moyenne: {avg_corr:.3f}")
    print(f"  SNR moyen: {avg_snr:.1f} dB")

    if avg_corr > 0.95:
        print("\n  CONSTAT: Les fichiers clean et noisy sont TRÈS similaires.")
        print("  Les fichiers 'clean' ne sont probablement PAS de vraies références propres,")
        print("  mais plutôt des versions légèrement filtrées des mêmes enregistrements.")
    elif avg_corr > 0.8:
        print("\n  CONSTAT: Différences modérées entre clean et noisy.")
        print("  Le traitement clean a retiré une partie du bruit mais pas tout.")
    else:
        print("\n  CONSTAT: Différences significatives entre clean et noisy — bonnes paires.")


if __name__ == "__main__":
    main()
