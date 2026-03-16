"""
Analyse le profil spectral des vrais fichiers radio pour calibrer
les degradations synthetiques du dataset d'entrainement.

Mesure sur chaque fichier OGG :
- Bande passante effective (-3dB, -10dB, -20dB)
- Profil de bruit (floor, distribution spectrale)
- Taux de clipping
- Enveloppe spectrale moyenne (pour comparaison avec le synthetique)

Genere un rapport + graphiques dans output/analysis/
"""
import os
import sys
import json
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import welch, find_peaks
import matplotlib

sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === Configuration ===

INPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "New", "DatasetRadioCom")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "analysis")
ANALYSIS_SR = 44100  # Analyser a 44.1 kHz pour voir tout le spectre
MAX_DURATION_S = 60  # Analyser les 60 premieres secondes de chaque fichier


def load_ogg(filepath, sr=ANALYSIS_SR, max_s=MAX_DURATION_S):
    """Charge un OGG via ffmpeg."""
    tmp = os.path.join(tempfile.gettempdir(), "analysis_tmp.wav")
    cmd = ["ffmpeg", "-y", "-i", filepath]
    if max_s:
        cmd += ["-t", str(max_s)]
    cmd += ["-ac", "1", "-ar", str(sr), tmp]
    subprocess.run(cmd, capture_output=True, check=True)
    data, sr_out = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return data, sr_out


def analyze_bandwidth(freqs, psd_db):
    """Trouve la bande passante a differents seuils sous le pic."""
    peak_db = np.max(psd_db)
    result = {}
    for threshold_name, threshold in [("-3dB", 3), ("-10dB", 10), ("-20dB", 20)]:
        mask = psd_db >= (peak_db - threshold)
        if np.any(mask):
            freq_above = freqs[mask]
            result[threshold_name] = {
                "low_hz": round(float(freq_above[0]), 1),
                "high_hz": round(float(freq_above[-1]), 1),
                "bandwidth_hz": round(float(freq_above[-1] - freq_above[0]), 1),
            }
        else:
            result[threshold_name] = {"low_hz": 0, "high_hz": 0, "bandwidth_hz": 0}
    return result


def analyze_noise_floor(data, sr):
    """Estime le bruit de fond via les segments les plus silencieux."""
    frame_len = int(sr * 0.02)  # Frames de 20ms
    n_frames = len(data) // frame_len
    if n_frames == 0:
        return {"noise_floor_db": -100, "noise_std_db": 0}

    frames = data[:n_frames * frame_len].reshape(n_frames, frame_len)
    frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
    frame_rms_db = 20 * np.log10(frame_rms + 1e-10)

    # Les 10% les plus silencieux = estimation du bruit
    percentile_10 = np.percentile(frame_rms_db, 10)
    noise_frames = frame_rms_db[frame_rms_db <= percentile_10]

    return {
        "noise_floor_db": round(float(np.mean(noise_frames)), 1),
        "noise_std_db": round(float(np.std(noise_frames)), 1),
        "signal_floor_db": round(float(np.percentile(frame_rms_db, 50)), 1),
        "snr_estimate_db": round(float(np.percentile(frame_rms_db, 50) - np.mean(noise_frames)), 1),
    }


def analyze_clipping(data, threshold=0.95):
    """Detecte le taux de clipping."""
    n_clipped = np.sum(np.abs(data) >= threshold)
    return {
        "clipping_ratio": round(float(n_clipped / len(data)), 6),
        "clipping_percent": round(float(n_clipped / len(data) * 100), 3),
        "peak_amplitude": round(float(np.max(np.abs(data))), 4),
    }


def analyze_spectral_shape(freqs, psd):
    """Analyse la forme spectrale (pente, pics)."""
    psd_db = 10 * np.log10(psd + 1e-20)

    # Pente spectrale (regression lineaire en dB)
    valid = (freqs > 100) & (freqs < 8000) & np.isfinite(psd_db)
    if np.sum(valid) > 10:
        coeffs = np.polyfit(freqs[valid], psd_db[valid], 1)
        slope_db_per_khz = coeffs[0] * 1000
    else:
        slope_db_per_khz = 0

    # Pics spectraux (formants ou interferences)
    peaks_idx, properties = find_peaks(psd_db, height=-60, prominence=5, distance=20)
    peak_freqs = freqs[peaks_idx].tolist()

    # Centroide
    centroid = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-10))

    return {
        "spectral_centroid_hz": round(centroid, 0),
        "spectral_slope_db_per_khz": round(float(slope_db_per_khz), 2),
        "n_spectral_peaks": len(peak_freqs),
        "peak_frequencies_hz": [round(f, 0) for f in peak_freqs[:10]],
    }


def analyze_file(filepath, filename):
    """Analyse complete d'un fichier radio."""
    print(f"\n  Analyse : {filename}")
    data, sr = load_ogg(filepath)
    print(f"    Duree: {len(data)/sr:.1f}s | SR: {sr} Hz | Peak: {np.max(np.abs(data)):.3f}")

    # PSD
    freqs, psd = welch(data, fs=sr, nperseg=4096, noverlap=2048)
    psd_db = 10 * np.log10(psd + 1e-20)

    # Analyses
    result = {
        "filename": filename,
        "duration_s": round(len(data) / sr, 1),
        "sr": sr,
    }
    result["bandwidth"] = analyze_bandwidth(freqs, psd_db)
    result["noise"] = analyze_noise_floor(data, sr)
    result["clipping"] = analyze_clipping(data)
    result["spectral"] = analyze_spectral_shape(freqs, psd)

    # Resume console
    bw3 = result["bandwidth"]["-3dB"]
    bw20 = result["bandwidth"]["-20dB"]
    print(f"    Bande -3dB  : {bw3['low_hz']:.0f} - {bw3['high_hz']:.0f} Hz ({bw3['bandwidth_hz']:.0f} Hz)")
    print(f"    Bande -20dB : {bw20['low_hz']:.0f} - {bw20['high_hz']:.0f} Hz ({bw20['bandwidth_hz']:.0f} Hz)")
    print(f"    SNR estime  : {result['noise']['snr_estimate_db']:.1f} dB")
    print(f"    Clipping    : {result['clipping']['clipping_percent']:.3f}%")
    print(f"    Centroide   : {result['spectral']['spectral_centroid_hz']:.0f} Hz")
    print(f"    Pente       : {result['spectral']['spectral_slope_db_per_khz']:.2f} dB/kHz")

    return result, freqs, psd, psd_db, data, sr


def plot_all_psd(all_results, output_path):
    """PSD superposes de tous les fichiers radio."""
    fig, ax = plt.subplots(figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

    for (result, freqs, psd, psd_db, _, _), color in zip(all_results, colors):
        label = result["filename"][:30]
        ax.plot(freqs, psd_db, label=label, color=color, alpha=0.7, linewidth=1)

    # Marquer la bande radio typique
    ax.axvspan(300, 3400, alpha=0.1, color="red", label="Bande AM (300-3400 Hz)")
    ax.axvline(300, color="red", linestyle="--", alpha=0.3)
    ax.axvline(3400, color="red", linestyle="--", alpha=0.3)

    ax.set_xlabel("Frequence (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Densite Spectrale - Fichiers Radio Reels")
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22050)
    ax.set_ylim(-120, -20)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_envelope_stats(all_results, output_path):
    """Enveloppe spectrale moyenne + ecart-type."""
    # Interpoler toutes les PSD sur une grille commune
    common_freqs = np.linspace(0, 22050, 2000)
    all_psd_db = []

    for result, freqs, psd, psd_db, _, _ in all_results:
        interp = np.interp(common_freqs, freqs, psd_db)
        all_psd_db.append(interp)

    all_psd_db = np.array(all_psd_db)
    mean_psd = np.mean(all_psd_db, axis=0)
    std_psd = np.std(all_psd_db, axis=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(common_freqs, mean_psd, color="blue", linewidth=2, label="Moyenne")
    ax.fill_between(common_freqs, mean_psd - std_psd, mean_psd + std_psd,
                    alpha=0.2, color="blue", label="+/- 1 ecart-type")

    ax.axvspan(300, 3400, alpha=0.1, color="red", label="Bande AM")
    ax.set_xlabel("Frequence (Hz)")
    ax.set_ylabel("PSD (dB)")
    ax.set_title("Enveloppe Spectrale Moyenne des Fichiers Radio")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 22050)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def generate_degradation_profile(all_results):
    """Genere un profil de degradation calibre sur les vrais fichiers."""
    bandwidths_3db = []
    bandwidths_20db = []
    snrs = []
    clipping_rates = []
    centroids = []
    slopes = []

    for result, *_ in all_results:
        bw3 = result["bandwidth"]["-3dB"]
        bw20 = result["bandwidth"]["-20dB"]
        bandwidths_3db.append((bw3["low_hz"], bw3["high_hz"]))
        bandwidths_20db.append((bw20["low_hz"], bw20["high_hz"]))
        snrs.append(result["noise"]["snr_estimate_db"])
        clipping_rates.append(result["clipping"]["clipping_ratio"])
        centroids.append(result["spectral"]["spectral_centroid_hz"])
        slopes.append(result["spectral"]["spectral_slope_db_per_khz"])

    low_3db = [b[0] for b in bandwidths_3db]
    high_3db = [b[1] for b in bandwidths_3db]

    profile = {
        "description": "Profil de degradation calibre sur les fichiers radio reels",
        "bandpass": {
            "low_hz_range": [round(min(low_3db)), round(max(low_3db))],
            "high_hz_range": [round(min(high_3db)), round(max(high_3db))],
            "comment": "Bande passante -3dB mesuree",
        },
        "snr": {
            "min_db": round(min(snrs), 1),
            "max_db": round(max(snrs), 1),
            "mean_db": round(float(np.mean(snrs)), 1),
            "comment": "SNR estime (signal median vs bruit p10)",
        },
        "clipping": {
            "mean_ratio": round(float(np.mean(clipping_rates)), 6),
            "max_ratio": round(float(np.max(clipping_rates)), 6),
        },
        "spectral_centroid_hz": {
            "min": round(min(centroids)),
            "max": round(max(centroids)),
            "mean": round(float(np.mean(centroids))),
        },
        "spectral_slope_db_per_khz": {
            "min": round(min(slopes), 2),
            "max": round(max(slopes), 2),
            "mean": round(float(np.mean(slopes)), 2),
        },
    }
    return profile


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("  ANALYSE DES FICHIERS RADIO REELS")
    print("  Calibration des degradations synthetiques")
    print("=" * 60)

    # Lister les fichiers
    filenames = sorted([f for f in os.listdir(INPUT_DIR) if f.endswith(".ogg")])
    print(f"  {len(filenames)} fichiers trouves dans {INPUT_DIR}")

    all_results = []
    for filename in filenames:
        filepath = os.path.join(INPUT_DIR, filename)
        result = analyze_file(filepath, filename)
        all_results.append(result)

    # Graphiques
    print("\n  Generation des graphiques...")
    plot_all_psd(all_results, os.path.join(OUTPUT_DIR, "radio_psd_all.png"))
    plot_envelope_stats(all_results, os.path.join(OUTPUT_DIR, "radio_psd_envelope.png"))

    # Profil de degradation
    profile = generate_degradation_profile(all_results)

    profile_path = os.path.join(OUTPUT_DIR, "degradation_profile.json")
    with open(profile_path, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"  PROFIL DE DEGRADATION CALIBRE")
    print(f"{'=' * 60}")
    print(f"  Bandpass low  : {profile['bandpass']['low_hz_range']} Hz")
    print(f"  Bandpass high : {profile['bandpass']['high_hz_range']} Hz")
    print(f"  SNR           : {profile['snr']['min_db']} - {profile['snr']['max_db']} dB (moy: {profile['snr']['mean_db']})")
    print(f"  Clipping      : {profile['clipping']['mean_ratio']*100:.3f}% moyen")
    print(f"  Centroide     : {profile['spectral_centroid_hz']['min']} - {profile['spectral_centroid_hz']['max']} Hz")
    print(f"  Pente         : {profile['spectral_slope_db_per_khz']['min']} a {profile['spectral_slope_db_per_khz']['max']} dB/kHz")
    print(f"\n  Profil sauvegarde : {profile_path}")
    print(f"  Graphiques       : {OUTPUT_DIR}")
    print(f"\n  -> Utilisez ce profil pour calibrer create_dataset.py")


if __name__ == "__main__":
    main()
