"""
Analyse de la reconstruction haute frequence.
Compare les spectrogrammes et l'energie HF (> 4 kHz) des sorties du pipeline.
"""
import os, sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

RUN_DIR = os.path.join("runs", "01__kwahmah_atc001")
OUT_DIR = os.path.join("output", "hf_analysis")
os.makedirs(OUT_DIR, exist_ok=True)

SR = 44100
HF_CUTOFF = 4000  # Hz - frequence a partir de laquelle on mesure la reconstruction

FILES = [
    ("00_raw",            "01__kwahmah_atc001__00_raw.wav"),
    ("01_demucs",         "01__kwahmah_atc001__01_demucs.wav"),
    ("02_unet_untrained", "01__kwahmah_atc001__00_unet_pas_entraine.wav"),
    ("03_epoch010",       "01__kwahmah_atc001__01_epoch010.wav"),
    ("04_epoch030",       "01__kwahmah_atc001__03_epoch030.wav"),
    ("05_epoch060",       "01__kwahmah_atc001__06_epoch060.wav"),
    ("06_best_model",     "01__kwahmah_atc001__07_best_model.wav"),
]

# Ajouter le clean de reference si disponible
CLEAN_REF = os.path.join("Dataset Radio (2)", "01__kwahmah_atc001-Clean.wav")

def load(path):
    data, sr = sf.read(path, dtype="float32")
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data

def hf_energy_ratio(sig, sr, cutoff=HF_CUTOFF):
    """Ratio energie > cutoff Hz / energie totale (en %)"""
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/sr)
    total = np.sum(fft**2) + 1e-12
    hf = np.sum(fft[freqs >= cutoff]**2)
    return 100 * hf / total

def hf_db(sig, sr, cutoff=HF_CUTOFF):
    """Niveau moyen en dB dans la bande > cutoff"""
    fft = np.abs(np.fft.rfft(sig))
    freqs = np.fft.rfftfreq(len(sig), 1/sr)
    hf_vals = fft[freqs >= cutoff]
    return 20 * np.log10(np.mean(hf_vals) + 1e-12)

# Charger tous les fichiers
signals = []
for label, fname in FILES:
    path = os.path.join(RUN_DIR, fname)
    if not os.path.exists(path):
        print(f"SKIP {fname}")
        continue
    sig = load(path)
    signals.append((label, sig))

# Ajouter reference clean
if os.path.exists(CLEAN_REF):
    import subprocess, tempfile
    tmp = os.path.join(tempfile.gettempdir(), "_clean_ref.wav")
    subprocess.run(["ffmpeg", "-y", "-i", CLEAN_REF, "-t", "30",
                    "-ar", str(SR), "-ac", "1", tmp],
                   capture_output=True, check=True)
    sig = load(tmp)
    signals.insert(0, ("CLEAN_ref", sig))

# Aligner les longueurs sur le plus court
min_len = min(len(s) for _, s in signals)
signals = [(l, s[:min_len]) for l, s in signals]

# ── Tableau energie HF ───────────────────────────────────────
print(f"\n{'Label':<22} {'HF ratio (%)':>12} {'HF niveau (dB)':>15}")
print("-" * 52)
for label, sig in signals:
    ratio = hf_energy_ratio(sig, SR)
    level = hf_db(sig, SR)
    print(f"{label:<22} {ratio:>12.3f} {level:>15.2f}")

# ── Spectrogrammes pleine bande (0 - 22 kHz) ────────────────
n = len(signals)
fig, axes = plt.subplots(n, 1, figsize=(14, 2.8 * n))
if n == 1:
    axes = [axes]

for ax, (label, sig) in zip(axes, signals):
    ax.specgram(sig, NFFT=2048, Fs=SR, noverlap=1536,
                cmap="inferno", vmin=-100, vmax=-20)
    ax.set_ylim(0, SR // 2)
    ax.axhline(y=HF_CUTOFF, color='cyan', linewidth=0.8, linestyle='--', alpha=0.7)
    ax.set_ylabel("Hz", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight='bold')
    # Ligne de separation basse/haute frequence
    ax.text(0.01, HF_CUTOFF + 200, f"<-- zone HF reconstruite", fontsize=7,
            color='cyan', transform=ax.get_yaxis_transform())

axes[-1].set_xlabel("Temps (s)")
plt.suptitle("Comparaison spectrale - reconstruction HF (ligne cyan = 4 kHz)",
             fontsize=11, fontweight='bold')
plt.tight_layout()
out_full = os.path.join(OUT_DIR, "spectrogrammes_fullband.png")
plt.savefig(out_full, dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSpectrogramme pleine bande : {out_full}")

# ── Zoom bande HF uniquement (4 - 22 kHz) ───────────────────
fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n))
if n == 1:
    axes = [axes]

for ax, (label, sig) in zip(axes, signals):
    ax.specgram(sig, NFFT=2048, Fs=SR, noverlap=1536,
                cmap="inferno", vmin=-100, vmax=-20)
    ax.set_ylim(HF_CUTOFF, SR // 2)
    ax.set_ylabel("Hz", fontsize=8)
    ax.set_title(label, fontsize=9, fontweight='bold')

axes[-1].set_xlabel("Temps (s)")
plt.suptitle("ZOOM bande HF uniquement (4 - 22 kHz) - energie reconstruite",
             fontsize=11, fontweight='bold')
plt.tight_layout()
out_hf = os.path.join(OUT_DIR, "spectrogrammes_HF_zoom.png")
plt.savefig(out_hf, dpi=150, bbox_inches='tight')
plt.close()
print(f"Spectrogramme zoom HF      : {out_hf}")
