"""
Test rapide : Demucs seul sur un fichier radio reel.
Compare raw vs sortie Demucs avec metriques et spectrogramme.
Limite a 30s pour resultat rapide.
"""
import os, sys, subprocess, tempfile, shutil
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

sys.stdout.reconfigure(encoding="utf-8")

MAX_S = 30
SR = 44100
INPUT = os.path.join("Dataset Radio (2)", "01__kwahmah_atc001.flac")
OUTPUT_DIR = os.path.join("output", "demucs_only_test")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Metriques
from pystoi import stoi as compute_stoi
try:
    from pesq import pesq as compute_pesq
    PESQ_OK = True
except ImportError:
    PESQ_OK = False

# === 1. Charger le fichier radio (30s max) ===
print(f"Chargement : {INPUT}")
tmp_in = os.path.join(tempfile.gettempdir(), "dtest_in.wav")
subprocess.run(["ffmpeg", "-y", "-i", INPUT, "-t", str(MAX_S),
                "-ar", str(SR), "-ac", "1", tmp_in], capture_output=True, check=True)
raw, _ = sf.read(tmp_in, dtype="float32")
print(f"  Duree : {len(raw)/SR:.1f}s @ {SR}Hz")

# === 2. Appliquer Demucs ===
print("Demucs en cours...")
tmp_out = os.path.join(tempfile.gettempdir(), "dtest_out")
cmd = [sys.executable, "-m", "demucs", "--two-stems", "vocals",
       "-n", "htdemucs", "-o", tmp_out, "-d", "cuda", tmp_in]
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print("ERREUR Demucs:", r.stderr[:300])
    sys.exit(1)

vocals_path = None
for root, dirs, files in os.walk(tmp_out):
    for f in files:
        if f == "vocals.wav":
            vocals_path = os.path.join(root, f)
            break

demucs_out, _ = sf.read(vocals_path, dtype="float32")
if demucs_out.ndim > 1:
    demucs_out = np.mean(demucs_out, axis=1)
demucs_out = demucs_out[:len(raw)]
print("  Demucs OK")

# === 3. Charger le clean de reference ===
clean_path = os.path.join("Dataset Radio (2)", "01__kwahmah_atc001-Clean.wav")
tmp_clean = os.path.join(tempfile.gettempdir(), "dtest_clean.wav")
subprocess.run(["ffmpeg", "-y", "-i", clean_path, "-t", str(MAX_S),
                "-ar", str(SR), "-ac", "1", tmp_clean], capture_output=True, check=True)
clean, _ = sf.read(tmp_clean, dtype="float32")
min_len = min(len(raw), len(demucs_out), len(clean))
raw = raw[:min_len]
demucs_out = demucs_out[:min_len]
clean = clean[:min_len]

# === 4. Metriques ===
def lsd(ref, deg, sr, n_fft=2048):
    _, Pref = welch(ref, sr, nperseg=n_fft)
    _, Pdeg = welch(deg, sr, nperseg=n_fft)
    return float(np.mean(np.abs(10*np.log10(Pref+1e-8) - 10*np.log10(Pdeg+1e-8))))

sr16 = 16000
from math import gcd
from scipy.signal import resample_poly
g = gcd(SR, sr16)
raw16 = resample_poly(raw, sr16//g, SR//g).astype(np.float32)
dem16 = resample_poly(demucs_out, sr16//g, SR//g).astype(np.float32)
cln16 = resample_poly(clean, sr16//g, SR//g).astype(np.float32)
mn = min(len(raw16), len(dem16), len(cln16))
raw16, dem16, cln16 = raw16[:mn], dem16[:mn], cln16[:mn]

print("\n--- Metriques (raw vs reference) ---")
stoi_raw = compute_stoi(cln16, raw16, sr16, extended=False)
lsd_raw  = lsd(clean, raw, SR)
print(f"  STOI raw   : {stoi_raw:.3f}")
print(f"  LSD  raw   : {lsd_raw:.2f} dB")
if PESQ_OK:
    pesq_raw = compute_pesq(sr16, cln16, raw16, "wb")
    print(f"  PESQ raw   : {pesq_raw:.3f}")

print("\n--- Metriques (Demucs only vs reference) ---")
stoi_dem = compute_stoi(cln16, dem16, sr16, extended=False)
lsd_dem  = lsd(clean, demucs_out, SR)
print(f"  STOI Demucs: {stoi_dem:.3f}  (delta: {stoi_dem-stoi_raw:+.3f})")
print(f"  LSD  Demucs: {lsd_dem:.2f} dB  (delta: {lsd_dem-lsd_raw:+.2f})")
if PESQ_OK:
    pesq_dem = compute_pesq(sr16, cln16, dem16, "wb")
    print(f"  PESQ Demucs: {pesq_dem:.3f}  (delta: {pesq_dem-pesq_raw:+.3f})")

# === 5. Spectrogramme comparatif ===
def plot_spec(ax, sig, sr, title):
    ax.specgram(sig, NFFT=1024, Fs=sr, noverlap=512, cmap="inferno", vmin=-80, vmax=0)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Temps (s)")
    ax.set_ylim(0, 8000)

fig, axes = plt.subplots(3, 1, figsize=(12, 9))
plot_spec(axes[0], raw,       SR, "RAW (entree)")
plot_spec(axes[1], demucs_out, SR, "Demucs only")
plot_spec(axes[2], clean,     SR, "Reference (clean)")
plt.tight_layout()
out_png = os.path.join(OUTPUT_DIR, "demucs_only_comparison.png")
plt.savefig(out_png, dpi=150)
print(f"\nSpectrogramme sauvegarde : {out_png}")

# Sauvegarder les WAV
sf.write(os.path.join(OUTPUT_DIR, "raw.wav"), raw, SR)
sf.write(os.path.join(OUTPUT_DIR, "demucs_only.wav"), demucs_out, SR)
sf.write(os.path.join(OUTPUT_DIR, "clean_ref.wav"), clean, SR)
print("WAV sauvegardes dans", OUTPUT_DIR)

shutil.rmtree(tmp_out, ignore_errors=True)
