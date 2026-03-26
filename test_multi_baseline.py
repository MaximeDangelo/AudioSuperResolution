"""
Test baseline multi-fichiers : RAW vs ResUNet non entraine.
Teste sur :
  - 4 fichiers ATC reels (avec clean de reference)
  - 5 fichiers synthetiques du dataset val (paires clean/raw deja disponibles)
Genere un tableau recapitulatif et des spectrogrammes.
"""
import os, sys, subprocess, tempfile
import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, resample_poly
from math import gcd

sys.stdout.reconfigure(encoding="utf-8")

from train import SpectralResUNet, SR, N_FFT, HOP_LENGTH, WIN_LENGTH, get_device
from pystoi import stoi as compute_stoi
try:
    from pesq import pesq as compute_pesq
    PESQ_OK = True
except ImportError:
    PESQ_OK = False

DEVICE  = get_device()
OUT_DIR = os.path.join("output", "multi_baseline_test")
os.makedirs(OUT_DIR, exist_ok=True)
ATC_DIR = "Dataset Radio (2)"
VAL_DIR = os.path.join("dataset", "val")

# === Fichiers a tester ===
ATC_FILES = [
    ("01__kwahmah_atc001.flac",            "01__kwahmah_atc001-Clean.wav",            "ATC-01 (UK)"),
    ("02__kwahmah_atc002.flac",            "02__kwahmah_atc002-Clean.wav",            "ATC-02 (UK)"),
    ("03_kwahmah_atc003.flac",             "03_kwahmah_atc003-Clean.wav",             "ATC-03 (UK)"),
    ("05__kwahmah_atc005.flac",            "05__kwahmah_atc005-Clean.wav",            "ATC-05 (UK)"),
]
SYNTH_IDS = ["synth_val_00000", "synth_val_00050", "synth_val_00100"]

# === Helpers ===
def load_wav(path, max_s=15):
    tmp = os.path.join(tempfile.gettempdir(), "mb_tmp.wav")
    subprocess.run(["ffmpeg", "-y", "-i", path, "-t", str(max_s),
                    "-ar", str(SR), "-ac", "1", tmp],
                   capture_output=True, check=True)
    data, _ = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return data

def lsd(ref, deg, n_fft=2048):
    _, Pr = welch(ref, SR, nperseg=n_fft)
    _, Pd = welch(deg, SR, nperseg=n_fft)
    return float(np.mean(np.abs(10*np.log10(Pr+1e-8) - 10*np.log10(Pd+1e-8))))

def metrics(clean, sig):
    sr16 = 16000
    g = gcd(SR, sr16)
    c16 = resample_poly(clean, sr16//g, SR//g).astype(np.float32)
    s16 = resample_poly(sig,   sr16//g, SR//g).astype(np.float32)
    mn  = min(len(c16), len(s16))
    c16, s16 = c16[:mn], s16[:mn]
    stoi = compute_stoi(c16, s16, sr16, extended=False)
    lsd_ = lsd(clean[:min(len(clean), len(sig))], sig[:min(len(clean), len(sig))])
    pesq_ = compute_pesq(sr16, c16, s16, "wb") if PESQ_OK else None
    return stoi, lsd_, pesq_

def apply_unet(raw, model):
    with torch.no_grad():
        x = torch.tensor(raw).unsqueeze(0).unsqueeze(0).to(DEVICE)
        out = model(x)
        result = out.squeeze().cpu().numpy().astype(np.float32)
    torch.cuda.empty_cache()
    return result

def plot_spec(ax, sig, title):
    ax.specgram(sig, NFFT=1024, Fs=SR, noverlap=512, cmap="inferno", vmin=-80, vmax=0)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 8000)
    ax.set_ylabel("Hz")
    ax.set_xlabel("s")

# === Charger le modele non entraine sur CPU pour eviter le GPU hang ===
print("Chargement ResUNet (poids aleatoires)...")
model = SpectralResUNet().to(torch.device("cpu"))
model.eval()
# Forcer CPU pour ce test (evite les GPU hangs sur ROCm avec plusieurs appels)
DEVICE = torch.device("cpu")

# === Boucle de test ===
results = []

print("\n=== Fichiers ATC reels ===")
for raw_name, clean_name, label in ATC_FILES:
    raw_path   = os.path.join(ATC_DIR, raw_name)
    clean_path = os.path.join(ATC_DIR, clean_name)
    if not os.path.exists(raw_path) or not os.path.exists(clean_path):
        print(f"  SKIP {label} (fichier manquant)")
        continue
    print(f"  {label}...")
    raw   = load_wav(raw_path)
    clean = load_wav(clean_path)
    mn = min(len(raw), len(clean))
    raw, clean = raw[:mn], clean[:mn]

    unet = apply_unet(raw, model)[:mn]

    st_r, ld_r, pe_r = metrics(clean, raw)
    st_u, ld_u, pe_u = metrics(clean, unet)

    results.append((label, "ATC reel",
                    st_r, ld_r, pe_r,
                    st_u, ld_u, pe_u))

    # Spectrogramme
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    plot_spec(axes[0], raw,   f"RAW  STOI={st_r:.3f} LSD={ld_r:.2f}dB")
    plot_spec(axes[1], unet,  f"UNet non entraine  STOI={st_u:.3f} LSD={ld_u:.2f}dB")
    plot_spec(axes[2], clean, "Reference clean")
    plt.suptitle(label, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"spec_{label.replace(' ','_')}.png"), dpi=130)
    plt.close()

print("\n=== Fichiers synthetiques (dataset val) ===")
for fid in SYNTH_IDS:
    raw_path   = os.path.join(VAL_DIR, "raw",   f"{fid}.wav")
    clean_path = os.path.join(VAL_DIR, "clean", f"{fid}.wav")
    if not os.path.exists(raw_path):
        print(f"  SKIP {fid}")
        continue
    print(f"  {fid}...")
    raw,   _ = sf.read(raw_path,   dtype="float32")
    clean, _ = sf.read(clean_path, dtype="float32")
    mn = min(len(raw), len(clean))
    raw, clean = raw[:mn], clean[:mn]

    unet = apply_unet(raw, model)[:mn]

    st_r, ld_r, pe_r = metrics(clean, raw)
    st_u, ld_u, pe_u = metrics(clean, unet)

    label = fid
    results.append((label, "Synthetique",
                    st_r, ld_r, pe_r,
                    st_u, ld_u, pe_u))

    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    plot_spec(axes[0], raw,   f"RAW  STOI={st_r:.3f} LSD={ld_r:.2f}dB")
    plot_spec(axes[1], unet,  f"UNet non entraine  STOI={st_u:.3f} LSD={ld_u:.2f}dB")
    plot_spec(axes[2], clean, "Reference clean")
    plt.suptitle(label, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"spec_{fid}.png"), dpi=130)
    plt.close()

# === Tableau recapitulatif ===
print("\n" + "="*90)
print(f"{'Fichier':<30} {'Type':<12} {'STOI raw':>9} {'STOI unet':>9} {'dSTOI':>7} "
      f"{'LSD raw':>8} {'LSD unet':>8} {'dLSD':>7}")
print("-"*90)
for (label, typ, st_r, ld_r, pe_r, st_u, ld_u, pe_u) in results:
    print(f"{label:<30} {typ:<12} {st_r:>9.3f} {st_u:>9.3f} {st_u-st_r:>+7.3f} "
          f"{ld_r:>8.2f} {ld_u:>8.2f} {ld_u-ld_r:>+7.2f}")

# Moyennes
if results:
    print("-"*90)
    st_r_m = np.mean([r[2] for r in results])
    st_u_m = np.mean([r[5] for r in results])
    ld_r_m = np.mean([r[3] for r in results])
    ld_u_m = np.mean([r[6] for r in results])
    print(f"{'MOYENNE':<30} {'':<12} {st_r_m:>9.3f} {st_u_m:>9.3f} {st_u_m-st_r_m:>+7.3f} "
          f"{ld_r_m:>8.2f} {ld_u_m:>8.2f} {ld_u_m-ld_r_m:>+7.2f}")
print("="*90)
print(f"\nSpectrogrammes sauvegardes dans : {OUT_DIR}")
