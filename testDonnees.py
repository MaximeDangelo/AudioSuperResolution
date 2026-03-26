"""
testDonnees.py - Pipeline de comparaison des checkpoints ResUNet.

Usage : python testDonnees.py <fichier_audio> [start_s] [duration_s]

  start_s    : debut du segment en secondes (defaut : 0)
  duration_s : duree du segment en secondes (defaut : 15)

Tous les fichiers de sortie ont exactement la meme duree et le meme segment
pour permettre une comparaison directe a l'ecoute.

Pour chaque fichier donne :
  1. Copie du segment raw original
  2. Sortie Demucs sur ce segment
  3. Sortie ResUNet non entraine (poids aleatoires) applique sur sortie Demucs
  4. Sorties ResUNet checkpoints epoch 10/20/30/40/50/60 appliques sur sortie Demucs
  5. Sortie best_model.pt applique sur sortie Demucs

Tout est sauve dans : runs/<nom_du_fichier_sans_extension>/
"""
import os, sys, subprocess, tempfile, shutil
import numpy as np
import soundfile as sf
import torch

sys.stdout.reconfigure(encoding="utf-8")

# Forcer CPU pour eviter les GPU hangs ROCm lors de chargements multiples
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from train import SpectralResUNet, SR

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints")
RUNS_DIR     = os.path.join(SCRIPT_DIR, "runs")
DEMUCS_MODEL = "htdemucs"

# Seuil RMS en dessous duquel une trame est consideree silencieuse
# Le modele est bypasse sur ces trames pour eviter l'hallucination
SILENCE_RMS_THRESHOLD = 0.005
SILENCE_FRAME_S = 0.1  # Taille des trames pour detection silence (100ms)

CHECKPOINTS = [
    ("00_unet_pas_entraine",   None),
    ("01_epoch010",            os.path.join(CKPT_DIR, "checkpoint_epoch010.pt")),
    ("02_epoch020",            os.path.join(CKPT_DIR, "checkpoint_epoch020.pt")),
    ("03_epoch030",            os.path.join(CKPT_DIR, "checkpoint_epoch030.pt")),
    ("04_epoch040",            os.path.join(CKPT_DIR, "checkpoint_epoch040.pt")),
    ("05_epoch050",            os.path.join(CKPT_DIR, "checkpoint_epoch050.pt")),
    ("06_epoch060",            os.path.join(CKPT_DIR, "checkpoint_epoch060.pt")),
    ("07_best_model",          os.path.join(CKPT_DIR, "best_model.pt")),
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def load_audio(path, start_s=0, duration_s=15):
    tmp = os.path.join(tempfile.gettempdir(), "_td_in.wav")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path,
         "-ss", str(start_s), "-t", str(duration_s),
         "-ar", str(SR), "-ac", "1", tmp],
        capture_output=True, check=True
    )
    data, _ = sf.read(tmp, dtype="float32")
    os.remove(tmp)
    return data

def run_demucs(data):
    tmp_in  = os.path.join(tempfile.gettempdir(), "_td_demucs_in.wav")
    tmp_out = os.path.join(tempfile.gettempdir(), "_td_demucs_out")
    sf.write(tmp_in, data, SR)

    cmd = [sys.executable, "-m", "demucs",
           "--two-stems", "vocals",
           "-n", DEMUCS_MODEL,
           "-o", tmp_out,
           "-d", "cpu",
           tmp_in]

    print("  [Demucs] traitement en cours...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [Demucs] ERREUR : {result.stderr[:300]}")
        return data

    vocals_path = None
    for root, _, files in os.walk(tmp_out):
        for f in files:
            if f == "vocals.wav":
                vocals_path = os.path.join(root, f)
                break
        if vocals_path:
            break

    if not vocals_path:
        print("  [Demucs] vocals.wav introuvable")
        return data

    vocals, _ = sf.read(vocals_path, dtype="float32")
    if vocals.ndim > 1:
        vocals = np.mean(vocals, axis=1)

    mn = min(len(data), len(vocals))
    out = vocals[:mn].astype(np.float32)

    try:
        os.remove(tmp_in)
        shutil.rmtree(tmp_out, ignore_errors=True)
    except Exception:
        pass

    return out

def apply_silence_gate(data, model_fn, sr=SR):
    """Applique model_fn trame par trame, bypasse sur les silences."""
    frame_len = int(SILENCE_FRAME_S * sr)
    out = np.zeros_like(data)
    n_frames = int(np.ceil(len(data) / frame_len))

    for i in range(n_frames):
        s = i * frame_len
        e = min(s + frame_len, len(data))
        frame = data[s:e]
        rms = np.sqrt(np.mean(frame ** 2))
        if rms < SILENCE_RMS_THRESHOLD:
            out[s:e] = frame  # bypass : retourner l'entree directement
        else:
            # Traiter le segment complet avec contexte pour eviter les coupures
            ctx_s = max(0, s - frame_len)
            ctx_e = min(len(data), e + frame_len)
            chunk = data[ctx_s:ctx_e]
            chunk_out = model_fn(chunk)
            # Extraire la partie qui nous interesse
            offset = s - ctx_s
            out[s:e] = chunk_out[offset:offset + (e - s)]

    return out

def run_unet(data, ckpt_path=None):
    model = SpectralResUNet().to(torch.device("cpu"))
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    def model_fn(chunk):
        with torch.no_grad():
            x   = torch.tensor(chunk).unsqueeze(0).unsqueeze(0)
            out = model(x).squeeze().numpy().astype(np.float32)
        return out

    out = apply_silence_gate(data, model_fn)

    if np.max(np.abs(out)) > 1e-6:
        out = out / np.max(np.abs(out)) * 0.9

    return out

def normalize(data):
    mx = np.max(np.abs(data))
    return data / mx * 0.9 if mx > 1e-6 else data

# ── Main ─────────────────────────────────────────────────────────────────────

if len(sys.argv) < 2:
    print("Usage : python testDonnees.py <fichier_audio> [start_s] [duration_s]")
    sys.exit(1)

input_path = sys.argv[1]
start_s    = float(sys.argv[2]) if len(sys.argv) > 2 else 0
duration_s = float(sys.argv[3]) if len(sys.argv) > 3 else 15

if not os.path.exists(input_path):
    print(f"Fichier introuvable : {input_path}")
    sys.exit(1)

base_name = os.path.splitext(os.path.basename(input_path))[0]
out_dir   = os.path.join(RUNS_DIR, base_name)
os.makedirs(out_dir, exist_ok=True)

print(f"\n=== testDonnees : {base_name} ===")
print(f"Segment : {start_s}s -> {start_s + duration_s}s ({duration_s}s)")
print(f"Dossier de sortie : {out_dir}\n")

# 1. Copie du raw
print("[1/3] Chargement et copie du segment raw...")
raw = load_audio(input_path, start_s, duration_s)
n_samples = len(raw)
sf.write(os.path.join(out_dir, f"{base_name}__00_raw.wav"), normalize(raw), SR)
print(f"  -> {base_name}__00_raw.wav")

# 2. Demucs
print("\n[2/3] Demucs...")
demucs_out = run_demucs(raw)
# Aligner sur la longueur du raw (segment fixe)
demucs_out = demucs_out[:n_samples]
if len(demucs_out) < n_samples:
    demucs_out = np.pad(demucs_out, (0, n_samples - len(demucs_out)))
sf.write(os.path.join(out_dir, f"{base_name}__01_demucs.wav"), normalize(demucs_out), SR)
print(f"  -> {base_name}__01_demucs.wav  ({len(demucs_out)/SR:.2f}s)")

# 3. ResUNet (tous les checkpoints appliques sur la sortie Demucs)
print(f"\n[3/3] ResUNet ({len(CHECKPOINTS)} modeles)...")
for label, ckpt_path in CHECKPOINTS:
    if ckpt_path and not os.path.exists(ckpt_path):
        print(f"  SKIP {label} (checkpoint introuvable : {ckpt_path})")
        continue

    status = "poids aleatoires" if ckpt_path is None else os.path.basename(ckpt_path)
    print(f"  {label} ({status})...")
    out = run_unet(demucs_out, ckpt_path)
    # Aligner exactement sur la longueur du segment
    out = out[:n_samples]
    if len(out) < n_samples:
        out = np.pad(out, (0, n_samples - len(out)))
    fname = f"{base_name}__{label}.wav"
    sf.write(os.path.join(out_dir, fname), out, SR)
    print(f"    -> {fname}")

print(f"\nTermine ! {len(CHECKPOINTS) + 2} fichiers dans : {out_dir}")
print("\nConvention de nommage :")
print("  00_raw          -> signal brut original")
print("  01_demucs       -> apres Demucs")
print("  00_unet_...     -> ResUNet non entraine (sur sortie Demucs)")
print("  01..06_epoch... -> ResUNet checkpoint epoch N (sur sortie Demucs)")
print("  07_best_model   -> meilleur modele (sur sortie Demucs)")
