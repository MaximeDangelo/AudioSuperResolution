"""
Comparaison de pipelines sur fichiers radio ATC.

Pour chaque fichier genere :
  1. raw          : signal original
  2. demucs       : Demucs seul
  3. audiosr_raw  : AudioSR seul (sur raw)
  4. demucs_audiosr : Demucs -> AudioSR (pipeline complet)

Tous les fichiers sont decoupes au meme segment (meme debut, meme duree).
AudioSR recommande < 5.12s pour les meilleurs resultats.
Chaque etape est lancee en subprocess separe pour liberer le GPU entre les etapes.
"""
import os, sys, subprocess, tempfile, shutil
import numpy as np
import soundfile as sf

sys.stdout.reconfigure(encoding="utf-8")

# === Config ===
SR_IN    = 16000   # AudioSR attend du 16kHz en entree
SR_OUT   = 48000   # AudioSR sort du 48kHz
DURATION = 5       # secondes par extrait (AudioSR recommande <= 5.12s)
OUT_DIR  = os.path.join("output", "pipeline_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    ("Dataset Radio (2)/07-wahmah_heathrow-air-traffic-control.flac",   75, "heathrow"),
    ("Dataset Radio (2)/08__kwahmah_hong-kong-air-traffic-control.flac",60, "hongkong"),
    ("Dataset Radio (2)/09_fumesKMDT1-App-Dep-West-Feb-02-2023-1800Z.mp3", 15, "us_approach"),
    ("Dataset Radio (2)/05__kwahmah_atc005.flac",                       5, "atc05"),
]

def extract_segment(src, start_s, dur_s, sr, out_path):
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ss", str(start_s), "-t", str(dur_s),
        "-ar", str(sr), "-ac", "1", out_path
    ], capture_output=True, check=True)

def run_demucs(wav_path, out_wav):
    """Lance Demucs dans un subprocess Python separe pour isoler le GPU."""
    script = f"""
import sys, os, shutil, subprocess, torch
tmp_out = '{tempfile.gettempdir()}/demucs_cmp'
cmd = [sys.executable, '-m', 'demucs', '--two-stems', 'vocals', '-n', 'htdemucs', '-o', tmp_out]
cmd += ['-d', 'cuda' if torch.cuda.is_available() else 'cpu']
cmd += ['{wav_path}']
r = subprocess.run(cmd, capture_output=True, text=True)
if r.returncode != 0:
    print('DEMUCS_ERROR:' + r.stderr[:500], file=sys.stderr)
    sys.exit(1)
for root, dirs, files in os.walk(tmp_out):
    for f in files:
        if f == 'vocals.wav':
            shutil.copy2(os.path.join(root, f), '{out_wav}')
            shutil.rmtree(tmp_out, ignore_errors=True)
            sys.exit(0)
print('DEMUCS_ERROR: vocals.wav introuvable', file=sys.stderr)
sys.exit(1)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        err = result.stderr.split("DEMUCS_ERROR:")[-1][:200] if "DEMUCS_ERROR:" in result.stderr else result.stderr[-200:]
        print(f"    ERREUR Demucs: {err.strip()}")
        return False
    return True

def run_audiosr(wav_path, out_wav):
    """Lance AudioSR dans un subprocess Python separe pour isoler le GPU."""
    tmp_dir = os.path.join(tempfile.gettempdir(), "audiosr_tmp")
    script = f"""
import sys, os, shutil
os.makedirs('{tmp_dir}', exist_ok=True)
from audiosr import super_resolution, build_model
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
audiosr_model = build_model(model_name='speech', device=device)
waveform = super_resolution(audiosr_model, '{wav_path}', seed=42, guidance_scale=3.5, ddim_steps=50)
import soundfile as sf
# waveform shape: [batch, channels, samples]
wav = waveform[0, 0] if waveform.ndim == 3 else waveform[0]
sf.write('{out_wav}', wav, 48000)
del audiosr_model, waveform
torch.cuda.empty_cache()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        # Trouver la derniere ligne d'erreur utile
        lines = [l for l in result.stderr.strip().split('\n') if l.strip()]
        err = lines[-1] if lines else "erreur inconnue"
        print(f"    ERREUR AudioSR: {err[:200]}")
        return False
    return True

def resample_wav(src, dst, target_sr):
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ar", str(target_sr), "-ac", "1", dst
    ], capture_output=True, check=True)


# === Boucle principale ===
print("=== Comparaison de pipelines (5s par extrait) ===\n")

for src_path, start_s, name in FILES:
    if not os.path.exists(src_path):
        print(f"SKIP {name} (fichier introuvable)")
        continue

    print(f"--- {name} ---")
    file_dir = os.path.join(OUT_DIR, name)
    os.makedirs(file_dir, exist_ok=True)

    # 1. Extraire le segment raw a 16kHz
    raw_16k = os.path.join(file_dir, f"{name}_1_raw_16k.wav")
    extract_segment(src_path, start_s, DURATION, SR_IN, raw_16k)
    raw_48k = os.path.join(file_dir, f"{name}_1_raw_48k.wav")
    resample_wav(raw_16k, raw_48k, SR_OUT)
    print(f"  [1/4] raw extrait ({DURATION}s)")

    # 2. Demucs seul
    print(f"  [2/4] Demucs...")
    demucs_16k = os.path.join(file_dir, f"{name}_demucs_16k_tmp.wav")
    demucs_ok = run_demucs(raw_16k, demucs_16k)
    demucs_out = None
    if demucs_ok:
        demucs_out = os.path.join(file_dir, f"{name}_2_demucs_48k.wav")
        resample_wav(demucs_16k, demucs_out, SR_OUT)
        print(f"         -> {name}_2_demucs_48k.wav")
    else:
        print(f"         -> ECHEC")

    # 3. AudioSR seul (sur raw)
    print(f"  [3/4] AudioSR sur raw...")
    audiosr_raw_out = os.path.join(file_dir, f"{name}_3_audiosr_raw_48k.wav")
    if run_audiosr(raw_16k, audiosr_raw_out):
        print(f"         -> {name}_3_audiosr_raw_48k.wav")
    else:
        print(f"         -> ECHEC")

    # 4. Demucs -> AudioSR
    if demucs_ok:
        print(f"  [4/4] Demucs -> AudioSR...")
        audiosr_demucs_out = os.path.join(file_dir, f"{name}_4_demucs_audiosr_48k.wav")
        if run_audiosr(demucs_16k, audiosr_demucs_out):
            print(f"         -> {name}_4_demucs_audiosr_48k.wav")
        else:
            print(f"         -> ECHEC")
    else:
        print(f"  [4/4] SKIP (Demucs a echoue)")

    # Nettoyage tmp
    for tmp in [demucs_16k]:
        if os.path.exists(tmp):
            os.remove(tmp)

    print()

print("=== Termine ===")
print(f"Resultats dans : {OUT_DIR}")
print("Pour chaque fichier :")
print("  _1_raw_48k.wav            -> signal original (resample 48k)")
print("  _2_demucs_48k.wav         -> Demucs seul")
print("  _3_audiosr_raw_48k.wav    -> AudioSR seul (sur raw)")
print("  _4_demucs_audiosr_48k.wav -> Demucs puis AudioSR")
