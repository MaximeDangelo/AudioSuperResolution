"""
Transcription des fichiers radio reels avec Whisper-small.
Sauvegarde les transcriptions dans output/transcriptions/
"""
import os
import sys
import whisper
import soundfile as sf
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")

# === Configuration ===
AUDIO_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset Radio (2)")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "transcriptions")
MODEL_SIZE = "small"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Fichiers cibles (sans les copies)
FILES = [
    "01__kwahmah_atc001.flac",
    "02__kwahmah_atc002.flac",
    "03_kwahmah_atc003.flac",
    "05__kwahmah_atc005.flac",
    "06_kwahmah_atc006.flac",
    "07-wahmah_heathrow-air-traffic-control.flac",
    "08__kwahmah_hong-kong-air-traffic-control.flac",
]

print(f"Chargement de Whisper-{MODEL_SIZE}...")
model = whisper.load_model(MODEL_SIZE)
print(f"Modele charge.\n")

results = []

for fname in FILES:
    fpath = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(fpath):
        print(f"[SKIP] {fname} introuvable")
        continue

    name = os.path.splitext(fname)[0]
    print(f"{'='*60}")
    print(f"Fichier : {fname}")

    # Transcription
    result = model.transcribe(
        fpath,
        language="en",
        task="transcribe",
        verbose=False
    )

    text = result["text"].strip()
    print(f"Transcription :\n{text}\n")

    # Sauvegarde texte
    out_txt = os.path.join(OUTPUT_DIR, f"{name}.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(text)

    # Sauvegarde segments detailles
    out_seg = os.path.join(OUTPUT_DIR, f"{name}_segments.txt")
    with open(out_seg, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            start = seg["start"]
            end = seg["end"]
            seg_text = seg["text"].strip()
            f.write(f"[{start:6.2f}s -> {end:6.2f}s]  {seg_text}\n")

    results.append((fname, text))

# Rapport global
print(f"\n{'='*60}")
print("RAPPORT GLOBAL")
print(f"{'='*60}\n")

report_path = os.path.join(OUTPUT_DIR, "rapport_transcriptions.txt")
with open(report_path, "w", encoding="utf-8") as f:
    for fname, text in results:
        f.write(f"=== {fname} ===\n")
        f.write(text + "\n\n")

print(f"Transcriptions sauvegardees dans : {OUTPUT_DIR}")
print(f"Rapport global : {report_path}")
