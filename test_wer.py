"""
Test WER : compare les transcriptions Whisper de chaque variante du pipeline.
Utilise les transcriptions de reference disponibles dans Dataset Radio (2)/.
"""
import os, sys, glob
import whisper
import torch
from jiwer import wer, cer

sys.stdout.reconfigure(encoding="utf-8")

PIPELINE_DIR = os.path.join("output", "pipeline_comparison")
LABELS = {
    "_1_raw_48k.wav":            "RAW",
    "_2_demucs_48k.wav":         "Demucs",
    "_3_audiosr_raw_48k.wav":    "AudioSR",
    "_4_demucs_audiosr_48k.wav": "Demucs+AudioSR",
}

print("Chargement de Whisper (base)...")
model = whisper.load_model("base", device="cuda" if torch.cuda.is_available() else "cpu")

results = []

for folder in sorted(os.listdir(PIPELINE_DIR)):
    folder_path = os.path.join(PIPELINE_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    print(f"\n{'='*70}")
    print(f"  {folder}")
    print(f"{'='*70}")

    transcriptions = {}
    for suffix, label in LABELS.items():
        wav_path = os.path.join(folder_path, f"{folder}{suffix}")
        if not os.path.exists(wav_path):
            print(f"  {label:15s} : FICHIER MANQUANT")
            continue

        result = model.transcribe(wav_path, language="en")
        text = result["text"].strip()
        transcriptions[label] = text
        print(f"  {label:15s} : {text}")

    # Comparer les transcriptions entre elles (RAW comme reference)
    if "RAW" in transcriptions and len(transcriptions) > 1:
        print(f"\n  --- WER par rapport au RAW (Whisper sur RAW = reference) ---")
        ref = transcriptions["RAW"]
        for label, hyp in transcriptions.items():
            if label == "RAW":
                continue
            if ref and hyp:
                w = wer(ref.lower(), hyp.lower())
                c = cer(ref.lower(), hyp.lower())
                print(f"  {label:15s} : WER={w:.1%}  CER={c:.1%}")
                results.append((folder, label, w, c))

    # Si transcriptions sont tres differentes entre variantes, c'est un signe
    # que le traitement modifie l'intelligibilite
    print()

# Tableau recapitulatif
if results:
    print(f"\n{'='*70}")
    print(f"  RECAPITULATIF (WER par rapport a Whisper sur RAW)")
    print(f"{'='*70}")
    print(f"  {'Fichier':<20s} {'Pipeline':<15s} {'WER':>8s} {'CER':>8s}")
    print(f"  {'-'*55}")
    for folder, label, w, c in results:
        indicator = " <<" if w < 0.3 else " !!" if w > 0.7 else ""
        print(f"  {folder:<20s} {label:<15s} {w:>7.1%} {c:>7.1%}{indicator}")

    # Moyennes par pipeline
    from collections import defaultdict
    by_pipeline = defaultdict(list)
    for _, label, w, _ in results:
        by_pipeline[label].append(w)
    print(f"\n  --- Moyenne WER par pipeline ---")
    for label, wers in sorted(by_pipeline.items(), key=lambda x: sum(x[1])/len(x[1])):
        avg = sum(wers) / len(wers)
        print(f"  {label:<15s} : WER moyen = {avg:.1%}")

print(f"\nNote : WER bas = transcription similaire au RAW (pas forcement meilleur)")
print(f"       WER eleve = le traitement modifie beaucoup le signal")
print(f"       Pour un vrai WER il faudrait comparer a la transcription humaine")
