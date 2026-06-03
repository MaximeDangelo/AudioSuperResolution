"""
Preparation du dataset militaire aeronautique pour fine-tuning Whisper.

Etape 1 : extraction des dialogues du PDF (ACE4ACES / Thales Belgium)
Etape 2 : generation audio TTS via edge-tts (voix multiples)
Etape 3 : filtre bandpass 300-3400 Hz (simulation canal radio, 50% des paires)
Etape 4 : sauvegarde paires WAV + manifest.json
"""
import os
import sys
import re
import json
import random
import asyncio
import numpy as np
import soundfile as sf
import torchaudio
import torch
import pdfplumber
from scipy.signal import butter, sosfilt
import edge_tts

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PDF_PATH    = os.path.join(os.path.expanduser("~"), "Téléchargements",
                           "Orders & Commands to Fighter Pilots.pdf")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "data", "military_whisper")
AUDIO_DIR   = os.path.join(OUTPUT_DIR, "audio")
MANIFEST    = os.path.join(OUTPUT_DIR, "manifest.json")
SR          = 16000

os.makedirs(AUDIO_DIR, exist_ok=True)

# Voix edge-tts : diversite accent/genre
VOICES = [
    "en-US-GuyNeural",
    "en-US-AriaNeural",
    "en-GB-RyanNeural",
    "en-GB-SoniaNeural",
    "en-AU-WilliamNeural",
    "en-AU-NatashaNeural",
]

# Probabilite d'appliquer la degradation radio
RADIO_PROB = 0.5


# ------------------------------------------------------------------ #
#  Extraction PDF
# ------------------------------------------------------------------ #

def extract_dialogues_from_pdf(pdf_path):
    """
    Extrait les lignes de dialogue entre guillemets du PDF.
    Cible les lignes commencant par '-' contenant du texte entre guillemets.
    """
    dialogues = []
    # Patterns : guillemets droits ou typographiques
    quote_pattern = re.compile(r'["“”](.*?)["“”]', re.DOTALL)
    # Prefixes a supprimer avant d'isoler le texte
    prefix_pattern = re.compile(
        r'^[\-\–\•]\s*(?:'
        r'Controller(?:\s+\([^)]+\))?|Pilot(?:\s+\([^)]+\))?|'
        r'Lead|Wingman|Flight Lead|Mission Commander|Strike Lead|'
        r'JTAC|AWACS|FAC|Ground Forces?|Recovery Control|'
        r'EW Aircraft[^:]*|C2 HQ|Viper Lead|Eagle Lead|'
        r'[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*'
        r')?\s*(?:to\s+\w+)?\s*:?\s*',
        re.IGNORECASE
    )

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if not text:
                continue
            for line in text.split("\n"):
                line = line.strip()
                # Chercher les lignes avec des guillemets
                matches = quote_pattern.findall(line)
                for match in matches:
                    clean = match.strip()
                    # Filtres qualite
                    if len(clean) < 10:
                        continue
                    if len(clean) > 300:
                        continue
                    # Exclure les lignes qui semblent etre des titres ou notes
                    if clean.startswith("The ") and len(clean) > 150:
                        continue
                    dialogues.append(clean)

    # Deduplication
    seen = set()
    unique = []
    for d in dialogues:
        key = d.lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(d)

    return unique


# ------------------------------------------------------------------ #
#  Filtre bandpass radio (300 - 3400 Hz)
# ------------------------------------------------------------------ #

def bandpass_radio(audio_np, sr=SR, low=300, high=3400, order=6):
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio_np).astype(np.float32)


# ------------------------------------------------------------------ #
#  Generation TTS (async edge-tts)
# ------------------------------------------------------------------ #

async def tts_to_wav(text, voice, out_path_mp3):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path_mp3)


def mp3_to_wav_16k(mp3_path, wav_path):
    wav, sr = torchaudio.load(mp3_path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    audio_np = wav.squeeze().numpy().astype(np.float32)
    # Normalisation
    peak = np.abs(audio_np).max()
    if peak > 0:
        audio_np = audio_np / peak * 0.9
    sf.write(wav_path, audio_np, SR)
    return audio_np


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

async def main():
    print("=" * 60)
    print("Extraction des dialogues du PDF...")
    dialogues = extract_dialogues_from_pdf(PDF_PATH)
    print(f"  {len(dialogues)} dialogues extraits (apres deduplication)\n")

    if len(dialogues) == 0:
        print("[ERREUR] Aucun dialogue extrait. Verifier le chemin du PDF.")
        return

    # Apercu
    print("Apercu des 5 premiers dialogues :")
    for d in dialogues[:5]:
        print(f"  - {d}")
    print()

    manifest = []
    mp3_tmp = os.path.join(AUDIO_DIR, "_tmp.mp3")
    errors = 0

    print(f"Generation TTS + sauvegarde WAV pour {len(dialogues)} dialogues...")

    for i, text in enumerate(dialogues):
        voice = random.choice(VOICES)
        wav_name = f"{i:04d}.wav"
        wav_path = os.path.join(AUDIO_DIR, wav_name)

        try:
            # TTS -> MP3 temporaire
            await tts_to_wav(text, voice, mp3_tmp)
            # Conversion MP3 -> WAV 16kHz
            audio_np = mp3_to_wav_16k(mp3_tmp, wav_path)

            # Version avec degradation radio (bandpass) - 50% des paires
            apply_radio = random.random() < RADIO_PROB
            if apply_radio:
                audio_radio = bandpass_radio(audio_np)
                wav_radio_name = f"{i:04d}_radio.wav"
                wav_radio_path = os.path.join(AUDIO_DIR, wav_radio_name)
                sf.write(wav_radio_path, audio_radio, SR)
                manifest.append({
                    "audio": os.path.join("audio", wav_radio_name),
                    "text": text,
                    "voice": voice,
                    "degraded": True
                })
            else:
                manifest.append({
                    "audio": os.path.join("audio", wav_name),
                    "text": text,
                    "voice": voice,
                    "degraded": False
                })

            if (i + 1) % 50 == 0 or i == len(dialogues) - 1:
                print(f"  [{i+1}/{len(dialogues)}] {wav_name} | voice={voice} | radio={apply_radio}")

        except Exception as e:
            print(f"  [ERREUR] {i} : {e}")
            errors += 1
            continue

    # Nettoyage fichier temporaire
    if os.path.exists(mp3_tmp):
        os.remove(mp3_tmp)

    # Sauvegarde manifest
    with open(MANIFEST, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Dataset pret : {len(manifest)} paires audio/texte")
    print(f"Erreurs      : {errors}")
    print(f"Manifest     : {MANIFEST}")
    print(f"Audios       : {AUDIO_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
