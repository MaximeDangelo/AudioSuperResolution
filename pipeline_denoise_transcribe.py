"""
Pipeline de debruitage et transcription - 100% local, aucune connexion web.

Composantes :
  1. Compression speech-aware (RMS -10 dB, gain max 40 dB, plancher -60 dB)
  2. Debruitage noisereduce (spectral subtraction adaptative)
  3. Transcription Whisper (baseline small + prompt ATC/militaire)
  4. Fenetre glissante (chunks 25s, overlap 2s) pour fichiers longs
  5. Anti-hallucination : 4 indicateurs (logprob, no_speech, tokens alternants, repetition)
  6. Corrections post-ASR : callsigns, frequences, phraseologie
  7. Sortie JSON horodatee + TXT + WAV

Modeles utilises (tous locaux) :
  - openai/whisper-small  : dans ~/.cache/whisper/ (telecharge une seule fois)
  - checkpoints/whisper_atco2/best/ : modele fine-tune local

Resultats : runs/pipeline_final/
"""
import os
import sys
import re
import json
import datetime
import collections
import numpy as np
import torch
import torchaudio
import soundfile as sf
import noisereduce as nr
import whisper
# transformers non utilise dans ce pipeline (reserve au fine-tuning)

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RADIO_DIR   = os.path.join(SCRIPT_DIR, "Dataset Radio (2)")
ATCO2_MODEL = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_atco2", "best")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "runs", "pipeline_final")
SR          = 16000

CHUNK_S     = 25
OVERLAP_S   = 2
CONF_THRESH = -1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

FILES = [
    "01__kwahmah_atc001.flac",
    "02__kwahmah_atc002.flac",
    "03_kwahmah_atc003.flac",
    "05__kwahmah_atc005.flac",
    "06_kwahmah_atc006.flac",
    "07-wahmah_heathrow-air-traffic-control.flac",
    "08__kwahmah_hong-kong-air-traffic-control.flac",
]

# Prompt ATC/militaire (local, pas de web)
MILITARY_PROMPT = (
    "Eagle one, vector zero four five, bogey at bearing one eight zero, range thirty miles. "
    "Viper two, weapons free, engage bandit, break right. "
    "SAM launch detected, deploy countermeasures, break left. "
    "IFF mode four, no response, track as hostile, weapons tight. "
    "Bingo fuel, RTB, squawk seven seven zero zero. Mayday mayday mayday. "
    "JTAC, cleared hot, BDA requested. Fox three away. Tally, no joy, splash. "
    "Fence in, fence out, joker state. Link sixteen, guard frequency two four three decimal zero. "
    "Speedbird two three two, right heading two three five, report localizer established runway two seven. "
    "KLM four four mike, descend altitude four thousand feet, speed one eight zero knots. "
    "Delta Lima, report final runway two three. Squawk seven seven zero zero."
)

# ------------------------------------------------------------------ #
#  1. Compression speech-aware (TFE : -10 dB RMS, gain max 40 dB)
# ------------------------------------------------------------------ #

def speech_aware_compress(audio_np, target_rms_db=-10.0, max_gain_db=40.0, floor_db=-60.0):
    """
    Aligne le niveau RMS vers -10 dBFS.
    Limite le gain max a 40 dB (evite d'amplifier le silence).
    Ne traite pas les signaux en dessous du plancher -60 dB.
    """
    rms = np.sqrt(np.mean(audio_np ** 2) + 1e-10)
    rms_db = 20.0 * np.log10(rms)

    if rms_db < floor_db:
        return audio_np  # signal trop faible, ne pas amplifier

    gain_db = np.clip(target_rms_db - rms_db, -max_gain_db, max_gain_db)
    gain    = 10.0 ** (gain_db / 20.0)
    out     = audio_np * gain

    # Ecreter si necessaire
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out / peak * 0.95
    return out.astype(np.float32)


# ------------------------------------------------------------------ #
#  2. Debruitage noisereduce
# ------------------------------------------------------------------ #

def denoise_nr(audio_np, sr=SR):
    return nr.reduce_noise(
        y=audio_np, sr=sr, stationary=False,
        prop_decrease=0.85, n_fft=512, time_constant_s=2.0,
    ).astype(np.float32)


# ------------------------------------------------------------------ #
#  3. Anti-hallucination : 4 indicateurs (TFE section 4.2)
# ------------------------------------------------------------------ #

def is_hallucination(text, avg_logprob, no_speech_prob):
    """
    Detecte les hallucinations Whisper selon 4 criteres :
      1. Confiance acoustique faible (avg_logprob)
      2. Probabilite de silence elevee (no_speech_prob)
      3. Motifs alternants : 2 tokens les plus frequents > 80% du texte
      4. Repetition de sous-sequences (pattern 'ABCABC')
    """
    # Indicateur 1 : confiance acoustique
    if avg_logprob < CONF_THRESH:
        return True, "logprob"

    # Indicateur 2 : silence
    if no_speech_prob > 0.6:
        return True, "silence"

    tokens = text.lower().split()
    if not tokens:
        return True, "vide"

    # Indicateur 3 : motifs alternants
    if len(tokens) >= 4:
        counts = collections.Counter(tokens)
        top2   = sum(v for _, v in counts.most_common(2))
        if top2 / len(tokens) > 0.80:
            return True, "alternant"

    # Indicateur 4 : repetition de sous-sequences
    if len(tokens) >= 8:
        half = len(tokens) // 2
        if tokens[:half] == tokens[half: half * 2]:
            return True, "repetition"

    return False, None


# ------------------------------------------------------------------ #
#  4. Corrections post-ASR (callsigns, frequences, phraseologie)
# ------------------------------------------------------------------ #

# Confusions phonetiques frequentes de Whisper sur les communications radio
POST_ASR = [
    # Phraseologie
    (r'\bport final\b',          'report final',       re.IGNORECASE),
    (r'\bPort final\b',          'Report final',       0),
    (r'\bno joy\b',              'no joy',             re.IGNORECASE),
    # Callsigns mal reconnus
    (r'\bDelphi Lima\b',         'Delta Lima',         re.IGNORECASE),
    (r'\bAlpha Fight\b',         'Alpha flight',       re.IGNORECASE),
    (r'\bHamburg\b',             'Hammer',             re.IGNORECASE),
    (r'\bSbebert\b',             'Speedbird',          re.IGNORECASE),
    (r'\bsbebert\b',             'speedbird',          0),
    (r'\bSliver\b',              'Speedbird',          re.IGNORECASE),
    (r'\bSleever\b',             'Speedbird',          re.IGNORECASE),
    # Frequences : "121 decimal 7" -> "121.7"
    (r'\b(\d{2,3})\s+decimal\s+(\d)\b', r'\1.\2',     re.IGNORECASE),
    # Format OACI : supprimer "feet" apres une frequence radio
    (r'(\d{3}\.\d)\s+feet\b',    r'\1',                0),
    # Altitudes communes mal epelees
    (r'\bangels\s+(\w+)\b',      r'angels \1',         re.IGNORECASE),
]

def post_asr_correct(text):
    """Applique les corrections post-ASR par substitution regex."""
    for pattern, repl, flags in POST_ASR:
        text = re.sub(pattern, repl, text, flags=flags)
    return text


# ------------------------------------------------------------------ #
#  5. Chargement audio
# ------------------------------------------------------------------ #

def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav.squeeze().numpy().astype(np.float32)


# ------------------------------------------------------------------ #
#  6. Transcription avec fenetre glissante + anti-hallucination
# ------------------------------------------------------------------ #

def transcribe_pipeline(model, audio_np, sr=SR):
    """
    Transcription complete :
      - Fenetre glissante (25s / overlap 2s)
      - 4 indicateurs anti-hallucination
      - Corrections post-ASR
      - Retourne texte + segments horodates
    """
    chunk_len   = int(CHUNK_S   * sr)
    overlap_len = int(OVERLAP_S * sr)
    step_len    = chunk_len - overlap_len
    total_len   = len(audio_np)

    segments_out   = []   # (start_s, end_s, text, rejected_reason)
    seen_end       = 0.0
    halluc_count   = 0

    start = 0
    while start < total_len:
        end      = min(start + chunk_len, total_len)
        chunk    = audio_np[start:end]
        offset_s = start / sr

        result = model.transcribe(chunk, language="en", task="transcribe",
                                   verbose=False, initial_prompt=MILITARY_PROMPT)

        for seg in result["segments"]:
            seg_start = offset_s + seg["start"]
            seg_end   = offset_s + seg["end"]
            text_raw  = seg["text"].strip()
            lp = seg.get("avg_logprob", 0.0)
            nl = seg.get("no_speech_prob", 0.0)

            # Deduplication overlap
            if seg_start < seen_end - 0.5:
                continue

            # Anti-hallucination (4 indicateurs)
            hallu, reason = is_hallucination(text_raw, lp, nl)
            if hallu:
                halluc_count += 1
                segments_out.append({
                    "start": round(seg_start, 2),
                    "end":   round(seg_end,   2),
                    "text":  "",
                    "raw":   text_raw,
                    "rejected": reason,
                    "logprob": round(lp, 3),
                })
                continue

            # Corrections post-ASR
            text_corrected = post_asr_correct(text_raw)

            segments_out.append({
                "start":    round(seg_start, 2),
                "end":      round(seg_end,   2),
                "text":     text_corrected,
                "raw":      text_raw,
                "rejected": None,
                "logprob":  round(lp, 3),
            })
            seen_end = max(seen_end, seg_end)

        if end == total_len:
            break
        start += step_len

    segments_out.sort(key=lambda s: s["start"])
    accepted   = [s for s in segments_out if s["rejected"] is None]
    full_text  = " ".join(s["text"] for s in accepted).strip()
    return full_text, segments_out, halluc_count


# ------------------------------------------------------------------ #
#  7. Main
# ------------------------------------------------------------------ #

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device      : {device}")
    print(f"Modeles     : locaux uniquement (aucune connexion web)")
    print(f"Chunk       : {CHUNK_S}s | Overlap : {OVERLAP_S}s")
    print(f"Anti-halluc : 4 indicateurs (logprob / silence / alternant / repetition)\n")

    print("Chargement Whisper small (local)...")
    model = whisper.load_model("small")
    print("OK\n")

    all_results = []

    for fname in FILES:
        fpath = os.path.join(RADIO_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[SKIP] {fname}")
            continue

        name = os.path.splitext(fname)[0]
        print("=" * 70)
        print(f"Fichier : {fname}")

        # Chargement
        audio_raw = load_audio(fpath)
        dur       = len(audio_raw) / SR
        print(f"  Duree : {dur:.1f}s")

        # Etape 1 : compression speech-aware
        audio_comp = speech_aware_compress(audio_raw)
        rms_before = 20 * np.log10(np.sqrt(np.mean(audio_raw**2)) + 1e-10)
        rms_after  = 20 * np.log10(np.sqrt(np.mean(audio_comp**2)) + 1e-10)
        print(f"  Compression : {rms_before:.1f} dB -> {rms_after:.1f} dB RMS")

        # Etape 2 : debruitage noisereduce
        print("  Debruitage noisereduce...", end="", flush=True)
        audio_nr = denoise_nr(audio_comp)
        print(" OK")

        # Etape 3 : transcription pipeline complet
        print("  Transcription (sliding + anti-halluc + post-ASR)...", end="", flush=True)
        text_final, segments, n_halluc = transcribe_pipeline(model, audio_nr)
        n_accepted = sum(1 for s in segments if s["rejected"] is None)
        print(f" OK ({n_accepted} segments acceptes, {n_halluc} hallucinations filtrees)")

        print(f"\n  Transcription : {text_final[:120]}\n")

        # Sauvegarde WAV nettoye
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_clean.wav"), audio_nr, SR)

        # Sauvegarde JSON horodate (format TFE)
        ts = datetime.datetime.now().isoformat()
        output_json = {
            "file":        fname,
            "duration_s":  round(dur, 1),
            "processed_at": ts,
            "pipeline": {
                "compression_rms_db": round(rms_after, 1),
                "denoising": "noisereduce",
                "asr_model": "whisper-small + prompt ATC/NATO",
                "anti_hallucination": "4 indicateurs (logprob/silence/alternant/repetition)",
                "post_asr_corrections": len(POST_ASR),
            },
            "transcription": text_final,
            "segments": segments,
            "stats": {
                "total_segments":    len(segments),
                "accepted_segments": n_accepted,
                "rejected_halluc":   n_halluc,
            },
        }

        json_path = os.path.join(OUTPUT_DIR, f"{name}_transcription.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output_json, f, ensure_ascii=False, indent=2)

        # Sauvegarde TXT simple
        txt_path = os.path.join(OUTPUT_DIR, f"{name}_transcription.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Fichier : {fname}\n")
            f.write(f"Duree   : {dur:.1f}s\n")
            f.write(f"Traite le : {ts}\n\n")
            f.write("--- Transcription ---\n")
            f.write(text_final + "\n\n")
            f.write("--- Segments horodates ---\n")
            for s in segments:
                if s["rejected"] is None:
                    f.write(f"[{s['start']:7.2f}s -> {s['end']:7.2f}s]  {s['text']}\n")
                else:
                    f.write(f"[{s['start']:7.2f}s -> {s['end']:7.2f}s]  [FILTRE:{s['rejected']}] {s['raw'][:50]}\n")

        all_results.append(output_json)

    # Rapport global
    report_path = os.path.join(OUTPUT_DIR, "rapport_global.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("=" * 70)
    print(f"Resultats : {OUTPUT_DIR}")
    print(f"Fichiers generes par audio : _clean.wav  _transcription.json  _transcription.txt")
    print(f"Rapport global : rapport_global.json")


if __name__ == "__main__":
    main()
