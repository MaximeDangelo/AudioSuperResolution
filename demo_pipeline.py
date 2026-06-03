"""
Demo du pipeline de debruitage et transcription radio.
Utilisation : python demo_pipeline.py <fichier_audio>
Exemple     : python demo_pipeline.py "Dataset Radio (2)/02__kwahmah_atc002.flac"

Montre chaque etape du traitement et compare avant/apres.
100% local - aucune connexion web requise.
"""
import os
import sys
import re
import json
import collections
import datetime
import numpy as np
import torch
import torchaudio
import soundfile as sf
import noisereduce as nr
import whisper

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ATCO2_MODEL = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_atco2", "best")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "runs", "demo")
SR          = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Parametres pipeline
CHUNK_S     = 25
OVERLAP_S   = 2
CONF_THRESH = -1.0

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

POST_ASR = [
    (r'\bport final\b',          'report final',  re.IGNORECASE),
    (r'\bPort final\b',          'Report final',  0),
    (r'\bDelphi Lima\b',         'Delta Lima',    re.IGNORECASE),
    (r'\bHamburg\b',             'Hammer',        re.IGNORECASE),
    (r'\bSbebert\b',             'Speedbird',     re.IGNORECASE),
    (r'\bsbebert\b',             'speedbird',     0),
    (r'\bSleever\b',             'Speedbird',     re.IGNORECASE),
    (r'(\d{2,3})\s+decimal\s+(\d)', r'\1.\2',    re.IGNORECASE),
    (r'(\d{3}\.\d)\s+feet\b',    r'\1',           0),
]


def sep(char="=", n=65):
    print(char * n)

def header(title):
    sep()
    print(f"  {title}")
    sep()

def step(n, title):
    print(f"\n[Etape {n}] {title}")
    print("-" * 50)


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav.squeeze().numpy().astype(np.float32)


def speech_aware_compress(audio_np):
    rms = np.sqrt(np.mean(audio_np**2) + 1e-10)
    rms_db = 20.0 * np.log10(rms)
    if rms_db < -60:
        return audio_np
    gain_db = np.clip(-10.0 - rms_db, -40.0, 40.0)
    gain = 10.0 ** (gain_db / 20.0)
    out = audio_np * gain
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out / peak * 0.95
    return out.astype(np.float32), rms_db, rms_db + gain_db


def is_hallucination(text, lp, nl):
    if lp < CONF_THRESH:        return True, "confiance faible"
    if nl > 0.6:                return True, "silence detecte"
    tokens = text.lower().split()
    if len(tokens) >= 4:
        c = collections.Counter(tokens)
        top2 = sum(v for _, v in c.most_common(2))
        if top2 / len(tokens) > 0.80:
            return True, "motif alternant"
    if len(tokens) >= 8:
        h = len(tokens) // 2
        if tokens[:h] == tokens[h:h*2]:
            return True, "repetition"
    return False, None


def post_asr(text):
    corrections = []
    for pat, repl, flags in POST_ASR:
        new = re.sub(pat, repl, text, flags=flags)
        if new != text:
            corrections.append((re.search(pat, text, flags).group(0), repl))
            text = new
    return text, corrections


def transcribe_full(model, audio_np):
    chunk_len  = int(CHUNK_S   * SR)
    overlap_len = int(OVERLAP_S * SR)
    step_len   = chunk_len - overlap_len
    total_len  = len(audio_np)

    segs_out  = []
    seen_end  = 0.0
    start     = 0

    while start < total_len:
        end   = min(start + chunk_len, total_len)
        chunk = audio_np[start:end]
        off   = start / SR

        result = model.transcribe(chunk, language="en", task="transcribe",
                                   verbose=False, initial_prompt=MILITARY_PROMPT)
        for seg in result["segments"]:
            s = off + seg["start"]
            e = off + seg["end"]
            if s < seen_end - 0.5:
                continue
            txt = seg["text"].strip()
            lp  = seg.get("avg_logprob", 0.0)
            nl  = seg.get("no_speech_prob", 0.0)
            hallu, reason = is_hallucination(txt, lp, nl)
            if hallu:
                segs_out.append({"start": s, "end": e, "text": "",
                                 "raw": txt, "status": f"FILTRE ({reason})",
                                 "corrections": [], "logprob": lp})
                continue
            corrected, corrections = post_asr(txt)
            segs_out.append({"start": s, "end": e, "text": corrected,
                             "raw": txt, "status": "OK",
                             "corrections": corrections, "logprob": lp})
            seen_end = max(seen_end, e)

        if end == total_len:
            break
        start += step_len

    segs_out.sort(key=lambda x: x["start"])
    return segs_out


def main():
    # Chemin audio en argument ou fichier par defaut
    if len(sys.argv) > 1:
        audio_path = sys.argv[1]
        if not os.path.isabs(audio_path):
            audio_path = os.path.join(SCRIPT_DIR, audio_path)
    else:
        audio_path = os.path.join(SCRIPT_DIR, "Dataset Radio (2)", "02__kwahmah_atc002.flac")

    if not os.path.exists(audio_path):
        print(f"Fichier introuvable : {audio_path}")
        sys.exit(1)

    fname = os.path.basename(audio_path)
    name  = os.path.splitext(fname)[0]

    # ----------------------------------------------------------------
    header(f"DEMO PIPELINE AUDIO RADIO  |  {fname}")

    print(f"\n  Fichier  : {audio_path}")
    print(f"  Modele   : Whisper-small (local, ~/.cache/whisper/)")
    print(f"  Prompt   : ATC/NATO militaire (local)")
    print(f"  Sortie   : {OUTPUT_DIR}/")

    # Chargement
    step(1, "Chargement de l'audio")
    audio_raw = load_audio(audio_path)
    dur = len(audio_raw) / SR
    rms_raw = 20 * np.log10(np.sqrt(np.mean(audio_raw**2)) + 1e-10)
    print(f"  Duree   : {dur:.2f}s")
    print(f"  RMS     : {rms_raw:.1f} dBFS")
    print(f"  Samples : {len(audio_raw):,}  |  SR : {SR} Hz")
    sf.write(os.path.join(OUTPUT_DIR, f"{name}_1_raw.wav"), audio_raw, SR)
    print(f"  Sauvegarde : {name}_1_raw.wav")

    # Transcription brute (reference)
    step(2, "Transcription brute (AVANT traitement)")
    print("  Chargement Whisper small...", end="", flush=True)
    model = whisper.load_model("small")
    print(" OK")
    result_raw = model.transcribe(audio_raw[:int(30*SR)], language="en",
                                   task="transcribe", verbose=False)
    text_raw = result_raw["text"].strip()
    print(f"\n  >> {text_raw}")

    # Compression speech-aware
    step(3, "Compression speech-aware (alignement niveau RMS)")
    audio_comp, rms_before, rms_after = speech_aware_compress(audio_raw)
    gain_applied = rms_after - rms_before
    print(f"  RMS avant : {rms_before:.1f} dBFS")
    print(f"  RMS apres : {rms_after:.1f} dBFS  (cible : -10 dBFS)")
    print(f"  Gain applique : {gain_applied:+.1f} dB")
    sf.write(os.path.join(OUTPUT_DIR, f"{name}_2_compressed.wav"), audio_comp, SR)
    print(f"  Sauvegarde : {name}_2_compressed.wav")

    # Debruitage noisereduce
    step(4, "Debruitage noisereduce (spectral subtraction adaptative)")
    audio_nr = nr.reduce_noise(y=audio_comp, sr=SR, stationary=False,
                                prop_decrease=0.85, n_fft=512, time_constant_s=2.0
                                ).astype(np.float32)
    rms_nr = 20 * np.log10(np.sqrt(np.mean(audio_nr**2)) + 1e-10)
    snr_gain = rms_nr - rms_before
    print(f"  Methode : soustraction spectrale non-stationnaire")
    print(f"  Reduction bruit : 85% du bruit estime soustrait")
    print(f"  RMS signal : {rms_nr:.1f} dBFS")
    sf.write(os.path.join(OUTPUT_DIR, f"{name}_3_denoised.wav"), audio_nr, SR)
    print(f"  Sauvegarde : {name}_3_denoised.wav")

    # Transcription pipeline complet
    step(5, "Transcription avec fenetre glissante + anti-hallucination + corrections")
    print(f"  Fenetre : {CHUNK_S}s | Overlap : {OVERLAP_S}s")
    print(f"  Anti-hallucination : 4 indicateurs")
    print(f"  Corrections post-ASR : {len(POST_ASR)} regles\n")

    segs = transcribe_full(model, audio_nr)

    accepted = [s for s in segs if s["status"] == "OK"]
    filtered = [s for s in segs if s["status"] != "OK"]
    all_corrections = [c for s in accepted for c in s["corrections"]]

    print(f"  Segments traites  : {len(segs)}")
    print(f"  Segments acceptes : {len(accepted)}")
    print(f"  Hallucinations    : {len(filtered)}")
    print(f"  Corrections ASR   : {len(all_corrections)}")

    # Affichage segments horodates
    if len(segs) <= 20:
        print("\n  Segments horodates :")
        for s in segs:
            if s["status"] == "OK":
                corr_str = ""
                if s["corrections"]:
                    corr_str = "  [CORRIGE: " + ", ".join(f"'{a}'->'{b}'" for a,b in s["corrections"]) + "]"
                print(f"    [{s['start']:6.2f}s -> {s['end']:6.2f}s]  {s['text']}{corr_str}")
            else:
                print(f"    [{s['start']:6.2f}s -> {s['end']:6.2f}s]  {s['status']}")

    # Transcription finale
    text_final = " ".join(s["text"] for s in accepted).strip()

    sep("-")
    print("\n  TRANSCRIPTION FINALE (APRES pipeline) :")
    print(f"  >> {text_final}")
    sep("-")

    # Comparaison
    step(6, "Comparaison AVANT / APRES")
    print(f"  AVANT : {text_raw}")
    print()
    print(f"  APRES : {text_final}")

    if all_corrections:
        print("\n  Corrections post-ASR appliquees :")
        for a, b in set(all_corrections):
            print(f"    '{a}' -> '{b}'")

    # Sauvegarde JSON
    step(7, "Sauvegarde des resultats")
    ts = datetime.datetime.now().isoformat()
    output = {
        "file":         fname,
        "duration_s":   round(dur, 2),
        "processed_at": ts,
        "pipeline": {
            "compression_rms_db":   round(rms_after, 1),
            "gain_applied_db":       round(gain_applied, 1),
            "denoising":            "noisereduce (non-stationnaire, prop=0.85)",
            "asr_model":            "whisper-small + prompt ATC/NATO militaire",
            "chunk_s":              CHUNK_S,
            "overlap_s":            OVERLAP_S,
            "anti_hallucination":   "4 indicateurs",
            "post_asr_rules":       len(POST_ASR),
        },
        "transcription_raw":   text_raw,
        "transcription_final": text_final,
        "corrections_applied": list({(a, b) for a, b in all_corrections}),
        "stats": {
            "total_segments":    len(segs),
            "accepted_segments": len(accepted),
            "filtered_halluc":   len(filtered),
            "corrections":       len(all_corrections),
        },
        "segments": [
            {"start": round(s["start"], 2), "end": round(s["end"], 2),
             "text":  s["text"], "status": s["status"], "logprob": round(s["logprob"], 3)}
            for s in segs
        ],
    }

    json_path = os.path.join(OUTPUT_DIR, f"{name}_result.json")
    txt_path  = os.path.join(OUTPUT_DIR, f"{name}_result.txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"Pipeline audio radio - {fname}\n")
        f.write(f"Traite le : {ts}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"AVANT TRAITEMENT :\n{text_raw}\n\n")
        f.write(f"APRES PIPELINE :\n{text_final}\n\n")
        f.write("SEGMENTS HORODATES :\n")
        for s in segs:
            if s["status"] == "OK":
                f.write(f"[{s['start']:7.2f}s -> {s['end']:7.2f}s]  {s['text']}\n")
            else:
                f.write(f"[{s['start']:7.2f}s -> {s['end']:7.2f}s]  [{s['status']}]\n")

    print(f"  JSON          : {json_path}")
    print(f"  TXT horodate  : {txt_path}")
    print(f"  Audios (WAV)  : {OUTPUT_DIR}/{name}_1_raw.wav")
    print(f"                : {OUTPUT_DIR}/{name}_2_compressed.wav")
    print(f"                : {OUTPUT_DIR}/{name}_3_denoised.wav")

    sep()
    print("  DEMO TERMINEE")
    sep()


if __name__ == "__main__":
    main()
