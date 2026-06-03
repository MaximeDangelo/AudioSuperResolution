"""
Evaluation de modeles Whisper pre-entraines ATC sur nos fichiers radio reels.

Compare :
  [A] Whisper-small baseline (openai/whisper-small)
  [B] Notre fine-tune ATCO2 2000 (checkpoints/whisper_atco2/best/)
  [C] luigisaetta/whisper-atco2-medium  (medium, entraine sur ATCO2 complet)
  [D] jlvdoorn/whisper-large-v2-atco2-asr-atcosim (large-v2, ATCO2+ATCOSIM complet)

Les modeles sont charges et liberes sequentiellement pour eviter les OOM.
Resultats : runs/eval_pretrained_atc/
"""
import os
import sys
import json
import gc
import numpy as np
import torch
import torchaudio
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
RADIO_DIR    = os.path.join(SCRIPT_DIR, "Dataset Radio (2)")
OUR_MODEL    = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_atco2", "best")
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "runs", "eval_pretrained_atc")
SR           = 16000
MAX_DURATION = 60

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

MODELS = [
    ("A", "small",                            "Baseline small (openai)"),
    ("B", OUR_MODEL,                          "Notre ATCO2-2000 small"),
    ("C", "luigisaetta/whisper-atco2-medium", "luigisaetta medium (ATCO2 complet)"),
]

MILITARY_PROMPT = (
    "Eagle 1, turn heading 270, climb to flight level 150, speed 350 knots. "
    "Speedbird 232, descend to 9000 feet, report localizer established runway 27. "
    "Squawk 7700. Roger, copy, wilco. KLM, descend altitude 4000 feet, speed 180 knots."
)


def load_audio(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    audio = wav.squeeze().numpy().astype(np.float32)
    max_samples = MAX_DURATION * SR
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    return audio


def transcribe_hf(model_id_or_path, audio_np, device):
    """Charge, transcrit, libere le modele HuggingFace."""
    proc  = WhisperProcessor.from_pretrained(model_id_or_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_id_or_path).to(device)
    model.eval()

    inputs = proc(audio_np, sampling_rate=SR, return_tensors="pt")
    # Compatibilite avec les anciennes versions de transformers (pas de lang_to_id)
    try:
        forced = proc.get_decoder_prompt_ids(language="english", task="transcribe")
    except Exception:
        forced = None
    with torch.no_grad():
        gen_kwargs = {"max_new_tokens": 448}
        if forced:
            gen_kwargs["forced_decoder_ids"] = forced
        ids = model.generate(inputs.input_features.to(device), **gen_kwargs)
    text = proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

    del model, proc, inputs, ids
    torch.cuda.empty_cache()
    gc.collect()
    return text


def transcribe_openai(model_id, audio_np):
    """Charge, transcrit, libere le modele openai-whisper."""
    model = whisper.load_model(model_id)
    result = model.transcribe(audio_np, language="en", task="transcribe",
                              verbose=False, initial_prompt=None)
    text = result["text"].strip()
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    # Chargement des audios
    audios = {}
    for fname in FILES:
        path = os.path.join(RADIO_DIR, fname)
        if not os.path.exists(path):
            print(f"[SKIP] {fname}")
            continue
        audios[fname] = load_audio(path)
        dur = len(audios[fname]) / SR
        print(f"Charge : {fname} ({dur:.1f}s)")
    print()

    # Structure : results[fname][label] = texte
    results = {f: {} for f in audios}

    for label, model_id, desc in MODELS:
        is_local = os.path.isdir(model_id)
        # openai-whisper prend des noms courts : tiny, base, small, medium, large...
        is_openai_baseline = not is_local and not model_id.startswith("openai/") and "/" not in model_id

        print(f"{'='*70}")
        print(f"[{label}] {desc}")
        print(f"    {model_id}")

        for fname, audio_np in audios.items():
            print(f"  {fname}...", end="", flush=True)
            try:
                if is_openai_baseline:
                    text = transcribe_openai(model_id, audio_np)
                else:
                    text = transcribe_hf(model_id if not is_local else model_id, audio_np, device)
                results[fname][label] = text
                print(f" {text[:80]}")
            except Exception as e:
                results[fname][label] = f"[ERREUR: {e}]"
                print(f" ERREUR: {e}")
        print()

    # Rapport JSON
    report_json = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Rapport texte
    report_txt = os.path.join(OUTPUT_DIR, "rapport_comparaison.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("COMPARAISON MODELES WHISPER ATC - FICHIERS RADIO REELS\n")
        f.write("=" * 70 + "\n\n")
        for fname, preds in results.items():
            f.write(f"=== {fname} ===\n")
            for label, model_id, desc in MODELS:
                text = preds.get(label, "N/A")
                f.write(f"[{label}] {desc[:30]:<30s} : {text}\n")
            f.write("\n")

    # Affichage final
    print("=" * 70)
    print("RECAPITULATIF PAR FICHIER\n")
    for fname, preds in results.items():
        print(f"--- {fname} ---")
        for label, _, desc in MODELS:
            t = preds.get(label, "N/A")
            print(f"  [{label}] {t[:90]}")
        print()

    print(f"Rapport JSON  : {report_json}")
    print(f"Rapport texte : {report_txt}")


if __name__ == "__main__":
    main()
