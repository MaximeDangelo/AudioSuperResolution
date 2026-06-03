"""
Evaluation du Whisper fine-tune militaire sur fichiers radio reels.

Compare 3 approches :
  [A] Whisper-small baseline (aucun prompt)
  [B] Whisper-small + prompt militaire
  [C] Whisper-small fine-tune militaire (checkpoints/whisper_military/best/)

Resultats sauvegardes dans runs/whisper_military/eval_radio/
"""
import os
import sys
import json
import numpy as np
import torch
import torchaudio
import whisper
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
RADIO_DIR       = os.path.join(SCRIPT_DIR, "Dataset Radio (2)")
FINETUNED_DIR   = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_atco2", "best")
OUTPUT_DIR      = os.path.join(SCRIPT_DIR, "runs", "whisper_atco2", "eval_radio")
WHISPER_MODEL   = "small"
SR              = 16000
MAX_DURATION_S  = 60

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

MILITARY_PROMPT = (
    "Eagle 1, turn heading 270, climb to flight level 150, speed 350 knots. "
    "Viper 2, weapons free, engage bandit bearing 045, range 20 miles. "
    "Ghost 3, scramble immediately, bogey inbound bearing 180. "
    "Speedbird 232, descend to 9000 feet, report localizer established runway 27. "
    "Squawk 7700. RTB via direct route. Roger, copy, wilco, over and out. "
    "SAM launch detected bearing 090, deploy countermeasures, break right. "
    "IFF mode 4, no IFF response, maintain radar track. "
    "Weapons tight, hold fire until positive ID. "
    "Alpha flight, vector 045 to intercept, altitude 20000 feet. "
    "JTAC, request BDA on target grid 43NK, over. "
    "Hammer 2-1, priority one is enemy armor, cleared hot. "
    "Link 16 net, switch to guard frequency 243.0. "
    "Reduce speed 180 knots, report final runway 23. "
    "Descend altitude 4000 feet, speed 180 knots."
)


def load_mono_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    return wav.squeeze(0).numpy().astype(np.float32)


def transcribe_baseline(model, audio_np, use_prompt=False):
    result = model.transcribe(
        audio_np,
        language="en",
        task="transcribe",
        verbose=False,
        initial_prompt=MILITARY_PROMPT if use_prompt else None,
    )
    return result["text"].strip()


def transcribe_finetuned(processor, model, audio_np, device):
    """Transcription avec le modele fine-tune HuggingFace."""
    inputs = processor(audio_np, sampling_rate=SR, return_tensors="pt")
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="en",
            task="transcribe",
        )
    return processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    max_samples = int(MAX_DURATION_S * SR)

    # --- Passe 1 : baseline (openai-whisper) ---
    print(f"Chargement Whisper-{WHISPER_MODEL} baseline...")
    wh_baseline = whisper.load_model(WHISPER_MODEL)
    print("  OK\n")

    audios = {}
    pass1  = {}
    for fname in FILES:
        fpath = os.path.join(RADIO_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[SKIP] {fname} introuvable")
            continue
        audio_np = load_mono_16k(fpath)
        if len(audio_np) > max_samples:
            audio_np = audio_np[:max_samples]
        audios[fname] = audio_np

        name = os.path.splitext(fname)[0]
        print(f"[A/B] {fname} ({len(audio_np)/SR:.1f}s)")
        text_a = transcribe_baseline(wh_baseline, audio_np, use_prompt=False)
        text_b = transcribe_baseline(wh_baseline, audio_np, use_prompt=True)
        pass1[fname] = (text_a, text_b)
        print(f"  [A] {text_a[:100]}")
        print(f"  [B] {text_b[:100]}\n")

    # Liberer le modele baseline avant de charger le fine-tune
    del wh_baseline
    torch.cuda.empty_cache()
    import gc; gc.collect()
    print("Baseline libere, chargement fine-tune...")

    # --- Passe 2 : fine-tune (HuggingFace) ---
    print(f"Chargement Whisper fine-tune : {FINETUNED_DIR}...")
    ft_processor = WhisperProcessor.from_pretrained(FINETUNED_DIR)
    ft_model     = WhisperForConditionalGeneration.from_pretrained(FINETUNED_DIR).to(device)
    ft_model.eval()
    print("  OK\n")

    results = []
    for fname in FILES:
        if fname not in audios:
            continue
        audio_np      = audios[fname]
        text_a, text_b = pass1[fname]
        name           = os.path.splitext(fname)[0]
        duration       = len(audio_np) / SR

        print(f"[C] {fname}")
        text_c = transcribe_finetuned(ft_processor, ft_model, audio_np, device)
        print(f"  [A] Baseline  : {text_a[:100]}")
        print(f"  [B] + prompt  : {text_b[:100]}")
        print(f"  [C] Fine-tune : {text_c[:100]}\n")

        for suffix, text in [("baseline", text_a), ("prompt", text_b), ("finetuned", text_c)]:
            with open(os.path.join(OUTPUT_DIR, f"{name}_{suffix}.txt"), "w", encoding="utf-8") as f:
                f.write(text)

        results.append({
            "file":      fname,
            "duration":  round(duration, 1),
            "baseline":  text_a,
            "prompt":    text_b,
            "finetuned": text_c,
        })

    # Rapport global JSON
    report_json = os.path.join(OUTPUT_DIR, "comparison_results.json")
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Rapport lisible
    report_txt = os.path.join(OUTPUT_DIR, "rapport_comparaison.txt")
    with open(report_txt, "w", encoding="utf-8") as f:
        f.write("RAPPORT COMPARAISON TRANSCRIPTION - WHISPER MILITAIRE\n")
        f.write("[A] Baseline  |  [B] + prompt militaire  |  [C] Fine-tune militaire\n")
        f.write("=" * 70 + "\n\n")
        for r in results:
            f.write(f"=== {r['file']} ({r['duration']}s) ===\n")
            f.write(f"[A] Baseline  : {r['baseline']}\n")
            f.write(f"[B] Prompt    : {r['prompt']}\n")
            f.write(f"[C] Fine-tune : {r['finetuned']}\n\n")

    print("=" * 70)
    print(f"Resultats JSON  : {report_json}")
    print(f"Rapport texte   : {report_txt}")
    print(f"Fichiers TXT    : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
