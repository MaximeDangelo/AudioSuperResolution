"""
Fine-tuning Whisper-small sur le dataset militaire aeronautique.
Base : openai/whisper-small
Data : data/military_whisper/manifest.json (272 paires audio/texte TTS)
Checkpoint : checkpoints/whisper_military/best/

Ameliorations v2 :
  - Encodeur gele (seul le decodeur est fine-tune) => moins d'overfitting
  - Augmentation radio sur 100% des samples (bandpass + bruit blanc)
  - Audio de base propre utilise comme source (normalisation du chemin)
  - 10 epochs, LR 3e-5
"""
import os
import sys
import json
import random
import numpy as np
import torch
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from scipy.signal import butter, sosfilt

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import Dataset
import evaluate

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MANIFEST     = os.path.join(SCRIPT_DIR, "data", "military_whisper", "manifest.json")
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "military_whisper")
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_military")
LOG_DIR      = os.path.join(SCRIPT_DIR, "runs", "whisper_military")
MODEL_ID     = "openai/whisper-small"
SR           = 16000
VAL_RATIO    = 0.15
SEED         = 42

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ #
#  Simulation canal radio (augmentation)
# ------------------------------------------------------------------ #

def bandpass_radio(audio_np, sr=SR, low=300, high=3400, order=5):
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio_np).astype(np.float32)


def add_radio_noise(audio_np, snr_db_range=(15, 30)):
    snr_db = random.uniform(*snr_db_range)
    signal_rms = np.sqrt(np.mean(audio_np ** 2) + 1e-10)
    noise = np.random.randn(len(audio_np)).astype(np.float32)
    noise = bandpass_radio(noise)  # bruit filtre dans la bande radio
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)
    return (audio_np + noise).astype(np.float32)


def augment_radio(audio_np):
    """Simule le canal radio : bandpass 300-3400 Hz + bruit blanc filtre."""
    audio = bandpass_radio(audio_np)
    audio = add_radio_noise(audio)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.85
    return audio


def clean_audio_path(path):
    """Retourne le chemin vers l'audio TTS propre (sans le suffixe _radio)."""
    if path.endswith("_radio.wav"):
        base = path[:-len("_radio.wav")] + ".wav"
        if os.path.exists(base):
            return base
    return path


# ------------------------------------------------------------------ #
#  Chargement des donnees
# ------------------------------------------------------------------ #

def load_manifest():
    with open(MANIFEST, "r", encoding="utf-8") as f:
        entries = json.load(f)
    for e in entries:
        raw_path = os.path.join(DATA_DIR, e["audio"])
        e["audio"] = clean_audio_path(raw_path)
    return entries


def load_audio(path):
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def build_hf_dataset(entries, processor):
    all_features, all_labels = [], []
    for e in entries:
        try:
            audio = load_audio(e["audio"])
            audio = augment_radio(audio)
            inputs = processor(audio, sampling_rate=SR, return_tensors="np")
            labels = processor.tokenizer(e["text"], return_tensors="np").input_ids[0]
            all_features.append(inputs.input_features[0])
            all_labels.append(labels)
        except Exception as ex:
            print(f"  [SKIP] {e['audio']} : {ex}")
    return Dataset.from_dict({"input_features": all_features, "labels": all_labels})


# ------------------------------------------------------------------ #
#  Collator et metriques
# ------------------------------------------------------------------ #

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def make_compute_metrics(processor):
    wer_metric = evaluate.load("wer")

    def compute_metrics(pred):
        pred_ids  = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str  = processor.batch_decode(pred_ids,  skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        wer = wer_metric.compute(predictions=pred_str, references=label_str)
        return {"wer": round(wer, 4)}

    return compute_metrics


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")

    entries = load_manifest()
    random.shuffle(entries)
    split   = int(len(entries) * (1 - VAL_RATIO))
    train_e = entries[:split]
    val_e   = entries[split:]
    print(f"Train : {len(train_e)} | Val : {len(val_e)}")

    print(f"\nChargement {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    # Geler l'encodeur : seul le decodeur est fine-tune (biais vocabulaire)
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Encodeur gele. Params entrainables : {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)")

    print("Construction des datasets (avec augmentation radio)...")
    train_ds = build_hf_dataset(train_e, processor)
    val_ds   = build_hf_dataset(val_e,   processor)
    print(f"  Train : {len(train_ds)} | Val : {len(val_ds)}")

    collator = DataCollatorSpeechSeq2Seq(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_steps=20,
        num_train_epochs=10,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_dir=LOG_DIR,
        logging_steps=10,
        report_to="none",
        fp16=False,
        dataloader_num_workers=0,
        seed=SEED,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
    )

    print("\nDemarrage du fine-tuning (encodeur gele, augmentation radio)...")
    trainer.train()

    # Sauvegarde du modele final
    best_dir = os.path.join(CKPT_DIR, "best")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"\nModele sauvegarde : {best_dir}")

    # Evaluation et sauvegarde des resultats
    print("\nEvaluation finale...")
    results = trainer.evaluate()
    print(f"  WER final : {results.get('eval_wer', 'N/A')}")

    results_path = os.path.join(LOG_DIR, "results_v2.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Resultats : {results_path}")


if __name__ == "__main__":
    main()
