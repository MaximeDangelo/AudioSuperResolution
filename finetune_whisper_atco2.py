"""
Fine-tuning Whisper-small sur donnees ATC reelles (ATCO2 + militaire TTS).

Sources :
  - luigisaetta/atco2_atcosim  : 8142 segments ATC reels (LiveATC, 16 kHz)
  - data/military_whisper/     : 272 paires TTS militaires (avec augmentation radio)

Ameliorations vs v1/v2 :
  - Donnees reelles : vraie acoustique radio (pas du TTS propre)
  - Encodeur gele : seul le decodeur est fine-tune (biais vocabulaire)
  - Streaming HuggingFace : pas de stockage local des donnees ATC
  - Melange ATCO2 + militaire : vocabulaire ATC et militaire
"""
import os
import sys
import io
import json
import random
import numpy as np
import torch
import soundfile as sf
from dataclasses import dataclass
from typing import Any, Dict, List
from scipy.signal import butter, sosfilt

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from datasets import load_dataset, Audio
import evaluate

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
MANIFEST     = os.path.join(SCRIPT_DIR, "data", "military_whisper", "manifest.json")
DATA_DIR     = os.path.join(SCRIPT_DIR, "data", "military_whisper")
CKPT_DIR     = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_atco2")
LOG_DIR      = os.path.join(SCRIPT_DIR, "runs", "whisper_atco2")
MODEL_ID     = "openai/whisper-small"
HF_DATASET   = "luigisaetta/atco2_atcosim"
SR           = 16000
MAX_ATCO2    = 2000   # nb echantillons ATCO2 utilises
VAL_RATIO    = 0.10
SEED         = 42

os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR,  exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------------ #
#  Augmentation radio (pour les paires TTS militaires uniquement)
# ------------------------------------------------------------------ #

def bandpass_radio(audio_np, sr=SR, low=300, high=3400, order=5):
    sos = butter(order, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio_np).astype(np.float32)


def add_radio_noise(audio_np, snr_db_range=(15, 30)):
    snr_db = random.uniform(*snr_db_range)
    signal_rms = np.sqrt(np.mean(audio_np ** 2) + 1e-10)
    noise = np.random.randn(len(audio_np)).astype(np.float32)
    noise = bandpass_radio(noise)
    noise_rms = np.sqrt(np.mean(noise ** 2) + 1e-10)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    noise = noise * (target_noise_rms / noise_rms)
    return (audio_np + noise).astype(np.float32)


def augment_radio(audio_np):
    audio = bandpass_radio(audio_np)
    audio = add_radio_noise(audio)
    peak = np.abs(audio).max()
    if peak > 0:
        audio = audio / peak * 0.85
    return audio


def clean_audio_path(path):
    if path.endswith("_radio.wav"):
        base = path[:-len("_radio.wav")] + ".wav"
        if os.path.exists(base):
            return base
    return path


# ------------------------------------------------------------------ #
#  Chargement des donnees
# ------------------------------------------------------------------ #

def load_atco2_samples(n_max, processor):
    """Charge n_max echantillons ATCO2 reels depuis HuggingFace (streaming)."""
    print(f"Chargement ATCO2 depuis HuggingFace ({HF_DATASET})...")
    ds = load_dataset(HF_DATASET, split="train", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))

    import gc
    features, labels, skipped = [], [], 0
    for i, sample in enumerate(ds):
        if len(features) >= n_max:
            break
        try:
            audio_raw = sample["audio"]
            audio_np, sr = sf.read(io.BytesIO(audio_raw["bytes"]))
            if audio_np.ndim > 1:
                audio_np = audio_np.mean(axis=1)
            audio_np = audio_np.astype(np.float32)

            # Resample si necessaire
            if sr != SR:
                import torchaudio
                wav_t = torch.tensor(audio_np).unsqueeze(0)
                wav_t = torchaudio.transforms.Resample(sr, SR)(wav_t)
                audio_np = wav_t.squeeze().numpy()

            # Texte : ATCO2 est en minuscules sans ponctuation
            text = sample.get("sentence", sample.get("text", "")).strip()
            if not text or len(audio_np) < SR * 0.5:
                skipped += 1
                continue

            inp = processor(audio_np, sampling_rate=SR, return_tensors="np")
            lbl = processor.tokenizer(text, return_tensors="np").input_ids[0]
            features.append(inp.input_features[0].astype(np.float16))  # float16 = moitie moins de RAM
            labels.append(lbl)
            del audio_np, inp  # liberer la memoire immediatement

            if len(features) % 500 == 0:
                gc.collect()
                print(f"  {len(features)}/{n_max} charges...")
        except Exception as e:
            skipped += 1
    gc.collect()
    print(f"  {len(features)} echantillons ATCO2 charges | {skipped} ignores")
    return features, labels


def load_military_samples(processor):
    """Charge les paires TTS militaires avec augmentation radio."""
    with open(MANIFEST, "r", encoding="utf-8") as f:
        entries = json.load(f)

    features, labels = [], []
    for e in entries:
        try:
            path = os.path.join(DATA_DIR, e["audio"])
            path = clean_audio_path(path)
            audio, sr = sf.read(path)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)
            audio = augment_radio(audio)

            inp = processor(audio, sampling_rate=SR, return_tensors="np")
            lbl = processor.tokenizer(e["text"], return_tensors="np").input_ids[0]
            features.append(inp.input_features[0].astype(np.float16))
            labels.append(lbl)
        except Exception as ex:
            print(f"  [SKIP] {e['audio']} : {ex}")
    print(f"  {len(features)} echantillons militaires TTS charges")
    return features, labels


# ------------------------------------------------------------------ #
#  Collator et metriques
# ------------------------------------------------------------------ #

@dataclass
class DataCollatorSpeechSeq2Seq:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Reconvertir en float32 (HF Dataset stocke en liste Python, pas numpy)
        input_features = [{"input_features": np.array(f["input_features"], dtype=np.float32)} for f in features]
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
    from datasets import Dataset

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    print(f"Chargement {MODEL_ID}...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language="English", task="transcribe")
    model     = WhisperForConditionalGeneration.from_pretrained(MODEL_ID)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens    = []

    # Geler l'encodeur
    for param in model.model.encoder.parameters():
        param.requires_grad = False
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Encodeur gele. Params entrainables : {n_trainable:,} / {n_total:,} ({100*n_trainable/n_total:.1f}%)\n")

    # Chargement des donnees
    atco2_feat, atco2_lbl   = load_atco2_samples(MAX_ATCO2, processor)
    mil_feat,   mil_lbl     = load_military_samples(processor)

    # Fusion et shuffle
    all_feat   = atco2_feat + mil_feat
    all_labels = atco2_lbl  + mil_lbl
    combined   = list(zip(all_feat, all_labels))
    random.shuffle(combined)
    all_feat, all_labels = zip(*combined)
    all_feat   = list(all_feat)
    all_labels = list(all_labels)

    # Split train/val
    split    = int(len(all_feat) * (1 - VAL_RATIO))
    train_ds = Dataset.from_dict({"input_features": all_feat[:split],  "labels": all_labels[:split]})
    val_ds   = Dataset.from_dict({"input_features": all_feat[split:],  "labels": all_labels[split:]})
    print(f"\nDataset final : Train={len(train_ds)} | Val={len(val_ds)}")
    print(f"  dont ATCO2={len(atco2_feat)} | militaire TTS={len(mil_feat)}\n")

    collator = DataCollatorSpeechSeq2Seq(processor=processor)

    training_args = Seq2SeqTrainingArguments(
        output_dir=CKPT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        warmup_steps=50,
        num_train_epochs=12,
        evaluation_strategy="epoch",
        save_strategy="no",
        load_best_model_at_end=False,
        predict_with_generate=True,
        generation_max_length=225,
        logging_dir=LOG_DIR,
        logging_steps=20,
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

    print("Demarrage du fine-tuning (ATCO2 reel + militaire TTS, encodeur gele)...")
    trainer.train()

    # Sauvegarde
    best_dir = os.path.join(CKPT_DIR, "best")
    trainer.save_model(best_dir)
    processor.save_pretrained(best_dir)
    print(f"\nModele sauvegarde : {best_dir}")

    results = trainer.evaluate()
    print(f"  WER final : {results.get('eval_wer', 'N/A')}")

    results_path = os.path.join(LOG_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Resultats : {results_path}")


if __name__ == "__main__":
    main()
