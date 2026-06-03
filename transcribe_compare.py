"""
Transcription avant/apres debruitage SGMSE+ fine-tune.
Pour chaque fichier radio : transcription brute + transcription post-SGMSE+.
Genere un rapport de comparaison dans output/transcriptions_compare/
"""
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import soundfile as sf
import whisper

sys.stdout.reconfigure(encoding="utf-8")

# === Configuration ===
SCRIPT_DIR     = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR      = os.path.join(SCRIPT_DIR, "Dataset Radio (2)")
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "sgmse-voicebank")
CHECKPOINT     = os.path.join(SCRIPT_DIR, "checkpoints", "sgmse", "best_model.pt")
OUTPUT_DIR     = os.path.join(SCRIPT_DIR, "output", "transcriptions_compare")
WHISPER_MODEL  = "small"
SR_MODEL       = 16000
N_FFT          = 510
HOP_LENGTH     = 128
MAX_DURATION_S = 60  # limiter les fichiers longs

FILES = [
    "01__kwahmah_atc001.flac",
    "02__kwahmah_atc002.flac",
    "03_kwahmah_atc003.flac",
    "05__kwahmah_atc005.flac",
    "06_kwahmah_atc006.flac",
    "07-wahmah_heathrow-air-traffic-control.flac",
    "08__kwahmah_hong-kong-air-traffic-control.flac",
]

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------------ #
#  Utilitaires SGMSE+
# ------------------------------------------------------------------ #

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_window(device):
    return torch.hann_window(N_FFT).to(device)


def stft(x, window):
    return torch.stft(x, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=window, center=True, return_complex=True)


def istft(spec, window, length=None):
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       window=window, center=True, length=length)


def spec_fwd(S, factor=0.15, exponent=0.5):
    mag = S.abs() ** exponent
    ph  = S.angle()
    return mag * torch.exp(1j * ph) * factor


def spec_back(S, factor=0.15, exponent=0.5):
    S   = S / factor
    mag = S.abs() ** (1.0 / exponent)
    ph  = S.angle()
    return mag * torch.exp(1j * ph)


def pad_spec(spec):
    T = spec.shape[-1]
    num_pad = (64 - T % 64) % 64
    if num_pad > 0:
        real = F.pad(spec.real, (0, num_pad))
        imag = F.pad(spec.imag, (0, num_pad))
        spec = torch.complex(real, imag)
    return spec


def load_sgmse(device):
    from speechbrain.inference.enhancement import SGMSEEnhancement
    print("Chargement SGMSE+ pre-entraine...")
    pretrained = SGMSEEnhancement.from_hparams(
        source="speechbrain/sgmse-voicebank",
        savedir=PRETRAINED_DIR,
    )
    score_model = pretrained.mods.score_model

    print(f"Chargement checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location=device)
    print(f"  Epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    if "ema_state_dict" in ckpt and "shadow_params" in ckpt["ema_state_dict"]:
        shadow = ckpt["ema_state_dict"]["shadow_params"]
        for p, s in zip(score_model.dnn.parameters(), shadow):
            p.data.copy_(s.to(device))
        print("  Poids EMA charges")
    else:
        score_model.load_state_dict(ckpt["model_state_dict"])
        print("  Poids normaux charges")

    score_model = score_model.to(device).eval()
    return score_model


def enhance(score_model, wav, device, window):
    with torch.no_grad():
        x    = wav.unsqueeze(0).to(device)
        norm = torch.clamp(x.abs().amax(dim=-1, keepdim=True), min=1e-8)
        y    = x / norm

        Y     = spec_fwd(stft(y, window)).unsqueeze(1)
        T_orig = Y.shape[-1]
        Yp    = pad_spec(Y)

        x_hat = score_model.enhance(Yp)
        Xh    = x_hat[:, :, :, :T_orig].squeeze(1)
        Xh    = spec_back(Xh)
        out   = istft(Xh, window, length=y.shape[-1]) * norm
        return out.squeeze(0).cpu()


def load_mono_16k(path):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR_MODEL:
        wav = torchaudio.transforms.Resample(sr, SR_MODEL)(wav)
    return wav.squeeze(0)


# ------------------------------------------------------------------ #
#  Prompt militaire aeronautique (biais vocabulaire Whisper)
# ------------------------------------------------------------------ #

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


# ------------------------------------------------------------------ #
#  Transcription Whisper
# ------------------------------------------------------------------ #

def transcribe(model, audio_np, use_prompt=True):
    result = model.transcribe(
        audio_np.astype(np.float32),
        language="en",
        task="transcribe",
        verbose=False,
        initial_prompt=MILITARY_PROMPT if use_prompt else None,
    )
    return result["text"].strip(), result["segments"]


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    device = get_device()
    print(f"Device: {device}\n")

    score_model = load_sgmse(device)
    window      = get_window(device)
    max_samples = MAX_DURATION_S * SR_MODEL

    print(f"\nChargement Whisper-{WHISPER_MODEL}...")
    whisper_model = whisper.load_model(WHISPER_MODEL)
    print("Whisper charge.\n")

    results = []

    for fname in FILES:
        fpath = os.path.join(AUDIO_DIR, fname)
        if not os.path.exists(fpath):
            print(f"[SKIP] {fname} introuvable")
            continue

        name = os.path.splitext(fname)[0]
        print("=" * 70)
        print(f"Fichier : {fname}")

        # --- Chargement ---
        wav_raw = load_mono_16k(fpath)
        if len(wav_raw) > max_samples:
            wav_raw = wav_raw[:max_samples]
            print(f"  (tronque a {MAX_DURATION_S}s)")

        # --- Transcription brute (sans prompt) ---
        print("  Transcription RAW...", end="", flush=True)
        raw_np = wav_raw.numpy()
        text_raw, segs_raw = transcribe(whisper_model, raw_np, use_prompt=False)
        print(" OK")

        # --- Transcription brute avec prompt militaire ---
        print("  Transcription RAW + prompt...", end="", flush=True)
        text_raw_prompt, segs_raw_prompt = transcribe(whisper_model, raw_np, use_prompt=True)
        print(" OK")

        # --- Debruitage SGMSE+ ---
        print("  Debruitage SGMSE+...", end="", flush=True)
        wav_enh = enhance(score_model, wav_raw, device, window)
        enh_np  = wav_enh.numpy()
        print(" OK")

        # --- Transcription debruitee + prompt ---
        print("  Transcription SGMSE+ + prompt...", end="", flush=True)
        text_enh_prompt, segs_enh_prompt = transcribe(whisper_model, enh_np, use_prompt=True)
        print(" OK")

        print(f"\n  [RAW]              {text_raw}")
        print(f"  [RAW + prompt]     {text_raw_prompt}")
        print(f"  [SGMSE+ + prompt]  {text_enh_prompt}\n")

        # --- Sauvegardes audio ---
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_raw.wav"),   raw_np, SR_MODEL)
        sf.write(os.path.join(OUTPUT_DIR, f"{name}_sgmse.wav"), enh_np, SR_MODEL)

        # --- Sauvegardes texte ---
        with open(os.path.join(OUTPUT_DIR, f"{name}_raw.txt"), "w", encoding="utf-8") as f:
            f.write(text_raw)

        with open(os.path.join(OUTPUT_DIR, f"{name}_raw_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text_raw_prompt)

        with open(os.path.join(OUTPUT_DIR, f"{name}_sgmse_prompt.txt"), "w", encoding="utf-8") as f:
            f.write(text_enh_prompt)

        # --- Segments detailles ---
        with open(os.path.join(OUTPUT_DIR, f"{name}_raw_segments.txt"), "w", encoding="utf-8") as f:
            for seg in segs_raw:
                f.write(f"[{seg['start']:6.2f}s -> {seg['end']:6.2f}s]  {seg['text'].strip()}\n")

        with open(os.path.join(OUTPUT_DIR, f"{name}_raw_prompt_segments.txt"), "w", encoding="utf-8") as f:
            for seg in segs_raw_prompt:
                f.write(f"[{seg['start']:6.2f}s -> {seg['end']:6.2f}s]  {seg['text'].strip()}\n")

        with open(os.path.join(OUTPUT_DIR, f"{name}_sgmse_prompt_segments.txt"), "w", encoding="utf-8") as f:
            for seg in segs_enh_prompt:
                f.write(f"[{seg['start']:6.2f}s -> {seg['end']:6.2f}s]  {seg['text'].strip()}\n")

        results.append((fname, text_raw, text_raw_prompt, text_enh_prompt))

    # --- Rapport global ---
    report_path = os.path.join(OUTPUT_DIR, "rapport_comparaison.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("RAPPORT COMPARAISON TRANSCRIPTION\n")
        f.write("RAW  |  RAW + prompt militaire  |  SGMSE+ + prompt militaire\n")
        f.write("=" * 70 + "\n\n")
        for fname, t_raw, t_raw_p, t_enh_p in results:
            f.write(f"=== {fname} ===\n")
            f.write(f"[RAW]             {t_raw}\n")
            f.write(f"[RAW + prompt]    {t_raw_p}\n")
            f.write(f"[SGMSE+ + prompt] {t_enh_p}\n\n")

    print("=" * 70)
    print(f"Audios et transcriptions sauvegardes dans : {OUTPUT_DIR}")
    print(f"Rapport : {report_path}")


if __name__ == "__main__":
    main()
