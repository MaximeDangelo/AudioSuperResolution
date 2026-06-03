"""
Test de robustesse de la transcription sur vocabulaire militaire aeronautique.

1. Genere des audios TTS pour 20 phrases militaires cles (5 categories)
2. Applique le pipeline complet (noisereduce + Whisper ATCO2 v4)
3. Compare avec Whisper baseline
4. Calcule le WER par categorie
5. Identifie les termes mal reconnus
"""
import os, sys, asyncio, io, json, re
import numpy as np
import soundfile as sf
import torch
import torchaudio
import edge_tts
import noisereduce as nr
import whisper
from scipy.signal import butter, sosfilt
from jiwer import wer as compute_wer
from transformers import WhisperProcessor, WhisperForConditionalGeneration

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
ATCO2_MODEL = os.path.join(SCRIPT_DIR, "checkpoints", "whisper_atco2", "best")
OUTPUT_DIR  = os.path.join(SCRIPT_DIR, "runs", "test_military")
SR          = 16000

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------------ #
#  Phrases de test par categorie militaire
# ------------------------------------------------------------------ #

TEST_PHRASES = [
    # --- Interception et controle ---
    ("intercept_01", "Eagle one, vector zero four five, bogey at bearing one eight zero, range thirty miles, altitude angels fifteen."),
    ("intercept_02", "Viper two, weapons free, engage bandit, break right, break right."),
    ("intercept_03", "Ghost flight, tally two bandits, ten o clock low, fox three away."),
    ("intercept_04", "Alpha flight, splash one, remaining bogey is at bearing two seven zero, angels ten."),
    ("intercept_05", "Hammer two one, no joy, request vector to merge."),

    # --- Gestion des menaces ---
    ("threat_01", "SAM launch detected, bearing zero nine zero, deploy countermeasures, break left."),
    ("threat_02", "IFF mode four, no response, track as hostile, weapons tight."),
    ("threat_03", "Bingo fuel, returning to base, vector two three five, squawk seven seven zero zero."),
    ("threat_04", "Mayday mayday mayday, Eagle three is hit, ejecting, position grid four three november kilo."),
    ("threat_05", "Radar lock, break break, chaff and flares, defensive."),

    # --- Appui feu et CAS ---
    ("cas_01", "JTAC, Hammer two one, request close air support, target grid four three november kilo two four."),
    ("cas_02", "Cleared hot, single pass, friendlies are five hundred meters north of target."),
    ("cas_03", "BDA: two vehicles destroyed, fire suppressed, pull off south."),
    ("cas_04", "Abort abort abort, friendlies too close, safe escape heading one eight zero."),
    ("cas_05", "Splash, good hits on target, request BDA confirmation."),

    # --- Navigation et RTB ---
    ("nav_01", "Eagle one, bingo state, RTB via direct route, angels twenty, speed three five zero knots."),
    ("nav_02", "Viper flight, switch to guard frequency two four three decimal zero, over."),
    ("nav_03", "Link sixteen update, picture shows two groups, north and south, commit north group."),
    ("nav_04", "Fence in, all systems go, combat spread, combat spread."),
    ("nav_05", "Joker fuel, fence out, RTB, squawk normal."),
]

CATEGORIES = {
    "intercept": ["intercept_01","intercept_02","intercept_03","intercept_04","intercept_05"],
    "threat":    ["threat_01","threat_02","threat_03","threat_04","threat_05"],
    "cas":       ["cas_01","cas_02","cas_03","cas_04","cas_05"],
    "nav":       ["nav_01","nav_02","nav_03","nav_04","nav_05"],
}

VOICE = "en-US-GuyNeural"

# ------------------------------------------------------------------ #
#  Generation TTS + simulation radio
# ------------------------------------------------------------------ #

async def tts(text, path_mp3):
    comm = edge_tts.Communicate(text, VOICE)
    await comm.save(path_mp3)

def bandpass(audio, sr=SR, low=300, high=3400):
    sos = butter(5, [low, high], btype="band", fs=sr, output="sos")
    return sosfilt(sos, audio).astype(np.float32)

def add_noise(audio, snr_db=20):
    sig_rms  = np.sqrt(np.mean(audio**2) + 1e-10)
    noise    = bandpass(np.random.randn(len(audio)).astype(np.float32))
    n_rms    = np.sqrt(np.mean(noise**2) + 1e-10)
    noise   *= sig_rms / n_rms / (10**(snr_db/20))
    out      = audio + noise
    peak     = np.abs(out).max()
    return (out / peak * 0.85).astype(np.float32) if peak > 0 else out

async def generate_audio():
    mp3_tmp = os.path.join(OUTPUT_DIR, "_tmp.mp3")
    results = {}
    print(f"Generation TTS pour {len(TEST_PHRASES)} phrases...")
    for name, text in TEST_PHRASES:
        wav_clean = os.path.join(OUTPUT_DIR, f"{name}_clean.wav")
        wav_radio = os.path.join(OUTPUT_DIR, f"{name}_radio.wav")
        if os.path.exists(wav_radio):
            wav, _ = sf.read(wav_radio)
            results[name] = (text, wav.astype(np.float32))
            continue
        await tts(text, mp3_tmp)
        wav_t, sr = torchaudio.load(mp3_tmp)
        if wav_t.shape[0] > 1: wav_t = wav_t.mean(0, keepdim=True)
        if sr != SR: wav_t = torchaudio.transforms.Resample(sr, SR)(wav_t)
        audio = wav_t.squeeze().numpy().astype(np.float32)
        peak  = np.abs(audio).max()
        if peak > 0: audio = audio / peak * 0.9
        sf.write(wav_clean, audio, SR)
        # Version radio degradee
        radio = add_noise(bandpass(audio), snr_db=22)
        sf.write(wav_radio, radio, SR)
        results[name] = (text, radio)
        print(f"  {name} OK")
    if os.path.exists(mp3_tmp): os.remove(mp3_tmp)
    return results

# ------------------------------------------------------------------ #
#  Normalisation texte pour WER
# ------------------------------------------------------------------ #

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------------------------------------------------------ #
#  Transcription
# ------------------------------------------------------------------ #

MILITARY_PROMPT = (
    "Eagle one, weapons free, engage bandit bearing zero four five, range twenty miles. "
    "Viper two, vector two seven zero, altitude angels fifteen, speed three five zero knots. "
    "SAM launch detected, deploy countermeasures, break right. "
    "Squawk seven seven zero zero. Mayday mayday mayday. "
    "JTAC, cleared hot, BDA requested. Fox three away. Tally, no joy, splash. "
    "Bingo fuel, RTB, fence in, fence out, joker state. "
    "IFF mode four, no response. Link sixteen, guard frequency two four three decimal zero."
)

def transcribe_baseline(model_base, audio_np):
    result = model_base.transcribe(audio_np, language="en", task="transcribe", verbose=False)
    return result["text"].strip()

def transcribe_baseline_prompt(model_base, audio_np):
    result = model_base.transcribe(audio_np, language="en", task="transcribe",
                                    verbose=False, initial_prompt=MILITARY_PROMPT)
    return result["text"].strip()

def transcribe_atco2(ft_proc, ft_model, audio_np, device):
    inp = ft_proc(audio_np, sampling_rate=SR, return_tensors="pt")
    try:
        forced = ft_proc.get_decoder_prompt_ids(language="english", task="transcribe")
    except Exception:
        forced = None
    with torch.no_grad():
        kw = {"max_new_tokens": 200}
        if forced: kw["forced_decoder_ids"] = forced
        ids = ft_model.generate(inp.input_features.to(device), **kw)
    return ft_proc.batch_decode(ids, skip_special_tokens=True)[0].strip()

def transcribe_pipeline(ft_proc, ft_model, audio_np, device):
    """ATCO2 + noisereduce."""
    denoised = nr.reduce_noise(y=audio_np, sr=SR, stationary=False,
                                prop_decrease=0.85, n_fft=512, time_constant_s=2.0)
    return transcribe_atco2(ft_proc, ft_model, denoised.astype(np.float32), device)

# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}\n")

    # Generation des audios
    audio_data = asyncio.run(generate_audio())
    print()

    # Chargement modeles
    print("Chargement Whisper baseline...")
    model_base = whisper.load_model("small")
    print("Chargement ATCO2 v4...")
    ft_proc  = WhisperProcessor.from_pretrained(ATCO2_MODEL)
    ft_model = WhisperForConditionalGeneration.from_pretrained(ATCO2_MODEL).to(device)
    ft_model.eval()
    print()

    # Transcriptions
    all_results = []
    wers = {"baseline": [], "prompt": [], "atco2": [], "pipeline": []}

    for name, text in TEST_PHRASES:
        if name not in audio_data:
            continue
        ref, audio_np = audio_data[name]

        t_base    = transcribe_baseline(model_base, audio_np)
        t_prompt  = transcribe_baseline_prompt(model_base, audio_np)
        t_atco2   = transcribe_atco2(ft_proc, ft_model, audio_np, device)
        t_pipe    = transcribe_pipeline(ft_proc, ft_model, audio_np, device)

        ref_n  = normalize(ref)
        for key, pred in [("baseline", t_base), ("prompt", t_prompt),
                           ("atco2", t_atco2), ("pipeline", t_pipe)]:
            try:
                w = compute_wer(ref_n, normalize(pred))
            except Exception:
                w = 1.0
            wers[key].append(w)

        all_results.append({
            "name": name, "reference": ref,
            "baseline": t_base, "prompt": t_prompt,
            "atco2": t_atco2, "pipeline": t_pipe,
            "wer_baseline": round(wers["baseline"][-1], 3),
            "wer_prompt":   round(wers["prompt"][-1], 3),
            "wer_atco2":    round(wers["atco2"][-1], 3),
            "wer_pipeline": round(wers["pipeline"][-1], 3),
        })

    # Affichage resultats
    print("=" * 75)
    print(f"{'Phrase':<14} {'WER base':>9} {'WER+prompt':>10} {'WER ATCO2':>10} {'WER pipe':>9}")
    print("-" * 75)
    for r in all_results:
        print(f"{r['name']:<14} {r['wer_baseline']:>9.3f} {r['wer_prompt']:>10.3f} "
              f"{r['wer_atco2']:>10.3f} {r['wer_pipeline']:>9.3f}")

    print("-" * 75)
    for key in wers:
        avg = np.mean(wers[key])
        print(f"  Moyenne WER {key:<12}: {avg:.3f} ({avg*100:.1f}%)")

    # WER par categorie
    print("\nWER par categorie (pipeline final) :")
    for cat, names in CATEGORIES.items():
        cat_wers = [r["wer_pipeline"] for r in all_results if r["name"] in names]
        if cat_wers:
            print(f"  {cat:<12} : {np.mean(cat_wers):.3f} ({np.mean(cat_wers)*100:.1f}%)")

    # Exemples de transcriptions
    print("\nExemples de transcriptions (pipeline final) :")
    for r in all_results[:5]:
        print(f"\n  [{r['name']}]")
        print(f"  REF  : {r['reference'][:100]}")
        print(f"  PRED : {r['pipeline'][:100]}")

    # Sauvegarde
    report_path = os.path.join(OUTPUT_DIR, "military_wer_results.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"results": all_results, "wer_means": {k: float(np.mean(v)) for k, v in wers.items()}}, f, ensure_ascii=False, indent=2)

    print(f"\nResultats : {report_path}")

if __name__ == "__main__":
    main()
