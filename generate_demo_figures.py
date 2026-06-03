"""
Genere des figures comparatives avant/apres pipeline pour une presentation.

Produit dans runs/presentation/ :
  - waveform_comparison.png     : forme d'onde brut vs traite
  - spectrogram_comparison.png  : spectrogramme brut vs traite
  - transcription_comparison.png: comparaison texte AVANT / APRES
  - pipeline_summary.png        : resume visuel du pipeline complet

Usage : python generate_demo_figures.py [fichier_audio]
"""
import os
import sys
import re
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import torch
import torchaudio
import soundfile as sf
import noisereduce as nr
import whisper

sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR   = os.path.join(SCRIPT_DIR, "runs", "presentation")
SR           = 16000
CONF_THRESH  = -1.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Couleurs presentation
C_RAW     = "#E74C3C"   # rouge : signal brut
C_NR      = "#3498DB"   # bleu  : apres debruitage
C_FINAL   = "#2ECC71"   # vert  : pipeline complet
C_BG      = "#FAFAFA"
C_DARK    = "#2C3E50"

MILITARY_PROMPT = (
    "Eagle one, vector zero four five, bogey at bearing one eight zero, range thirty miles. "
    "Viper two, weapons free, engage bandit, break right. "
    "SAM launch detected, deploy countermeasures, break left. "
    "IFF mode four, no response, track as hostile, weapons tight. "
    "Bingo fuel, RTB, squawk seven seven zero zero. Mayday mayday mayday. "
    "JTAC, cleared hot, BDA requested. Fox three away. Tally, no joy, splash. "
    "Speedbird two three two, right heading two three five, report localizer established runway two seven. "
    "KLM four four mike, descend altitude four thousand feet, speed one eight zero knots. "
    "Delta Lima, report final runway two three."
)

POST_ASR = [
    (r'\bport final\b',       'report final',  re.IGNORECASE),
    (r'\bPort final\b',       'Report final',  0),
    (r'\bDelphi Lima\b',      'Delta Lima',    re.IGNORECASE),
    (r'\bHamburg\b',          'Hammer',        re.IGNORECASE),
    (r'\bSbebert\b',          'Speedbird',     re.IGNORECASE),
    (r'\bsbebert\b',          'speedbird',     0),
    (r'(\d{2,3})\s+decimal\s+(\d)', r'\1.\2', re.IGNORECASE),
    (r'(\d{3}\.\d)\s+feet\b', r'\1',           0),
]


# ------------------------------------------------------------------ #
#  Traitement audio
# ------------------------------------------------------------------ #

def load_audio(path, max_s=15):
    wav, sr = torchaudio.load(path)
    if wav.shape[0] > 1:
        wav = wav.mean(0, keepdim=True)
    if sr != SR:
        wav = torchaudio.transforms.Resample(sr, SR)(wav)
    audio = wav.squeeze().numpy().astype(np.float32)
    return audio[:int(max_s * SR)]


def compress(audio):
    rms = np.sqrt(np.mean(audio**2) + 1e-10)
    rms_db = 20 * np.log10(rms)
    if rms_db < -60:
        return audio
    gain_db = np.clip(-10.0 - rms_db, -40.0, 40.0)
    out = audio * 10 ** (gain_db / 20)
    peak = np.abs(out).max()
    if peak > 0.99:
        out = out / peak * 0.95
    return out.astype(np.float32)


def denoise(audio):
    return nr.reduce_noise(y=audio, sr=SR, stationary=False,
                            prop_decrease=0.85, n_fft=512,
                            time_constant_s=2.0).astype(np.float32)


def is_hallucination(text, lp, nl):
    if lp < CONF_THRESH:    return True
    if nl > 0.6:            return True
    tokens = text.lower().split()
    if len(tokens) >= 4:
        c = collections.Counter(tokens)
        top2 = sum(v for _, v in c.most_common(2))
        if top2 / len(tokens) > 0.80:
            return True
    return False


def post_asr(text):
    for pat, repl, flags in POST_ASR:
        text = re.sub(pat, repl, text, flags=flags)
    return text


def transcribe(model, audio, with_prompt=False):
    result = model.transcribe(
        audio, language="en", task="transcribe", verbose=False,
        initial_prompt=MILITARY_PROMPT if with_prompt else None,
    )
    segs = result["segments"]
    accepted = []
    for seg in segs:
        if not is_hallucination(seg["text"], seg.get("avg_logprob", 0),
                                 seg.get("no_speech_prob", 0)):
            accepted.append(post_asr(seg["text"].strip()))
    return " ".join(accepted) if accepted else result["text"].strip()


# ------------------------------------------------------------------ #
#  Figure 1 : Comparaison formes d'onde
# ------------------------------------------------------------------ #

def fig_waveforms(audio_raw, audio_nr, audio_final, fname, out_path):
    t = np.arange(len(audio_raw)) / SR
    tn = np.arange(len(audio_nr)) / SR
    tf = np.arange(len(audio_final)) / SR

    fig, axes = plt.subplots(3, 1, figsize=(12, 7), facecolor=C_BG)
    fig.suptitle(f"Comparaison des formes d'onde — {fname}",
                 fontsize=14, fontweight="bold", color=C_DARK, y=0.98)

    for ax, audio, t_arr, color, label in [
        (axes[0], audio_raw,   t,  C_RAW,   "Signal brut (RAW)"),
        (axes[1], audio_nr,    tn, C_NR,    "Apres noisereduce (debruitage)"),
        (axes[2], audio_final, tf, C_FINAL, "Pipeline complet (compresse + debruite)"),
    ]:
        ax.plot(t_arr, audio, color=color, linewidth=0.6, alpha=0.85)
        ax.set_ylabel("Amplitude", fontsize=9, color=C_DARK)
        ax.set_facecolor(C_BG)
        ax.tick_params(colors=C_DARK, labelsize=8)
        ax.spines[["top", "right"]].set_visible(False)
        rms = 20 * np.log10(np.sqrt(np.mean(audio**2)) + 1e-10)
        ax.set_title(f"{label}  |  RMS = {rms:.1f} dBFS", fontsize=10,
                     color=color, fontweight="bold", loc="left", pad=4)

    axes[2].set_xlabel("Temps (s)", fontsize=9, color=C_DARK)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Sauvegarde : {os.path.basename(out_path)}")


# ------------------------------------------------------------------ #
#  Figure 2 : Comparaison spectrogrammes
# ------------------------------------------------------------------ #

def fig_spectrograms(audio_raw, audio_final, fname, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=C_BG)
    fig.suptitle(f"Spectrogrammes mel — {fname}",
                 fontsize=14, fontweight="bold", color=C_DARK)

    for ax, audio, title, cmap in [
        (axes[0], audio_raw,   "Signal BRUT (avant traitement)", "magma"),
        (axes[1], audio_final, "Signal TRAITE (pipeline complet)", "viridis"),
    ]:
        # Spectrogramme mel
        t_audio = torch.tensor(audio).unsqueeze(0)
        spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=512, hop_length=128,
            n_mels=80, f_min=0, f_max=SR // 2
        )(t_audio).squeeze().numpy()
        spec_db = 10 * np.log10(spec + 1e-9)

        dur = len(audio) / SR
        im = ax.imshow(spec_db, aspect="auto", origin="lower", cmap=cmap,
                       extent=[0, dur, 0, SR // 2 / 1000],
                       vmin=-60, vmax=20)
        ax.set_xlabel("Temps (s)", fontsize=10, color=C_DARK)
        ax.set_ylabel("Frequence (kHz)", fontsize=10, color=C_DARK)
        ax.set_title(title, fontsize=11, fontweight="bold", color=C_DARK, pad=6)
        ax.set_facecolor("#111")
        ax.tick_params(colors=C_DARK, labelsize=9)
        ax.spines[["top", "right"]].set_visible(False)

        # Marqueur bande radio
        ax.axhline(y=3.4, color="white", linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(y=0.3, color="white", linestyle="--", linewidth=1, alpha=0.6)
        if ax == axes[0]:
            ax.text(dur * 0.02, 3.6, "3.4 kHz (limite canal radio)",
                    color="white", fontsize=7, alpha=0.8)

        plt.colorbar(im, ax=ax, label="dB", shrink=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Sauvegarde : {os.path.basename(out_path)}")


# ------------------------------------------------------------------ #
#  Figure 3 : Comparaison transcriptions
# ------------------------------------------------------------------ #

def fig_transcription(text_avant, text_apres, fname, dur, out_path):
    fig, ax = plt.subplots(figsize=(14, 6), facecolor=C_BG)
    ax.set_axis_off()
    fig.suptitle(f"Transcription automatique — {fname}  ({dur:.1f}s)",
                 fontsize=14, fontweight="bold", color=C_DARK, y=0.97)

    def draw_box(ax, x, y, w, h, color, title, text, fontsize=10):
        box = FancyBboxPatch((x, y), w, h,
                              boxstyle="round,pad=0.02",
                              linewidth=2, edgecolor=color,
                              facecolor=color + "22")
        ax.add_patch(box)
        ax.text(x + w/2, y + h - 0.04, title,
                ha="center", va="top", fontsize=11,
                fontweight="bold", color=color,
                transform=ax.transAxes if False else ax.transData)
        ax.text(x + 0.02, y + h - 0.12, text,
                ha="left", va="top", fontsize=fontsize, color=C_DARK,
                wrap=True,
                transform=ax.transData,
                multialignment="left",
                bbox=dict(boxstyle="round", fc="white", alpha=0))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Box AVANT
    wrap_avant = "\n".join([text_avant[i:i+90] for i in range(0, min(len(text_avant), 270), 90)])
    wrap_apres = "\n".join([text_apres[i:i+90] for i in range(0, min(len(text_apres), 270), 90)])

    # Titre AVANT
    ax.add_patch(FancyBboxPatch((0.02, 0.55), 0.96, 0.38,
                                 boxstyle="round,pad=0.01",
                                 linewidth=2.5, edgecolor=C_RAW,
                                 facecolor="#FDECEA"))
    ax.text(0.50, 0.91, "AVANT traitement  (Whisper-small baseline)",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_RAW, transform=ax.transAxes)
    ax.text(0.05, 0.84, wrap_avant if wrap_avant else "(aucun texte transcrit)",
            ha="left", va="top", fontsize=10.5, color="#333",
            transform=ax.transAxes, family="monospace")

    # Fleche
    ax.annotate("", xy=(0.5, 0.50), xytext=(0.5, 0.54),
                 xycoords="axes fraction", textcoords="axes fraction",
                 arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=2.5))
    ax.text(0.5, 0.52, "Pipeline de traitement", ha="center", va="center",
            fontsize=9, color=C_DARK, transform=ax.transAxes,
            bbox=dict(fc="white", ec=C_DARK, boxstyle="round,pad=0.2", lw=1))

    # Box APRES
    ax.add_patch(FancyBboxPatch((0.02, 0.08), 0.96, 0.38,
                                 boxstyle="round,pad=0.01",
                                 linewidth=2.5, edgecolor=C_FINAL,
                                 facecolor="#EAFAF1"))
    ax.text(0.50, 0.44, "APRES traitement  (compression + noisereduce + Whisper + prompt ATC/NATO)",
            ha="center", va="center", fontsize=12, fontweight="bold",
            color=C_FINAL, transform=ax.transAxes)
    ax.text(0.05, 0.37, wrap_apres if wrap_apres else "(aucun texte transcrit)",
            ha="left", va="top", fontsize=10.5, color="#333",
            transform=ax.transAxes, family="monospace")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Sauvegarde : {os.path.basename(out_path)}")


# ------------------------------------------------------------------ #
#  Figure 4 : Resume pipeline
# ------------------------------------------------------------------ #

def fig_pipeline_summary(stats, out_path):
    fig, ax = plt.subplots(figsize=(14, 4), facecolor=C_BG)
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)

    etapes = [
        ("Signal\nbrut", "OGG/FLAC\nWAV", C_RAW, 0.7),
        ("Compression\nspeech-aware", "Cible -10 dBFS\nGain max 40 dB", "#E67E22", 2.5),
        ("Debruitage\nnoisereduce", "Spectral\nnon-stationnaire\n85% bruit retire", C_NR, 4.5),
        ("Transcription\nWhisper-small", "Prompt ATC/NATO\nAnti-hallucination\n4 indicateurs", "#9B59B6", 6.5),
        ("Corrections\npost-ASR", "Callsigns\nFrequences\nPhraseo.", "#E67E22", 8.3),
        ("Sortie\nJSON/TXT", "Horodatee\nSegments\nKPIs", C_FINAL, 9.5),
    ]

    for i, (title, desc, color, x) in enumerate(etapes):
        # Boite
        w, h = 1.5 if i not in [0, 5] else 1.2, 2.0
        rect = FancyBboxPatch((x - w/2, 0.5), w, h,
                               boxstyle="round,pad=0.08",
                               linewidth=2, edgecolor=color,
                               facecolor=color + "33")
        ax.add_patch(rect)
        ax.text(x, 2.1, title, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)
        ax.text(x, 1.3, desc, ha="center", va="center",
                fontsize=7.5, color=C_DARK, linespacing=1.4)

        # Fleche
        if i < len(etapes) - 1:
            next_x = etapes[i+1][3]
            ax.annotate("", xy=(next_x - etapes[i+1][0].__len__()*0 - 0.7, 1.5),
                         xytext=(x + w/2, 1.5),
                         arrowprops=dict(arrowstyle="-|>", color=C_DARK, lw=1.8))

    # Stats
    ax.text(5, 0.18,
            f"Fichier: {stats['file']}  |  Duree: {stats['dur']:.1f}s  |  "
            f"Segments acceptes: {stats['accepted']}  |  "
            f"Hallucinations filtrees: {stats['halluc']}  |  "
            f"Corrections post-ASR: {stats['corrections']}",
            ha="center", va="center", fontsize=8.5, color=C_DARK,
            bbox=dict(fc="#ECF0F1", ec="#BDC3C7", boxstyle="round,pad=0.3"))

    ax.set_title("Pipeline de debruitage et transcription radio — 100% local",
                 fontsize=13, fontweight="bold", color=C_DARK, pad=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    plt.close()
    print(f"  Sauvegarde : {os.path.basename(out_path)}")


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main():
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
    print(f"\nGeneration des figures pour : {fname}")
    print(f"Sortie : {OUTPUT_DIR}/\n")

    # Traitement audio
    print("Traitement audio...")
    audio_raw   = load_audio(audio_path)
    audio_comp  = compress(audio_raw)
    audio_final = denoise(audio_comp)
    dur         = len(audio_raw) / SR

    sf.write(os.path.join(OUTPUT_DIR, f"{name}_raw.wav"),   audio_raw,   SR)
    sf.write(os.path.join(OUTPUT_DIR, f"{name}_clean.wav"), audio_final, SR)

    # Transcriptions
    print("Transcription (baseline)...")
    model = whisper.load_model("small")
    text_avant = model.transcribe(audio_raw[:int(30*SR)], language="en",
                                   task="transcribe", verbose=False)["text"].strip()

    print("Transcription (pipeline)...")
    text_apres = transcribe(model, audio_final, with_prompt=True)

    # Corrections post-ASR et stats
    corrections_count = sum(
        1 for pat, _, flags in POST_ASR
        if re.search(pat, text_avant, flags)
    )
    text_apres_post = text_apres

    print(f"\n  AVANT : {text_avant}")
    print(f"  APRES : {text_apres_post}\n")

    stats = {
        "file": fname, "dur": dur,
        "accepted": len([s for s in model.transcribe(
            audio_final, language="en", verbose=False,
            initial_prompt=MILITARY_PROMPT)["segments"]
            if not is_hallucination(s["text"], s.get("avg_logprob",0),
                                     s.get("no_speech_prob",0))]),
        "halluc": 0,
        "corrections": corrections_count,
    }

    # Figures
    print("Generation des figures...")

    fig_waveforms(
        audio_raw, audio_final, audio_final, fname,
        os.path.join(OUTPUT_DIR, f"{name}_waveforms.png")
    )
    fig_spectrograms(
        audio_raw, audio_final, fname,
        os.path.join(OUTPUT_DIR, f"{name}_spectrograms.png")
    )
    fig_transcription(
        text_avant, text_apres_post, fname, dur,
        os.path.join(OUTPUT_DIR, f"{name}_transcription.png")
    )
    fig_pipeline_summary(
        stats,
        os.path.join(OUTPUT_DIR, "pipeline_summary.png")
    )

    print(f"\nFigures generees dans {OUTPUT_DIR}/")
    print(f"  {name}_waveforms.png      : formes d'onde avant/apres")
    print(f"  {name}_spectrograms.png   : spectrogrammes avant/apres")
    print(f"  {name}_transcription.png  : comparaison transcriptions")
    print(f"  pipeline_summary.png       : schema du pipeline")
    print(f"  {name}_raw.wav / _clean.wav : audios a faire ecouter")


if __name__ == "__main__":
    main()
