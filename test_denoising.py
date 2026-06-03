"""
Test de débruitage avec SpeechBrain MetricGAN+.
Usage:
    python test_denoising.py --input votre_audio_bruite.wav
    python test_denoising.py --input votre_audio_bruite.wav --clean audio_reference_propre.wav
"""

import argparse
import os
import torch
import torchaudio
from speechbrain.inference.enhancement import SpectralMaskEnhancement


def compute_metrics(clean, enhanced, sr):
    """Calcule PESQ et STOI si les libs sont disponibles."""
    metrics = {}

    # Convertir en numpy
    clean_np = clean.squeeze().cpu().numpy()
    enhanced_np = enhanced.squeeze().cpu().numpy()

    # Tronquer à la même longueur
    min_len = min(len(clean_np), len(enhanced_np))
    clean_np = clean_np[:min_len]
    enhanced_np = enhanced_np[:min_len]

    try:
        from pesq import pesq
        metrics["PESQ"] = pesq(sr, clean_np, enhanced_np, "wb" if sr >= 16000 else "nb")
    except ImportError:
        print("  [!] pip install pesq  pour obtenir le score PESQ")
    except Exception as e:
        print(f"  [!] PESQ error: {e}")

    try:
        from pystoi import stoi
        metrics["STOI"] = stoi(clean_np, enhanced_np, sr, extended=False)
    except ImportError:
        print("  [!] pip install pystoi  pour obtenir le score STOI")
    except Exception as e:
        print(f"  [!] STOI error: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test débruitage SpeechBrain")
    parser.add_argument("--input", required=True, help="Fichier audio bruité (.wav)")
    parser.add_argument("--clean", default=None, help="Fichier audio propre de référence (optionnel, pour métriques)")
    parser.add_argument("--output", default=None, help="Fichier de sortie (défaut: input_enhanced.wav)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    args = parser.parse_args()

    if args.output is None:
        base, ext = os.path.splitext(args.input)
        args.output = f"{base}_enhanced{ext}"

    # Charger le modèle pré-entraîné MetricGAN+
    print("Chargement du modèle MetricGAN+ (speechbrain/metricgan-plus-voicebank)...")
    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": args.device},
    )

    # Charger l'audio bruité pour info
    noisy_wav, sr = torchaudio.load(args.input)
    print(f"Audio d'entrée: {args.input}")
    print(f"  Sample rate: {sr} Hz | Durée: {noisy_wav.shape[1]/sr:.2f}s | Channels: {noisy_wav.shape[0]}")

    # Resample à 16kHz si nécessaire (MetricGAN+ attend 16kHz)
    if sr != 16000:
        print(f"  Resampling {sr} -> 16000 Hz...")
        resampler = torchaudio.transforms.Resample(sr, 16000)
        noisy_wav = resampler(noisy_wav)
        sr = 16000

    # Mono si stéréo
    if noisy_wav.shape[0] > 1:
        print("  Conversion en mono...")
        noisy_wav = noisy_wav.mean(dim=0, keepdim=True)

    # Débruitage
    print("Débruitage en cours...")
    enhanced = enhancer.enhance_batch(
        noisy_wav.to(args.device),
        lengths=torch.tensor([1.0]).to(args.device),
    )

    # Sauvegarder
    torchaudio.save(args.output, enhanced.cpu(), sr)
    print(f"Audio débruité sauvegardé: {args.output}")

    # Métriques si audio de référence fourni
    if args.clean:
        print("\nCalcul des métriques (clean vs enhanced)...")
        clean_wav, clean_sr = torchaudio.load(args.clean)
        if clean_sr != sr:
            clean_wav = torchaudio.transforms.Resample(clean_sr, sr)(clean_wav)
        if clean_wav.shape[0] > 1:
            clean_wav = clean_wav.mean(dim=0, keepdim=True)

        # Métriques noisy vs clean
        print("\n--- Avant débruitage (noisy vs clean) ---")
        noisy_metrics = compute_metrics(clean_wav, noisy_wav, sr)
        for k, v in noisy_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Métriques enhanced vs clean
        print("\n--- Après débruitage (enhanced vs clean) ---")
        enhanced_metrics = compute_metrics(clean_wav, enhanced.cpu(), sr)
        for k, v in enhanced_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Amélioration
        print("\n--- Amélioration ---")
        for k in enhanced_metrics:
            if k in noisy_metrics:
                delta = enhanced_metrics[k] - noisy_metrics[k]
                print(f"  {k}: {'+' if delta >= 0 else ''}{delta:.4f}")
    else:
        print("\n[info] Fournissez --clean audio_reference.wav pour calculer PESQ/STOI")

    print("\nTerminé. Comparez les fichiers:")
    print(f"  Original:  {args.input}")
    print(f"  Débruité:  {args.output}")


if __name__ == "__main__":
    main()
