"""
Test de débruitage + super-résolution avec Resemble Enhance.
Usage:
    python test_resemble.py --input audio_bruite.wav
    python test_resemble.py --input audio_bruite.wav --clean audio_reference.wav
"""

import argparse
import os
import torch
import torchaudio
from resemble_enhance.enhancer.inference import denoise, enhance


def compute_metrics(clean, enhanced, sr):
    """Calcule PESQ et STOI."""
    metrics = {}
    clean_np = clean.squeeze().cpu().numpy()
    enhanced_np = enhanced.squeeze().cpu().numpy()

    min_len = min(len(clean_np), len(enhanced_np))
    clean_np = clean_np[:min_len]
    enhanced_np = enhanced_np[:min_len]

    try:
        from pesq import pesq
        mode = "wb" if sr >= 16000 else "nb"
        metrics["PESQ"] = pesq(sr, clean_np, enhanced_np, mode)
    except Exception as e:
        print(f"  [!] PESQ error: {e}")

    try:
        from pystoi import stoi
        metrics["STOI"] = stoi(clean_np, enhanced_np, sr, extended=False)
    except Exception as e:
        print(f"  [!] STOI error: {e}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Test Resemble Enhance")
    parser.add_argument("--input", required=True, help="Fichier audio bruité")
    parser.add_argument("--clean", default=None, help="Fichier audio propre de référence")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    base, ext = os.path.splitext(args.input)

    # Charger audio
    noisy_wav, sr = torchaudio.load(args.input)
    if noisy_wav.shape[0] > 1:
        noisy_wav = noisy_wav.mean(dim=0, keepdim=True)
    noisy_wav = noisy_wav.squeeze()

    print(f"Audio d'entrée: {args.input}")
    print(f"  Sample rate: {sr} Hz | Durée: {len(noisy_wav)/sr:.2f}s")

    # === Test 1: Denoise seul ===
    print("\n=== DENOISE SEUL ===")
    print("Traitement en cours...")
    denoised, denoise_sr = denoise(noisy_wav, sr, device)
    out_denoise = f"{base}_resemble_denoised.wav"
    torchaudio.save(out_denoise, denoised.unsqueeze(0).cpu(), denoise_sr)
    print(f"Sauvegardé: {out_denoise}")

    # === Test 2: Enhance (denoise + super-résolution) ===
    print("\n=== DENOISE + SUPER-RÉSOLUTION ===")
    print("Traitement en cours (plus long, modèle de diffusion)...")
    enhanced, new_sr = enhance(noisy_wav, sr, device, nfe=64)
    out_enhance = f"{base}_resemble_enhanced.wav"
    torchaudio.save(out_enhance, enhanced.unsqueeze(0).cpu(), new_sr)
    print(f"Sauvegardé: {out_enhance} (sr={new_sr} Hz)")

    # === Métriques ===
    if args.clean:
        clean_wav, clean_sr = torchaudio.load(args.clean)
        if clean_wav.shape[0] > 1:
            clean_wav = clean_wav.mean(dim=0, keepdim=True)
        clean_wav = clean_wav.squeeze()

        # Pour comparer, resample tout au même sr que le clean
        target_sr = clean_sr

        # Resample noisy -> target_sr
        if sr != target_sr:
            noisy_cmp = torchaudio.transforms.Resample(sr, target_sr)(noisy_wav)
        else:
            noisy_cmp = noisy_wav

        # Resample denoised -> target_sr
        if denoise_sr != target_sr:
            denoised_cmp = torchaudio.transforms.Resample(denoise_sr, target_sr)(denoised.cpu())
        else:
            denoised_cmp = denoised.cpu()

        # Resample enhanced -> target_sr
        if new_sr != target_sr:
            enhanced_cmp = torchaudio.transforms.Resample(new_sr, target_sr)(enhanced.cpu())
        else:
            enhanced_cmp = enhanced.cpu()

        print(f"\nMétriques (comparées au clean à {target_sr} Hz)")

        print("\n--- Original (noisy vs clean) ---")
        m = compute_metrics(clean_wav, noisy_cmp, target_sr)
        for k, v in m.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- Denoise seul (vs clean) ---")
        m_den = compute_metrics(clean_wav, denoised_cmp, target_sr)
        for k, v in m_den.items():
            print(f"  {k}: {v:.4f}")

        print("\n--- Denoise + Super-résolution (vs clean) ---")
        m_enh = compute_metrics(clean_wav, enhanced_cmp, target_sr)
        for k, v in m_enh.items():
            print(f"  {k}: {v:.4f}")

        # Résumé
        print("\n--- Résumé des améliorations ---")
        print(f"  {'Métrique':<8} {'Original':>10} {'Denoise':>10} {'Enhance':>10}")
        for k in m:
            orig = m.get(k, 0)
            den = m_den.get(k, 0)
            enh = m_enh.get(k, 0)
            print(f"  {k:<8} {orig:>10.4f} {den:>10.4f} {enh:>10.4f}")
    else:
        print("\n[info] Fournissez --clean pour calculer les métriques PESQ/STOI")

    print("\nFichiers générés:")
    print(f"  Original:                {args.input}")
    print(f"  Denoise seul:            {out_denoise}")
    print(f"  Denoise + Super-résolution: {out_enhance}")


if __name__ == "__main__":
    main()
