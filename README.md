# Pipeline de Débruitage et Transcription des Communications Radio Aéronautiques

**Projet ACE4ACES — Thales Belgium (Tubize)**  
Stage ingénieur 2025–2026 — Maxime D'Angelo  

---

## Objectif

Améliorer la qualité audio et la transcription automatique des communications radio
entre pilotes de chasse et contrôle aérien, dans le cadre du projet ACE4ACES
(Advanced Cockpit Enhancements for Air Combat Efficiency and Success).

---

## Résultats clés

| Tâche | Modèle | Métrique | Score |
|-------|--------|----------|-------|
| Débruitage | SGMSE+ fine-tuné v5 | PESQ (validation synthétique) | **3,31 / 4,5** |
| Débruitage | SGMSE+ fine-tuné v5 | STOI (validation synthétique) | **0,91 / 1,0** |
| Transcription | Whisper-small + prompt ATC/NATO | WER (phrases militaires TTS) | **1,7 %** |
| Transcription | Whisper-small fine-tuné ATCO2 | WER (validation ATC réel) | **5,3 %** |

> **Limite principale documentée :** tous les modèles de débruitage dégradent les métriques
> sur les vrais fichiers radio (gap dataset synthétique / signal réel). Le bottleneck est
> la représentativité des données d'entraînement, pas l'architecture.

---

## Utilisation rapide

```bash
# Installation
python -m venv venv && source venv/bin/activate
pip install -r requirements-linux.txt   # Linux / ROCm AMD
# ou : requirements.txt pour Windows / CUDA

# Démo transcription sur un fichier audio
python demo_pipeline.py "Dataset Radio (2)/02__kwahmah_atc002.flac"

# Pipeline complet (batch sur tous les fichiers)
python pipeline_denoise_transcribe.py

# Figures de présentation
python generate_demo_figures.py "Dataset Radio (2)/06_kwahmah_atc006.flac"
```

Sorties dans `runs/demo/` : WAV intermédiaires + JSON horodaté + TXT segments.

---

## Architecture du pipeline

```
Signal radio (FLAC/WAV)
    │
    ▼
┌─────────────────────────────┐
│  1. Compression speech-aware │  Cible −10 dBFS, gain max 40 dB
└────────────┬────────────────┘
             │
    ▼
┌─────────────────────────────┐
│  2. Débruitage noisereduce  │  Spectral subtraction adaptative
└────────────┬────────────────┘
             │
    ▼
┌─────────────────────────────┐
│  3. Transcription Whisper   │  Prompt ATC/NATO militaire
│     + fenêtre glissante     │  Chunks 25s / overlap 2s
│     + anti-hallucination    │  4 indicateurs
│     + corrections post-ASR  │  Callsigns, fréquences, phraséologie
└────────────┬────────────────┘
             │
    ▼
Sortie JSON horodatée + TXT + WAV nettoyé
```

---

## Fichiers principaux

### Pipeline opérationnel
| Fichier | Rôle |
|---------|------|
| `pipeline_denoise_transcribe.py` | Pipeline complet batch (tous les fichiers) |
| `demo_pipeline.py` | Démo étape par étape (un fichier, affichage console) |
| `generate_demo_figures.py` | Figures comparatives PNG (présentation) |
| `inference.py` | Inférence débruitage seul (SpectralResUNet + Demucs) |

### Entraînement
| Fichier | Rôle |
|---------|------|
| `create_dataset.py` | Génération dataset synthétique cockpit v5 (3000+300 paires) |
| `train.py` | Entraînement SpectralResUNet / WaveformResUNet |
| `train_sgmse.py` | Fine-tuning SGMSE+ sur dataset cockpit |
| `finetune_whisper_atco2.py` | Fine-tuning décodeur Whisper sur ATCO2 + militaire |

### Évaluation et analyse
| Fichier | Rôle |
|---------|------|
| `evaluate_pretrained_atc.py` | Comparaison modèles Whisper ATC sur fichiers radio |
| `test_military_transcription.py` | WER sur 20 phrases militaires par catégorie |
| `test_sgmse_finetuned.py` | PESQ/STOI SGMSE+ fine-tuné sur fichiers réels |
| `analyze_radio.py` | Analyse spectrale des fichiers radio (bande, SNR) |
| `testDonnees.py` | Comparaison checkpoints SpectralResUNet par époque |

### Scripts expérimentaux (modèles écartés — conservés pour traçabilité)
| Fichier | Modèle | Raison de l'abandon |
|---------|--------|---------------------|
| `pipeline.py` | Pipeline initial avec VoiceFixer | Dégradation perceptuelle |
| `train_metricgan.py` | MetricGAN+ fine-tuné | Masquage spectral inefficace in-band |
| `train_sepformer.py` | SepFormer fine-tuné | PESQ 1,51, inférieur à SGMSE+ |
| `finetune_whisper_military.py` | Whisper sur TTS seul | Pas de généralisation au domaine réel |

---

## Modèles et checkpoints

| Modèle | Chemin | Taille | Usage |
|--------|--------|--------|-------|
| SGMSE+ fine-tuné v5 | `checkpoints/sgmse/best_model.pt` | ~500 Mo | Débruitage |
| Whisper ATCO2 v4 | `checkpoints/whisper_atco2/best/` | 925 Mo | Transcription ATC |
| Whisper-small (base) | `~/.cache/whisper/small.pt` | 461 Mo | Transcription militaire |
| SpectralResUNet | `checkpoints/best_model.pt` | ~50 Mo | Super-résolution spectrale |

---

## Documentation technique

| Document | Contenu |
|----------|---------|
| `DOCUMENTATION_PROJET.md` | Architecture complète, toutes les expériences, conclusions détaillées |
| `docs/Rapport_Transcription_ASR_TFE.docx` | Rapport scientifique ASR (9 sections, formules, résultats) |
| `docs/SGMSE_Documentation.docx` | Documentation technique SGMSE+ (architecture, SDE, fine-tuning) |
| `docs/Documentation_Transcription_ASR.docx` | Documentation technique transcription |
| `rapport hebdo/` | 6 rapports hebdomadaires de progression |

---

## Conclusions architecturales

1. **Le bottleneck est le dataset, pas le modèle.** Tous les modèles (SGMSE+, MetricGAN+,
   SepFormer) obtiennent de bonnes métriques sur données synthétiques mais dégradent les
   signaux radio réels. Des données cockpit réelles sont indispensables.

2. **Le prompt engineering surpasse le fine-tuning pour le vocabulaire militaire.**
   WER 1,7 % (prompt) vs 14,8 % (fine-tuning ATCO2) sur phrases militaires.
   Le fine-tuning ATCO2 est supérieur pour le format numérique OACI.

3. **Le décodeur est le facteur limitant en ASR, pas l'encodeur acoustique.**
   Whisper-small/2000 samples ≈ Whisper-medium/8000 samples en qualité ATC.

4. **Maturité TRL 3-4.** Prototype de laboratoire validé sur données synthétiques.
   Prérequis pour TRL 7-8 : données cockpit réelles, temps réel, certification DO-178C.

---

## Environnement technique

- **Langage :** Python 3.10
- **Framework ML :** PyTorch 2.5.1 + ROCm 6.2 (GPU AMD RX 6800, 16 Go VRAM)
- **Dépendances :** `requirements-linux.txt` (Linux/ROCm) · `requirements.txt` (Windows/CUDA)
- **Fonctionnement offline :** 100 % local, aucune connexion internet requise à l'inférence

---

*Documentation détaillée dans `DOCUMENTATION_PROJET.md` — contient l'historique complet des expériences,
les paramètres de chaque modèle, et les enseignements architecturaux.*
