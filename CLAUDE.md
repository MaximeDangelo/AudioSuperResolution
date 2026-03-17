# CLAUDE.md - Audio Super-Resolution (Pipeline Hybride)

## Description du projet

Projet d'**hyper-resolution audio** pour des enregistrements de communications radio (aviation/militaire). Pipeline hybride combinant :
1. **Demucs** (Facebook) pour le debruitage waveform (100%, sans dry/wet)
2. **SpectralResUNet** (custom) fine-tune sur des degradations radio synthetiques (denoise spectral + reconstruction HF)
3. **MetricGAN+** (SpeechBrain) pour le polissage final optimise PESQ (optionnel)

Note : VoiceFixer a ete teste et retire du pipeline (degradait les signaux radio : ecrasement des frequences, artefacts vocaux).

Spin-off du pipeline de debruitage situe dans `../New/` (qui utilise aussi Demucs + Whisper).

## Contexte

Les fichiers d'entree sont des communications radio avec :
- **Bande passante limitee** : ~300-3400 Hz (radio AM)
- **Bruit de fond** : souffle HF, interference, crackling, distorsion
- **Signal vocal degrade** : harmoniques perdues, compression excessive
- L'objectif est de reconstruire les frequences manquantes (4-20 kHz)

Note : DeepFilterNet a ete teste et abandonne (degrade l'intelligibilite sur signaux radio).

## Architecture

### Pipeline d'inference (`inference.py`)

```
Radio (OGG/FLAC/MP3) -> Demucs (denoise 100%) -> resample 44.1kHz -> SpectralResUNet (denoise spectral + reconstruction HF) -> [MetricGAN+] -> WAV 44.1kHz
```

### Pipeline d'entrainement

```
create_dataset.py (degradations radio + Demucs) -> train.py -> checkpoints/best_model.pt
```

Le dataset d'entrainement applique Demucs sur les paires degradees pour que le SpectralResUNet apprenne a corriger la sortie de Demucs (pas le signal brut).

## Fichiers

| Fichier | Role |
|---------|------|
| `pipeline.py` | Pipeline hybride initial (ancien, avec VoiceFixer) |
| `create_dataset.py` | Generation dataset de paires (Demucs(degrade)/clean) a 44.1 kHz |
| `train.py` | Entrainement du SpectralResUNet (fine-tuning) |
| `inference.py` | Inference complete (Demucs + SpectralResUNet + MetricGAN+) |
| `analyze_radio.py` | Analyse spectrale des fichiers radio reels |
| `requirements.txt` | Dependances Python (Windows) |
| `requirements-linux.txt` | Dependances Python (Linux/ROCm) |
| `setup_linux.sh` | Script d'installation Linux avec ROCm |

## Execution

```bash
# Linux (avec ROCm pour GPU AMD)
source venv/bin/activate
python create_dataset.py   # Generer le dataset (inclut Demucs)
python train.py             # Entrainer le SpectralResUNet
python inference.py         # Inference sur fichiers radio
```

## Modele SpectralResUNet

ResUNet operant dans le domaine STFT (spectrogramme) :
- **Entree** : signal audio 44.1 kHz -> STFT -> magnitude log
- **Encoder** : 4 blocs (Conv2D stride 2 + ResBlock) : 1 -> 32 -> 64 -> 128 -> 256
- **Decoder** : 4 blocs (ConvTranspose2D + skip connections)
- **Sortie** : masque spectral (filtre bruit) + estimation HF (reconstruit frequences)
- **Reconstruction** : masque * magnitude + HF, puis iSTFT avec phase originale
- **Loss** : L1 + Multi-Resolution STFT Loss
- **Optimizer** : AdamW + CosineAnnealing
- **Early stopping** : patience 15 epochs sur val_loss
- **Metriques** : PESQ, STOI toutes les 5 epochs

## Configuration

**inference.py** :
- `DENOISE_ENGINE` : "demucs" | "sepformer" | "none"
- `DEMUCS_DRY_WET` : 0.0 (100% Demucs)
- `USE_VOICEFIXER` : False (retire du pipeline)
- `USE_METRICGAN` : True (polissage final PESQ)
- `USE_FINETUNE` : True (SpectralResUNet)
- `MAX_DURATION_S` : 120s pour tests rapides

**create_dataset.py** :
- `TARGET_SR` : 44100 Hz
- `APPLY_DEMUCS_TO_RAW` : True (applique Demucs aux paires degradees)
- `N_TRAIN` / `N_VAL` : 5000 / 500
- Degradations : bandpass, downsample, bruit blanc/rose, crackling, interference, clipping, AGC, cockpit reverb, radio dropout

**train.py** :
- `BATCH_SIZE` : 8
- `LEARNING_RATE` : 3e-4
- `EPOCHS` : 80
- `N_FFT` : 2048, `HOP_LENGTH` : 512
- `EARLY_STOPPING_PATIENCE` : 15

## Donnees

- Entree radio : `Dataset Radio (2)/*.flac` et `../New/DatasetRadioCom/*.ogg`
- Dataset synthetique : `dataset/{train,val}/{clean,raw}/` (44.1 kHz)
- Checkpoints : `checkpoints/best_model.pt`
- Modeles pre-entraines SpeechBrain : `pretrained_models/` (telecharges auto)
- Resultats : `output/` (WAV + spectrogrammes + CSV KPIs)
- Rapports : `rapport hebdo/` (fichiers .docx hebdomadaires)

## Dependances

- `torch` / `torchaudio` - Deep learning (ROCm pour GPU AMD)
- `demucs` - Debruitage waveform (Facebook)
- `datasets` - HuggingFace (LibriSpeech)
- `soundfile` - I/O audio
- `numpy`, `scipy` - Traitement signal
- `matplotlib` - Graphiques
- `librosa`, `tqdm` - Utilitaires
- `speechbrain` - MetricGAN+ (polissage PESQ)
- `pesq`, `pystoi`, `mir_eval` - Metriques audio

## Conventions de code

- **Encodage console** : `sys.stdout.reconfigure(encoding="utf-8")`
- **Tableaux console** : caracteres ASCII uniquement
- **Matplotlib** : Backend `Agg`, pas d'accents dans les titres/labels
- **Langue** : Commentaires en francais (sans accents dans matplotlib)
