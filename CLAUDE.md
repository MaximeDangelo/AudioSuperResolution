# CLAUDE.md - Audio Super-Resolution (Pipeline Hybride)

## Description du projet

Projet d'**hyper-resolution audio** pour des enregistrements de communications radio (aviation/militaire). Pipeline hybride combinant :
1. **Demucs** (Facebook) ou **SepFormer** (SpeechBrain) pour le debruitage waveform
2. **VoiceFixer** pour la super-resolution pre-entrainee (sortie 44.1 kHz)
3. **SpectralResUNet** (custom) fine-tune sur des degradations radio synthetiques
4. **MetricGAN+** (SpeechBrain) pour le polissage final optimise PESQ (optionnel)

Spin-off du pipeline de debruitage situe dans `../New/` (qui utilise aussi Demucs + Whisper).

## Contexte

Les fichiers d'entree sont des communications radio avec :
- **Bande passante limitee** : ~300-3400 Hz (radio AM)
- **Bruit de fond** : souffle HF, interference, crackling, distorsion
- **Signal vocal degrade** : harmoniques perdues, compression excessive
- L'objectif est de reconstruire les frequences manquantes (4-20 kHz)

Note : DeepFilterNet a ete teste et abandonne (degrade l'intelligibilite sur signaux radio).

## Architecture

### Pipeline d'inference (`pipeline.py` ou `inference.py`)

```
OGG Radio -> [Demucs OU SepFormer] (denoise dry/wet 50%) -> VoiceFixer (SR 44.1kHz) -> [SpectralResUNet] -> [MetricGAN+] -> WAV 44.1kHz
```

### Pipeline d'entrainement

```
create_dataset.py -> train.py -> checkpoints/best_model.pt
```

## Fichiers

| Fichier | Role |
|---------|------|
| `pipeline.py` | Pipeline hybride complet (Demucs + VoiceFixer + KPIs) |
| `create_dataset.py` | Generation dataset de paires (clean/degrade) a 44.1 kHz |
| `train.py` | Entrainement du SpectralResUNet (fine-tuning) |
| `inference.py` | Inference complete avec les 3 etapes |
| `requirements.txt` | Dependances Python |

## Execution

```bash
# 1. Installer
python -m venv venv
venv\Scripts\pip install -r requirements.txt

# 2. Test rapide (pipeline pre-entraine uniquement, pas de fine-tuning)
venv\Scripts\python pipeline.py

# 3. Pour le fine-tuning : generer le dataset puis entrainer
venv\Scripts\python create_dataset.py
venv\Scripts\python train.py

# 4. Inference avec le modele fine-tune
venv\Scripts\python inference.py
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

## Configuration

**pipeline.py** :
- `DENOISE_ENGINE` : "demucs" | "sepformer" | "none"
- `DENOISE_DRY_WET` : 0.5 (50/50 original/denoise)
- `USE_METRICGAN` : True/False (polissage final PESQ)
- `VOICEFIXER_MODE` : 0 (normal), 1 (speech), 2 (restauration)
- `MAX_DURATION_S` : 120s pour tests rapides

**create_dataset.py** :
- `TARGET_SR` : 44100 Hz
- `N_TRAIN` / `N_VAL` : 5000 / 500
- Degradations : bandpass, downsample, bruit blanc/rose, crackling, interference, clipping

**train.py** :
- `BATCH_SIZE` : 8
- `LEARNING_RATE` : 3e-4
- `EPOCHS` : 80
- `N_FFT` : 2048, `HOP_LENGTH` : 512

## Donnees

- Entree radio : `../New/DatasetRadioCom/*.ogg`
- Dataset synthetique : `dataset/{train,val}/{clean,raw}/` (44.1 kHz)
- Checkpoints : `checkpoints/best_model.pt`
- Modeles pre-entraines SpeechBrain : `pretrained_models/` (telecharges auto)
- Resultats : `output/` (WAV + spectrogrammes + CSV KPIs)

## Dependances

- `torch` / `torchaudio` - Deep learning
- `demucs` - Debruitage waveform (Facebook)
- `voicefixer` - Super-resolution pre-entrainee
- `datasets` - HuggingFace (LibriSpeech)
- `soundfile` - I/O audio
- `numpy`, `scipy` - Traitement signal
- `matplotlib` - Graphiques
- `librosa`, `tqdm` - Utilitaires
- `speechbrain` - SepFormer (debruitage) + MetricGAN+ (polissage PESQ)

## Conventions de code

- **Encodage console** : `sys.stdout.reconfigure(encoding="utf-8")`
- **Tableaux console** : caracteres ASCII uniquement
- **Matplotlib** : Backend `Agg`, pas d'accents dans les titres/labels
- **Langue** : Commentaires en francais (sans accents dans matplotlib)
