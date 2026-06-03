# CLAUDE.md - Audio Super-Resolution (Pipeline Hybride)

## Description du projet

Projet d'**hyper-resolution audio** pour des enregistrements de communications radio (aviation/militaire). Pipeline hybride combinant :
1. **Demucs** (Facebook) pour le debruitage waveform (100%, sans dry/wet)
2. **SpectralResUNet** (custom) fine-tune sur des degradations radio synthetiques (denoise spectral + reconstruction HF)
3. **MetricGAN+** (SpeechBrain) pour le polissage final optimise PESQ (optionnel)

Notes :
- VoiceFixer a ete teste et retire du pipeline (degradait les signaux radio : ecrasement des frequences, artefacts vocaux).
- AudioSR (modele de diffusion, super-resolution 16kHz->48kHz) a ete teste en mars 2026. Fonctionne sur GPU AMD mais degrade l'intelligibilite des signaux radio (hallucination de mots, WER > 120%).

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
| `train.py` | Entrainement SpectralResUNet ou WaveformResUNet (`MODEL_TYPE` configurable) |
| `inference.py` | Inference complete (Demucs + SpectralResUNet + MetricGAN+) |
| `testDonnees.py` | Comparaison des checkpoints sur un fichier audio (genere runs/) |
| `analyze_radio.py` | Analyse spectrale des fichiers radio reels |
| `analyze_hf.py` | Analyse reconstruction HF : energie > 4kHz par etape du pipeline |
| `test_unet_untrained.py` | Genere des audios avec ResUNet poids aleatoires (baseline zero) |
| `test_pipeline_comparison.py` | Compare RAW / Demucs / AudioSR / Demucs+AudioSR sur 4 fichiers ATC |
| `test_wer.py` | Calcule le WER Whisper entre variantes du pipeline |
| `test_multi_baseline.py` | Comparaison multi-fichiers avec metriques (STOI, LSD, PESQ) |
| `train_metricgan.py` | Fine-tuning MetricGAN+ sur dataset radio (SpeechBrain) |
| `test_finetuned_real.py` | Test MetricGAN+ fine-tune sur vrais fichiers radio |
| `test_sgmse.py` | Test SGMSE+ (diffusion) sur vrais fichiers radio |
| `test_denoising.py` | Test denoising MetricGAN+ pre-entraine avec metriques |
| `analyze_dataset.py` | Analyse detaillee paires clean/noisy du dataset |
| `train_sgmse.py` | Fine-tuning SGMSE+ sur dataset radio (diffusion, 65.6M params) |
| `train_sepformer.py` | Fine-tuning SepFormer sur dataset radio (teste puis abandonne) |
| `test_resemble.py` | Test Resemble Enhance (denoise + super-resolution) |
| `test_sgmse_finetuned.py` | Test SGMSE+ fine-tune (EMA) sur fichiers radio reels + validation |
| `docs/SGMSE_Documentation.docx` | Documentation technique complete SGMSE+ (13 pages, en francais) |
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

## Modele SpectralResUNet (v2)

ResUNet operant dans le domaine STFT (spectrogramme) :
- **Entree** : signal audio 44.1 kHz -> STFT -> magnitude log
- **Encoder** : 4 blocs (Conv2D stride 2 + ResBlock) : 1 -> 32 -> 64 -> 128 -> 256
- **Decoder** : 4 blocs (ConvTranspose2D + skip connections)
- **Sortie** : masque spectral (filtre bruit) + estimation HF gatee par energie
- **Reconstruction** : masque * magnitude + HF * energy_gate, puis iSTFT avec phase originale
- **Loss** : L1 + Multi-Resolution STFT Loss + Identity Loss (x0.3)
- **Optimizer** : AdamW + CosineAnnealing
- **Early stopping** : patience 15 epochs sur val_loss
- **Metriques** : PESQ, STOI toutes les 5 epochs

### Corrections v2 (mars 2026)
- `IDENTITY_WEIGHT = 0.3` : penalise la sur-transformation (evite degradation sur signal propre)
- `CLEAN_PAIR_RATIO = 0.15` : 15% des exemples sont des paires (clean, clean) -> apprend l'identite
- **HF energy gate** : `hf * sigmoid(log(energy) + 4)` -> supprime hallucination HF sur silence
- **Silence gate** dans `testDonnees.py` : bypass du modele si RMS < 0.005 sur trames 100ms

### Conclusions tests perceptuels
- Demucs seul = resultat operationnel convaincant (meilleure partie du pipeline)
- ResUNet v1 = degradation a l'ecoute + hallucination sur silence (causes identifiees et corrigees en v2)
- ResUNet v2 (identity loss + HF gate + silence gate) = amelioration mais toujours moins bon que Demucs seul a l'ecoute
- Le ResUNet fait du debruitage HF, pas de reconstruction de nouvelles frequences (pas de super-resolution stricte)
- Pour de la vraie super-resolution : architecture generative recommandee (WSRGlow, normalizing flow)

### Tests comparatifs pipelines (mars 2026)

Comparaison sur 4 fichiers ATC (heathrow, hongkong, us_approach, atc05) :
- Pipeline teste : RAW / Demucs seul / AudioSR seul / Demucs+AudioSR
- Methode : transcription Whisper (base) + WER

Resultats WER moyen (par rapport a la transcription du RAW) :

| Pipeline | WER moyen |
|----------|-----------|
| Demucs seul | 68% |
| Demucs + AudioSR | 89% |
| AudioSR seul | 100% |

Resultats detailles par fichier :

| Fichier | Demucs | AudioSR | Demucs+AudioSR |
|---------|--------|---------|----------------|
| atc05 | 44% | 56% | 56% |
| heathrow | 44% | 75% | 100% |
| hongkong | 100% | 100% | 100% |
| us_approach | 83% | 167% | 100% |

**Conclusion** : tous les modeles pre-entraines (Demucs, AudioSR, MetricGAN+) degradent l'intelligibilite des signaux radio cockpit. Ces modeles n'ont jamais vu de signaux radio et sont hors distribution. AudioSR hallucine des mots inexistants. Demucs est le moins mauvais mais perd quand meme de l'information.

### Modele WaveformResUNet (temporel, v3 - mars 2026)

ResUNet temporel operant directement sur la forme d'onde (pas de STFT) :
- **Entree/sortie** : signal audio 44.1 kHz brut (pas de passage Demucs en amont)
- **Encoder** : 4 blocs Conv1D stride 4 : 1 -> 64 -> 128 -> 256 -> 512
- **Decoder** : 4 blocs ConvTranspose1D + skip connections
- **Sortie** : apprentissage residuel (output = input + prediction)
- **Avantage theorique** : preserve la phase naturellement (pas d'artefacts STFT/iSTFT)
- **Dataset** : paires raw -> clean SANS Demucs (`APPLY_DEMUCS_TO_RAW = False`)
- **Resultats (80 epochs)** : PESQ=1.27, STOI=0.569, val_loss=2.40
- **Probleme** : artefacts aigus (bips HF) non resolus meme apres 80 epochs
- **Conclusion** : le modele convolutif simple ne suffit pas pour la tache raw -> clean. La tache est trop complexe sans pre-traitement (Demucs). L'architecture est inadequate pour la super-resolution (CNN deterministe = moyenne des HF possibles).

### Modeles testes et ecartes

| Modele | Raison de l'abandon |
|--------|---------------------|
| VoiceFixer | Ecrasement des frequences, artefacts vocaux |
| DeepFilterNet | Degrade l'intelligibilite sur signaux radio |
| AudioSR (diffusion) | Hallucination de mots, WER moyen 100% |
| SpectralResUNet v1 | Hallucination sur silence, degradation a l'ecoute |
| SpectralResUNet v2 | Ameliore v1 mais toujours inferieur a Demucs seul |
| WaveformResUNet v3 | Artefacts aigus (bips HF), STOI=0.569 vs 0.736 spectral |
| MetricGAN+ pre-entraine | Aucun effet sur bruit radio (hors domaine) |
| MetricGAN+ fine-tune | Masquage spectral ne separe pas voix/bruit in-band |
| SGMSE+ pre-entraine | Aucun effet sur bruit radio (hors domaine) |
| SGMSE+ fine-tune v3 (15 ep, ATC) | PESQ 3.15 synthetique, non teste sur reel (dataset remplace) |
| SGMSE+ fine-tune v4 (11 ep, chasse) | PESQ 2.95 synthetique, degrade les fichiers reels (PESQ/STOI baissent) |
| SGMSE+ fine-tune v5 (dataset v5) | En cours - bruits non-stationnaires augmentes (gunfire/explosions 50%) |
| SepFormer fine-tune | Masquage spectral limite pour bruit in-band (PESQ 1.51) |
| Resemble Enhance | Pre-entraine hors domaine, non teste en fine-tuning |

### Enseignements architecturaux

- **Modeles pre-entraines hors domaine** : tous degradent les signaux radio (Demucs, AudioSR, MetricGAN+, SGMSE+, VoiceFixer, DeepFilterNet, Resemble Enhance). Les signaux radio cockpit sont hors distribution.
- **Spectral vs temporel** : le modele spectral (ResUNet STFT) donne de meilleurs metriques que le temporel (waveform), mais les deux restent inferieurs a Demucs seul a l'ecoute.
- **Entrainer sur sortie Demucs vs raw direct** : entrainer sur sortie Demucs = peu de marge d'amelioration (Demucs fait deja le gros du travail). Entrainer sur raw direct = tache trop difficile pour un CNN simple.
- **CNN deterministe vs generatif** : les CNN avec loss L1/STFT predisent la moyenne des sorties possibles -> resultat flou ou artefacts. La super-resolution audio necessite des architectures generatives (normalizing flow, diffusion, GAN).
- **Reconstruction HF sur signal radio** : les HF (>4 kHz) n'ont jamais ete transmises par la radio AM. Les "reconstruire" revient a les halluciner. L'apport a l'intelligibilite est minime ; le debruitage seul apporte 90% de la valeur operationnelle.
- **Masquage spectral vs generatif pour bruit in-band** : MetricGAN+ (masque spectral LSTM) ne fonctionne pas quand le bruit et la voix partagent la meme bande frequentielle (300-3400 Hz). Un modele generatif (diffusion/SGMSE+) qui regenere le signal propre est mieux adapte car il peut distinguer voix et bruit par leur structure temporelle (formants, prosodie vs bruit stationnaire).
- **Le bottleneck est le dataset, pas le modele** : tous les modeles (MetricGAN+, SepFormer, SGMSE+) obtiennent de bonnes metriques sur le dataset synthetique mais degradent les fichiers radio reels. Le gap synthetique/reel (type de bruit, voix source, canal radio) est le probleme fondamental. Ameliorer le modele ne sert a rien tant que le dataset n'est pas representatif. La solution serait d'utiliser de vrais enregistrements radio pour l'entrainement.

### Tests denoising SpeechBrain (mars 2026)

Travail realise depuis `../speechbrain-develop/` puis migre ici.

#### Analyse des fichiers radio reels

Caracteristiques des communications radio de pilotes de chasse :
- Bande passante tres etroite : 200 Hz - 4 kHz (certains fichiers < 2 kHz)
- Sample rates tres variables : 7 kHz a 48 kHz
- Bruit radio : souffle, crackling, interferences, dropout
- **Le bruit ET la voix occupent la meme bande** (300-3400 Hz) car tout passe par le meme canal radio

#### Qualite des fichiers "clean" de reference (Test1/)

| Fichier | Flatness | Verdict |
|---------|----------|---------|
| atc002_clean | 0.038 | BRUITE - mauvaise reference |
| atc003_clean | 0.026 | BRUITE - mauvaise reference |
| atc005_clean | 0.013 | OK |
| atc006_clean | 0.011 | OK |
| heathrow_clean | 0.004 | OK - propre |
| hongkong_clean | 0.004 | OK - propre |

**Important** : les metriques PESQ/STOI sur atc002 et atc003 sont faussees car les references "clean" contiennent du bruit residuel. Seuls atc005, atc006, heathrow et hongkong donnent des metriques fiables.

#### MetricGAN+ pre-entraine (Voicebank)

Test sur les vrais fichiers radio : **aucun effet** (PESQ 1.09 -> 1.14). Le modele ne reconnait pas le bruit radio.

#### MetricGAN+ fine-tune

- **Dataset v1** (bruit fullband, SNR trop bas) : PESQ 2.91 sur validation synthetique, mais aucun effet sur vrais fichiers radio
- **Dataset v2** (bruit en bande etroite, SNR calibre sur analyse reelle) : PESQ 2.63 sur validation synthetique, aucun effet sur vrais fichiers radio
- **Conclusion** : MetricGAN+ (masquage spectral LSTM, 2M params) est une impasse. Le masquage spectral ne peut pas separer voix et bruit quand ils partagent la meme bande.

#### SGMSE+ pre-entraine (Voicebank)

Test sur les vrais fichiers radio : **aucun effet** (memes resultats que MetricGAN+). Modele hors domaine.

#### Dataset synthetique v3 (ATC civil)

Corrections apportees apres analyse comparative vrais fichiers radio vs synthetique :
1. **Bruit filtre dans la bande du canal** : tous les bruits (blanc, rose, cockpit, HF) sont bandpass-filtres dans la meme bande que le signal (realiste : tout passe par le meme canal radio)
2. **SNR calibre** : 18-40 dB bruit blanc, 20-40 dB bruit rose (calibre sur les vrais fichiers ou le SNR est ~20-25 dB)
3. **Bruit cockpit dominant** : prob 0.7 (turbine, flux d'air, vibrations mecaniques - specifique chasse)
4. **Interference tonale dans la bande** : frequences contraintes a la bande du canal, pas de HF hors bande
5. **Clean = voix bandpassee sans bruit** : le modele apprend uniquement le denoising, pas la super-resolution

Statistiques du dataset :
- 3000 paires train + 300 val a 16 kHz
- Correlation clean/noisy : 0.94 moyenne
- SNR : 100% >= 0 dB, 66% >= 5 dB, 32% >= 20 dB

#### Dataset synthetique v4 (cockpit chasse)

Refonte complete du dataset pour representer un environnement cockpit de chasse (documentation NATO RTO-EN-HFM-111, HIWIRE, Elie et al. Interspeech 2021) :

**Degradations specifiques chasse ajoutees :**
- **Bruit cockpit 5 composantes** : moteur/combustion (spectre brownien -3 a -6 dB/octave), vibrations structurelles (<200Hz), aerodynamique (>600Hz), turbine HF (2-5kHz), enveloppe non-stationnaire
- **Masque O2** (prob 0.85) : attenuation HF -7 a -10 dB au-dessus de 3 kHz, attenuation LF -0.5 a -3 dB (Elie et al.)
- **Tirs/rafales** (prob 0.25) : impulsions 2-5ms avec decroissance exponentielle, taux 5-100 coups/s (tire espace a canon Vulcan), saturation burst, amplitude 8-25x RMS signal
- **Bruits impulsionnels** (prob 0.30) : explosions (200-500ms), post-combustion (500-1500ms), alertes cockpit (300-800ms, tonales), manoeuvres G (200-400ms)
- **Bruit cockpit** SNR 5-20 dB (calibre HIWIRE), prob 0.9
- **AGC** : attack 1-3ms (doc ~2ms), gain max 30-40 dB (doc 40 dB range)
- **Bruit residuel sur le clean** (prob 0.8, SNR 30-45 dB) : simule le bruit du micro meme en conditions ideales
- **Normalisation RMS** : target 0.01-0.04 (calibre sur vrais fichiers radio)

**Degradations retirees/modifiees :**
- Formant shift du masque O2 retire (causait echo a l'ecoute)
- Reverb cockpit reduite a 5% prob, wet 5-20% (etait 15% prob, 15-45% wet)

**Parametres principaux** :
```python
DEGRAD_PARAMS = {
    "bandpass_low_range": (280, 400),
    "bandpass_high_range": (2500, 3400),
    "downsample_rates": [4000, 6000, 8000],
    "max_noise_degradations": 4,
    "white_noise_snr_range": (10, 30),
    "pink_noise_snr_range": (10, 30),
    "cockpit_noise_prob": 0.9,
    "cockpit_noise_snr_range": (5, 20),
    "o2_mask_prob": 0.85,
    "gunfire_prob": 0.25,
    "impulse_noise_prob": 0.30,
    "clean_residual_noise_prob": 0.8,
}
```

#### Dataset synthetique v5 (cockpit chasse + bruits impulsionnels augmentes - create_dataset.py - version actuelle)

Meme base que v4. Seule modification : probabilites des bruits non-stationnaires doublees apres constat que le modele v4 ne supprimait pas les explosions/tirs sur les audios de test.

```python
"gunfire_prob": 0.50,      # etait 0.25 en v4
"impulse_noise_prob": 0.50, # etait 0.30 en v4
```

Dataset regenere le 31 mars 2026. 3000 paires train + 300 val a 16 kHz.

### Fine-tuning SGMSE+ (mars 2026 - en cours)

**Pourquoi SGMSE+** (et pas MetricGAN+ ou SepFormer) :
- Modele de diffusion (65.6M params) qui **regenere** le signal propre au lieu de masquer le bruit
- Adapte au bruit in-band : modelise la structure temporelle de la parole vs bruit stationnaire
- Etat de l'art pour la separation voix/bruit en conditions difficiles
- Meilleure generalisation cross-dataset que les modeles discriminatifs (demontre dans le paper)

**Paper** : Richter et al., "Speech Enhancement and Dereverberation with Diffusion-based Generative Models", IEEE/ACM TASLP 2023 (arXiv:2208.05830)

**Architecture SGMSE+** :
- Score model : NCSN++ (U-Net multi-resolution) operant sur spectrogramme STFT complexe
- SDE : Ornstein-Uhlenbeck Variance Exploding (OUVE) - diffuse vers le signal bruite, pas vers du bruit pur
- Sampling : Predictor-Corrector (30 pas de diffusion inverse)
- Entree/sortie : 16 kHz, STFT n_fft=510, hop=128
- Transformation spectrale : magnitude^0.5 * 0.15 (compression + normalisation)
- Documentation technique complete : `docs/SGMSE_Documentation.docx`

**SepFormer (teste avant SGMSE+)** :
- Transformer deterministe (masquage spectral, ~26M params)
- Fine-tune sur le meme dataset : PESQ 1.23 -> 1.51 (epoch 5)
- Abandonne au profit de SGMSE+ : le masquage spectral est fondamentalement limite pour le bruit in-band

**Configuration fine-tuning SGMSE+** (`train_sgmse.py`) :
- `BATCH_SIZE` : 2 (limite VRAM 16 GB)
- `LEARNING_RATE` : 1e-5 (fine-tuning, pas from scratch)
- `EPOCHS` : 30
- `SEGMENT_LENGTH_S` : 2.0s (256 trames STFT)
- `PATIENCE` : 10 (early stopping)
- `GRAD_CLIP` : 1.0
- `RESUME_FROM` : checkpoint automatique (reprise apres crash)
- EMA decay : 0.999 (Exponential Moving Average des poids)
- Optimizer : Adam sur les params du DNN uniquement

**Resultats fine-tuning sur dataset v3 (ATC civil, 15 epochs sur 30)** :

| Epoch | Train Loss | Val Loss | PESQ | STOI |
|-------|-----------|----------|------|------|
| 1 | 441.9 | 416.5 | 3.43 | 0.891 |
| 5 | 380.2 | 379.0 | 3.30 | 0.921 |
| 10 | 370.7 | 381.4 | 3.60 | 0.908 |
| 15 | 362.2 | 338.0 | 3.79 | 0.886 |

**Resultats fine-tuning sur dataset v4 (cockpit chasse, early stopping epoch 21, best epoch 11)** :
- Best val_loss : 569.1 (epoch 11)
- Loss plus elevee que v3 (dataset plus difficile : bruits non-stationnaires, SNR plus bas)

**Test sur echantillons de validation** (epoch 11, EMA, dataset v4) :

| Echantillon | PESQ | STOI |
|-------------|------|------|
| val_00000 | 4.29 | 0.873 |
| val_00005 | 2.91 | 0.876 |
| val_00010 | 2.55 | 0.869 |
| val_00050 | 2.55 | 0.748 |
| val_00100 | 2.48 | 0.809 |
| **Moyenne** | **2.95** | **0.835** |

**Test sur vrais fichiers radio** (epoch 11, EMA, dataset v4) :

| Fichier | PESQ raw | PESQ enh | STOI raw | STOI enh |
|---------|----------|----------|----------|----------|
| atc005 | 2.77 | 2.32 | 0.985 | 0.849 |
| atc006 | 1.20 | 1.24 | 0.968 | 0.850 |
| heathrow | 3.00 | 2.33 | 0.969 | 0.835 |
| hongkong | 2.43 | 1.66 | 0.960 | 0.694 |

**Conclusion dataset v4** : le modele degrade les fichiers radio reels (PESQ et STOI baissent sauf atc006). Les metriques sur synthetique (PESQ 2.95) ne transferent pas au domaine reel. Le gap entre le dataset synthetique et les vrais signaux radio reste le bottleneck principal du projet, quel que soit le modele utilise (MetricGAN+, SepFormer, SGMSE+).

Audios dans `runs/sgmse_test_v3/` (raw + enhanced pour 4 fichiers ATC + 5 echantillons validation).

**Fine-tuning sur dataset v5 (cockpit chasse + bruits impulsionnels augmentes, en cours - mars 2026)** :
- Objectif : meilleure suppression des tirs/explosions (gunfire_prob 0.50, impulse_noise_prob 0.50)
- Repart des poids pre-entraines Voicebank (pas depuis v4)
- Epoch 3 : Train 655.1, Val 588.4 (en cours)

**Bug EMA corrige** : `store_ema()` de SpeechBrain ne fonctionne pas correctement apres `load_state_dict()`. Il faut charger manuellement les shadow_params EMA dans les poids DNN :
```python
shadow_params = ckpt["ema_state_dict"]["shadow_params"]
for p, s in zip(score_model.dnn.parameters(), shadow_params):
    p.data.copy_(s.to(device))
```
Sans ce fix, le modele genere du bruit sature (RMS > 25) au lieu de debruiter.

#### Analyse de representativite du dataset synthetique

Comparaison des caracteristiques spectrales entre vrais fichiers radio et dataset synthetique :

| Aspect | Vrais fichiers radio | Dataset synthetique |
|--------|---------------------|-------------------|
| RMS | 0.01-0.04 (tres faible) | 0.05-0.14 (plus fort) |
| Bande passante | 280-2850 Hz | 300-1800 Hz |
| Repartition Low/Mid | 40-65% / 33-60% | 47-95% / 5-52% |
| SNR estime | 14-15 dB | 12-21 dB |
| Source vocale | Pilotes (stress, masque O2) | LibriSpeech (lecture studio) |
| Type de bruit | Crackling reel, interference | Bruit blanc/rose filtre |
| Canal radio | Compression AM reelle | AGC simule |

**Conclusion** : le dataset capture les grandes lignes (bande etroite, pas de HF, SNR correct) mais manque de realisme sur le type de bruit, le niveau du signal, et la voix source. Les bonnes metriques sur le dataset (PESQ 3.15) ne garantissent pas la meme qualite sur les vrais fichiers radio.

### Strategie globale du projet

```
Audio radio pilote de chasse (bruite, bande etroite ~300-3400 Hz)
    |
    v
 ETAPE 1 - DENOISING  <-- en cours (SGMSE+ fine-tune dataset v5, epoch 3/30)
    |
    v
 Audio propre (toujours bande etroite, mais sans bruit)
    |
    v
 ETAPE 2 - SUPER-RESOLUTION (pas commencee)
    |
    v
 Audio propre fullband (voix naturelle, 8-16+ kHz)
```

### Ameliorations futures

- **Dataset plus realiste** : utiliser de vrais enregistrements radio (LiveATC, Freesound, ZapSplat) au lieu de degradations synthetiques sur LibriSpeech. Les vrais bruits radio (crackling, interference, compression AM) ont une texture differente du bruit blanc/rose filtre.
- **Voix source adaptee** : entrainer sur des voix de pilotes (stress, masque O2, jargon) plutot que de la lecture studio.
- **Continuer l'entrainement SGMSE+ v5** : en cours (epoch 3/30), dataset v5 avec bruits impulsionnels augmentes (gunfire 50%, explosions 50%).
- **Export C++/ONNX pour deploiement embarque** : pour une utilisation temps reel dans un casque ou un systeme embarque, exporter le modele en ONNX et utiliser ONNX Runtime ou TensorRT. Le code Python (PyTorch) n'est pas adapte au deploiement embarque — le C++ avec inference optimisee reduirait la latence et la consommation memoire. Cela permettrait un traitement en temps reel sur processeur ARM ou DSP audio.
- **Reduction du nombre de pas de diffusion** : SGMSE+ utilise 30 pas (RTF ~1.77x). Des techniques comme la distillation (consistency models) ou le solveur ODE (14 pas, RTF 0.46x) pourraient accelerer l'inference pour le temps reel.
- **Conditionnement VAD/phoneme** : les auteurs de SGMSE+ suggerent d'ajouter un conditionnement sur l'activite vocale et l'identite phonemique pour reduire les artefacts d'hallucination.

## Configuration

**inference.py** :
- `DENOISE_ENGINE` : "demucs" | "sepformer" | "none"
- `DEMUCS_DRY_WET` : 0.0 (100% Demucs)
- `USE_VOICEFIXER` : False (retire du pipeline)
- `USE_METRICGAN` : True (polissage final PESQ)
- `USE_FINETUNE` : True (SpectralResUNet)
- `MAX_DURATION_S` : 120s pour tests rapides

**create_dataset.py** :
- `TARGET_SR` : 16000 Hz (denoising mode)
- `APPLY_DEMUCS_TO_RAW` : False (raw direct, sans Demucs en amont)
- `N_SYNTH_TRAIN` / `N_SYNTH_VAL` : 3000 / 300
- Degradations : bandpass (200-400 -> 2800-3800 Hz), downsample (4-8 kHz), bruit blanc/rose (filtre dans la bande), crackling, interference tonale (dans la bande), clipping, AGC, bruit cockpit (turbine/vibrations, filtre dans la bande), dropout
- Le clean ET le raw subissent le meme bandpass/downsample ; seul le raw recoit le bruit
- Tous les bruits sont bandpass-filtres dans la bande du canal radio (channel_bw)

**train.py** :
- `MODEL_TYPE` : "spectral" ou "temporal"
- `BATCH_SIZE` : 8
- `LEARNING_RATE` : 3e-4
- `EPOCHS` : 80
- `N_FFT` : 2048, `HOP_LENGTH` : 512 (mode spectral uniquement)
- `EARLY_STOPPING_PATIENCE` : 15
- `IDENTITY_WEIGHT` : 0.3
- `CLEAN_PAIR_RATIO` : 0.15
- `RESUME_FROM` : None (repart de zero) ou chemin vers un checkpoint

**testDonnees.py** :
- Usage : `python testDonnees.py <fichier> [start_s] [duration_s]`
- Genere dans `runs/<nom>/` : raw, demucs, unet non entraine, checkpoints epoch 10-60, best_model
- Tous les fichiers ont la meme duree (segment fixe identique)
- Silence gate : bypass du ResUNet sur trames silencieuses (RMS < 0.005)

## Donnees

- Entree radio : `Dataset Radio (2)/*.flac` et `../New/DatasetRadioCom/*.ogg`
- Fichiers "clean" de reference : `Dataset Radio (2)/Test1/` (attention : atc002 et atc003 sont bruites)
- Resultats MetricGAN+ fine-tune : `Dataset Radio (2)/metricgan_finetuned/`
- Resultats SGMSE+ : `Dataset Radio (2)/sgmse_enhanced/`
- Analyse spectrale : `Dataset Radio (2)/analysis/`
- Dataset synthetique denoising : `dataset/{train,val}/{clean,raw}/` (16 kHz, 3000+300 paires)
- Checkpoints SpectralResUNet : `checkpoints/best_model.pt`
- Checkpoints SGMSE+ fine-tune : `checkpoints/sgmse/best_model.pt` (epoch 15, EMA)
- Checkpoints SepFormer fine-tune : `checkpoints/sepformer/best_model.pt` (abandonne)
- Checkpoints MetricGAN+ fine-tune : `results/MetricGAN_radio/4234/save/`
- Modeles pre-entraines SpeechBrain : `pretrained_models/` (telecharges auto)
- Tests SGMSE+ pre-entraine : `runs/sgmse_test/` (raw + enhanced pour 6 fichiers radio + 5 echantillons dataset)
- Tests SGMSE+ fine-tune v4 : `runs/sgmse_test_v3/` (raw + enhanced pour 4 fichiers radio + 5 echantillons validation)
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
- `audiosr` - Super-resolution audio par diffusion (teste et ecarte)
- `openai-whisper` - Transcription automatique (evaluation WER)
- `jiwer` - Calcul WER/CER

## Langage et technologies

- **Langage** : Python 3.10 (tout le projet : entrainement, inference, dataset, tests)
- **Framework ML** : PyTorch 2.x + torchaudio (ROCm pour GPU AMD RX 6800)
- **Bibliotheques speech** : SpeechBrain (SGMSE+, SepFormer, MetricGAN+), Demucs (Facebook)
- **Pourquoi Python et pas C++** : 99% du temps est passe dans les kernels GPU (CUDA/HIP). Python ne fait qu'orchestrer les appels. Le bottleneck est le GPU, pas le langage. L'ecosysteme ML (SpeechBrain, HuggingFace, PyTorch) est en Python.
- **C++ pour le deploiement** : voir section "Ameliorations futures" - export ONNX + ONNX Runtime/TensorRT pour deploiement embarque temps reel.

## Conventions de code

- **Encodage console** : `sys.stdout.reconfigure(encoding="utf-8")`
- **Tableaux console** : caracteres ASCII uniquement
- **Matplotlib** : Backend `Agg`, pas d'accents dans les titres/labels
- **Langue** : Commentaires en francais (sans accents dans matplotlib)
