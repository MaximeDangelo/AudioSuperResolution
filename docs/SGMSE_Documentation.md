pou# SGMSE+ : Documentation Complete

## Speech Enhancement and Dereverberation with Diffusion-based Generative Models

**Auteurs** : Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, Timo Gerkmann
**Affiliation** : Signal Processing Group, Universitat Hamburg, Allemagne
**Publication** : IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023
**Paper** : [arXiv:2208.05830](https://arxiv.org/abs/2208.05830)
**Code** : [github.com/sp-uhh/sgmse](https://github.com/sp-uhh/sgmse)

---

## Table des matieres

1. [Contexte et motivation](#1-contexte-et-motivation)
2. [Principe general](#2-principe-general)
3. [Representation des donnees](#3-representation-des-donnees)
4. [Processus stochastique (SDE)](#4-processus-stochastique-sde)
5. [Objectif d'entrainement (Score Matching)](#5-objectif-dentrainement-score-matching)
6. [Architecture du reseau (NCSN++)](#6-architecture-du-reseau-ncsn)
7. [Inference : echantillonnage (Predictor-Corrector)](#7-inference--echantillonnage-predictor-corrector)
8. [Resultats experimentaux](#8-resultats-experimentaux)
9. [Forces et limites](#9-forces-et-limites)
10. [Application a notre projet](#10-application-a-notre-projet)
11. [Glossaire](#11-glossaire)
12. [References](#12-references)

---

## 1. Contexte et motivation

### Le probleme du rehaussement de la parole

Le rehaussement de la parole (speech enhancement) consiste a recuperer un signal vocal propre a partir d'un enregistrement corrompu par du bruit ou de la reverberation. C'est un probleme fondamental en traitement du signal audio.

### Approches existantes et leurs limites

**Modeles discriminatifs** (MetricGAN+, Conv-TasNet, DCCRN) :
- Apprennent une correspondance directe signal bruite -> signal propre
- Optimisent une loss point a point (L1, L2, PESQ)
- Probleme : necessitent de couvrir toutes les conditions de bruit possibles a l'entrainement
- Probleme : produisent des distorsions de parole desagreables quand le bruit est hors distribution
- Probleme : le masquage spectral ne peut pas separer voix et bruit quand ils partagent la meme bande

**Modeles generatifs anterieurs** (VAE, GAN) :
- VAE : reduction de dimensionnalite dans l'espace latent = perte d'information
- GAN : entrainement instable, mode collapse

**Modeles de diffusion** (SGMSE+) :
- Apprennent une distribution a priori sur la parole propre
- Ne cherchent pas a separer voix/bruit mais a **regenerer** la parole propre
- Meilleure generalisation aux conditions non vues a l'entrainement
- Pas de reduction de dimensionnalite (l'espace latent = meme taille que l'entree)

### Contribution principale de SGMSE+

L'innovation cle : au lieu de diffuser vers du bruit gaussien pur (comme Stable Diffusion pour les images), le processus forward diffuse le signal propre **vers le signal bruite**. Le signal bruite sert de "prior" informatif pour guider la reconstruction.

---

## 2. Principe general

### Analogie

Imagine un tableau ancien recouvert de crasse (le bruit). Deux approches :
- **Discriminative** : tu frottes avec un chiffon pour enlever la crasse. Si tu frottes trop, tu abimes la peinture.
- **Generative (SGMSE+)** : tu apprends a quoi ressemblent les tableaux propres. Puis tu **repeins** le tableau en t'inspirant de ce que tu vois sous la crasse.

### Vue d'ensemble du processus

```
ENTRAINEMENT (Forward Process) :
Signal propre x_0 -----> ajout progressif de bruit -----> signal corrompu x_T (proche du bruite y)
                         (30 etapes, guidees par le signal bruite y)

Le reseau apprend a INVERSER ce processus a chaque etape.

INFERENCE (Reverse Process) :
Signal bruite y (= x_T) -----> retrait progressif du bruit -----> estimation du signal propre x_0
                                (30 etapes, guidees par le score appris)
```

### Difference fondamentale avec les autres modeles de diffusion

| | Diffusion classique (DDPM, Stable Diffusion) | SGMSE+ |
|--|---|---|
| **Forward** | Propre -> bruit gaussien pur | Propre -> signal bruite (observation) |
| **Point de depart reverse** | Bruit aleatoire | Signal bruite + bruit gaussien |
| **Conditionnement** | Via texte, classe, etc. | Directement dans la SDE (drift term) |
| **Avantage** | Generation ex nihilo | Reconstruction fidele au contenu original |

---

## 3. Representation des donnees

### Domaine STFT complexe

SGMSE+ travaille dans le domaine du **spectrogramme STFT complexe**, pas directement sur la forme d'onde.

**Pourquoi le complexe ?**
- Les parties reelle et imaginaire des coefficients STFT ont une structure claire et exploitable
- Le bruit additif gaussien dans le domaine complexe correspond au modele de signal utilise pour le debruitage
- Travailler sur la magnitude seule perdrait l'information de phase

**Parametres STFT** :
- Sample rate : 16 000 Hz
- Taille FFT : n_fft = 510 (F = 256 bins frequentiels)
- Hop length : 128 (recouvrement ~75%)
- Fenetre : Hann periodique
- Resultat : spectrogramme complexe de dimension C^(K x F) ou K = nombre de trames, F = 256

### Transformation d'amplitude

Pour compenser la distribution a queue lourde des amplitudes STFT, une transformation non lineaire est appliquee :

```
c_transforme = beta * |c|^alpha * e^(j * angle(c))
```

Avec :
- **alpha = 0.5** : exposant de compression (racine carree de la magnitude)
- **beta = 0.15** : facteur d'echelle pour normaliser dans [0, 1]

**Pourquoi ?**
- Les fricatives et sons non voises ont une energie faible mais sont importants pour l'intelligibilite
- La compression alpha=0.5 amplifie les composantes faibles relativement aux fortes
- Le reseau opere sur des valeurs plus homogenes (meilleur pour l'optimisation)

**Transformation inverse** (pour revenir a l'audio) :
```
c_original = (c_transforme / beta) ^ (1/alpha) * e^(j * angle(c_transforme/beta))
```

---

## 4. Processus stochastique (SDE)

### Qu'est-ce qu'une SDE ?

Une SDE (Stochastic Differential Equation = Equation Differentielle Stochastique) decrit l'evolution d'un systeme soumis a la fois a une force deterministe et a du bruit aleatoire.

```
dx_t = f(x_t, y) dt + g(t) dw
```

- **x_t** : etat du processus au temps t (ici, le spectrogramme en cours de transformation)
- **f(x_t, y)** : coefficient de derive (drift) - la force deterministe
- **g(t)** : coefficient de diffusion - l'intensite du bruit ajoute
- **dw** : processus de Wiener (mouvement brownien = bruit blanc continu)

### Forward SDE (processus direct) : du propre vers le bruite

La SDE forward de SGMSE+ est une SDE lineaire dont le drift pousse le signal propre x_0 vers le signal bruite y :

```
dx_t = gamma * (y - x_t) dt + g(t) dw
```

**f(x_t, y) = gamma * (y - x_t)** : processus d'Ornstein-Uhlenbeck
- gamma (stiffness) : constante qui controle la vitesse de convergence vers y
- Quand x_t est loin de y, la force de rappel est forte
- Quand x_t est proche de y, la force de rappel est faible
- C'est un processus de **retour a la moyenne** (mean-reverting) vers y

**g(t)** : coefficient de diffusion (Variance Exploding) :
```
g(t) = sigma_min * (sigma_max/sigma_min)^t * sqrt(2 * log(sigma_max/sigma_min))
```
- sigma_min = 0.05, sigma_max = 0.5
- La variance du bruit augmente exponentiellement avec t
- Schema "Variance Exploding" (VE) : a la fin, le bruit domine

**Parametres** :
- gamma = 1.5 (stiffness)
- sigma_min = 0.05
- sigma_max = 0.5
- t dans [t_eps, T] avec t_eps = 0.03 (evite les instabilites numeriques a t=0)

### Solution analytique (noyau de perturbation)

Grace a la linearite de la SDE, on peut echantillonner directement x_t a n'importe quel temps t sans simuler toutes les etapes intermediaires :

```
p(x_t | x_0, y) = N_C(x_t ; mu(x_0, y, t), sigma(t)^2 * I)
```

Ou la **moyenne** decroit exponentiellement de x_0 vers y :
```
mu(x_0, y, t) = e^(-gamma*t) * x_0 + (1 - e^(-gamma*t)) * y
```

Et la **variance** :
```
sigma(t)^2 = sigma_min^2 * ((sigma_max/sigma_min)^(2t) - e^(-2*gamma*t)) * log(sigma_max/sigma_min)
             / (gamma + log(sigma_max/sigma_min))
```

**Interpretation** :
- A t=0 : mu = x_0 (signal propre), sigma = 0 -> on est sur le signal propre
- A t=T : mu ≈ y (signal bruite), sigma = grand -> on est autour du signal bruite + bruit gaussien
- La moyenne interpole exponentiellement entre propre et bruite
- La variance augmente, rendant l'echantillon de plus en plus "flou"

### Reverse SDE (processus inverse) : du bruite vers le propre

La SDE reverse est obtenue par la formule d'Anderson (1982) :

```
dx_t = [f(x_t, y) - g(t)^2 * gradient_x log p_t(x_t | y)] dt + g(t) dw_bar
```

Le terme cle est le **score** : `gradient_x log p_t(x_t | y)` = gradient de la log-densite de probabilite.

En remplacant le score exact (inconnu) par le score approxime par le reseau s_theta :

```
dx_t = [f(x_t, y) - g(t)^2 * s_theta(x_t, y, t)] dt + g(t) dw_bar
```

C'est la **plug-in reverse SDE** : on "branche" le reseau de neurones dans l'equation.

**Condition initiale** : on echantillonne x_T depuis une distribution fortement corrompue :
```
x_T ~ N_C(x_T ; y, sigma(T)^2 * I)
```
C'est le signal bruite y + du bruit gaussien supplementaire.

### Illustration du processus (Figure 2 du paper)

```
Temps:    0 ---------> T            T ---------> 0
          |  Forward SDE  |          |  Reverse SDE  |

Moyenne:  x_0 ~~~~> y (exponentiellement)    y ~~~~> x_0 (iterativement)
Variance: 0 ~~~~> sigma(T)^2                 sigma(T)^2 ~~~~> 0
Etat:     propre ~~~~> bruite+bruit          bruite+bruit ~~~~> propre
```

Le mismatch entre p_T (distribution forward a t=T) et p_tilde_T (condition initiale du reverse) est petit si gamma et sigma_max sont bien choisis. Le paper montre (Fig. 2 droite) que gamma=1.5 offre un bon compromis : le SNR entre mu et y tombe a ~15 dB a t=T/2.

---

## 5. Objectif d'entrainement (Score Matching)

### Le score : qu'est-ce que c'est ?

Le **score** est le gradient de la log-densite de probabilite par rapport aux donnees :

```
score(x_t) = gradient_x log p_t(x_t | y)
```

Intuitivement : pour chaque point x_t de l'espace des spectrogrammes, le score pointe dans la direction ou la probabilite augmente le plus = vers les spectrogrammes "vraisemblables" (propres).

### Denoising Score Matching

Grace a la forme analytique du noyau de perturbation (Gaussien), on peut calculer le score exact :

```
score_exact(x_t) = - (x_t - mu(x_0, y, t)) / sigma(t)^2
```

Ce qui est equivalent a :
```
score_exact(x_t) = - z / sigma(t)
```

ou z est le bruit gaussien qui a ete ajoute pour obtenir x_t depuis mu.

### Objectif d'entrainement

Le reseau s_theta apprend a approximer le score. La loss est :

```
L(theta) = E_t,x_0,y,z [ || s_theta(x_t, y, t) + z/sigma(t) ||^2 ]
```

Ou :
1. On echantillonne t uniformement dans [t_eps, T]
2. On prend une paire (x_0, y) du dataset (propre, bruite)
3. On echantillonne z ~ N(0, I)
4. On calcule x_t = mu(x_0, y, t) + sigma(t) * z
5. On passe (x_t, y, t) dans le reseau
6. On compare la sortie du reseau avec -z/sigma(t)

**Ponderation par sigma^2** : la loss est ponderee par sigma(t)^2 pour equilibrer les contributions des differents niveaux de bruit.

### Ce que le reseau apprend vraiment

Le reseau n'apprend PAS directement a estimer le bruit environnemental (comme dans un modele discriminatif). Il apprend la **direction vers les donnees propres** depuis n'importe quel point perturbe de l'espace. La loss ne contient pas y directement — le reseau ne voit jamais le bruit environnemental explicitement. Il apprend la structure de la parole propre.

### Procedure d'entrainement (etape par etape)

```
Pour chaque batch :
  1. Prendre un batch de paires (x_0, y) = (propre, bruite)
  2. STFT : convertir en spectrogrammes complexes
  3. Transformer : appliquer compression (alpha=0.5, beta=0.15)
  4. Echantillonner t ~ Uniforme[t_eps, T]
  5. Calculer mu et sigma depuis la SDE
  6. Echantillonner z ~ N_C(0, I)
  7. Calculer x_t = mu + sigma * z
  8. Passer (x_t, y, t) dans le reseau -> s_theta
  9. Loss = || s_theta + z/sigma ||^2 (pondere par sigma^2)
  10. Backward + optimizer step
  11. Mise a jour EMA (Exponential Moving Average) des poids
```

### Hyperparametres d'entrainement (paper original)

| Parametre | Valeur |
|-----------|--------|
| Optimizer | Adam |
| Learning rate | 1e-4 |
| Batch size effectif | 4 x 8 = 32 (4 GPU) |
| Epochs | 160 |
| EMA decay | 0.999 |
| Trames par exemple | K = 256 (decoupe aleatoire) |
| GPU | 4x NVIDIA RTX 6000 (24 GB) |
| Duree entrainement | ~1 jour |

---

## 6. Architecture du reseau (NCSN++)

### Vue d'ensemble

Le score model s_theta est un **NCSN++** (Noise Conditional Score Network ++), un U-Net profond multi-resolution adapte aux spectrogrammes complexes.

### Entree du reseau

Le reseau recoit en entree la concatenation de :
- **x_t** : spectrogramme perturbe (parties reelle et imaginaire = 2 canaux)
- **y** : spectrogramme bruite (parties reelle et imaginaire = 2 canaux)
- Total : **4 canaux** d'entree, dimension spatiale (F, K) = (256, K)

### Structure U-Net multi-resolution

```
Entree [x_t, y] (4 canaux, 256x256)
  |
  v
Conv2D 3x3 -> 128 canaux (256x256)
  |
  DownLayer (2 ResBlocks) -> 128 canaux (128x128)
  |
  DownLayer (2 ResBlocks) -> 128 canaux (64x64)
  |
  DownLayer (2 ResBlocks) -> 256 canaux (32x32)
  |
  DownLayer (2 ResBlocks) -> 256 canaux (16x16)    <-- Attention globale ici
  |
  DownLayer (2 ResBlocks) -> 256 canaux (8x8)
  |
  DownLayer (2 ResBlocks) -> 256 canaux (4x4)
  |
  BottleneckLayer (ResBlocks) -> 256 canaux (4x4)
  |
  UpLayer (3 ResBlocks + skip) -> 256 canaux (8x8)
  |
  UpLayer (3 ResBlocks + skip) -> 256 canaux (16x16)  <-- Attention globale ici
  |
  UpLayer (3 ResBlocks + skip) -> 256 canaux (32x32)
  |
  UpLayer (3 ResBlocks + skip) -> 128 canaux (64x64)
  |
  UpLayer (3 ResBlocks + skip) -> 128 canaux (128x128)
  |
  UpLayer (3 ResBlocks + skip) -> 128 canaux (256x256)
  |
  Conv2D -> 2 canaux (256x256)    = score (real + imag)
```

### Blocs residuels (ResBlock)

Chaque bloc residuel contient :
```
x -> GroupNorm -> Swish -> Conv2D 3x3 -> GroupNorm -> Swish -> Conv2D 3x3 -> + x
                                   ^
                                   |
                              t_emb (embedding du temps)
```

- **GroupNorm** : normalisation par groupes (plus stable que BatchNorm)
- **Swish** : activation f(x) = x * sigmoid(x) (lisse, pas de "dead neurons")
- **FIR up/down** : filtres a reponse impulsionnelle finie pour le sur/sous-echantillonnage (evite l'aliasing)

### Embedding du temps (Fourier)

Le temps t est encode via des **Fourier embeddings** :
```
t -> [sin(2*pi*f_1*t), cos(2*pi*f_1*t), ..., sin(2*pi*f_M*t), cos(2*pi*f_M*t)]
  -> couche lineaire -> vecteur t_emb de dimension M
```

Ce vecteur t_emb est injecte dans **chaque bloc residuel**, ce qui permet au reseau d'adapter son comportement selon le niveau de bruit (qui depend de t).

### Progressive Growing

En plus du chemin principal (encoder-decoder), le reseau a un **chemin de croissance progressive** :
- L'entree est sous-echantillonnee a chaque resolution et ajoutee aux feature maps du chemin contractant
- Symetriquement, les feature maps du chemin expansif sont sur-echantillonnees et contribuent a la sortie finale

Cela stabilise l'entrainement en haute resolution (technique empruntee a StyleGAN).

### Attention globale

Des mecanismes d'**attention globale** sont ajoutes a la resolution 16x16 (dans l'encoder et le decoder). Cela permet au reseau de capturer des dependances a longue portee dans le spectrogramme (par exemple, des harmoniques distantes ou des motifs temporels recurrents).

### Nombre de parametres

~**65.6 millions** de parametres entrainables (backbone DNN).

---

## 7. Inference : echantillonnage (Predictor-Corrector)

### Principe

A l'inference, on resout la reverse SDE numeriquement en N=30 pas discrets. L'algorithme alterne entre :
1. **Predictor** : avance d'un pas dans le temps inverse (solveur de SDE)
2. **Corrector** : raffine l'estimation avec quelques pas de dynamique de Langevin

### Algorithme Predictor-Corrector (PC)

```
ENTREE : signal bruite y (spectrogramme STFT transforme)

1. Echantillonner x_T ~ N_C(y, sigma(T)^2 * I)
2. Pour n = N, N-1, ..., 1 :
     t_n = n * T / N
     t_{n-1} = (n-1) * T / N

     // --- PREDICTOR (Reverse Diffusion) ---
     score = s_theta(x_n, y, t_n)        // appel au reseau
     drift = f(x_n, y) - g(t_n)^2 * score
     diffusion = g(t_n)
     z ~ N_C(0, I)
     dt = t_{n-1} - t_n                  // negatif (on remonte le temps)
     x_{n-1} = x_n + drift * dt + diffusion * sqrt(|dt|) * z

     // --- CORRECTOR (Annealed Langevin Dynamics) ---
     Pour i = 1, ..., r_steps :       // r_steps = 1 en pratique
       score = s_theta(x_{n-1}, y, t_{n-1})
       z ~ N_C(0, I)
       step_size = 2 * (snr * ||z|| / ||score||)^2
       x_{n-1} = x_{n-1} + step_size * score + sqrt(2 * step_size) * z

3. SORTIE : x_0 (spectrogramme propre estime)
```

### Parametres du sampler

| Parametre | Valeur | Role |
|-----------|--------|------|
| N | 30 | Nombre de pas de diffusion |
| corrector_steps | 1 | Pas de Langevin par etape |
| snr | 0.5 | Rapport signal/bruit du correcteur |
| sampler_type | "pc" | Predictor-corrector |
| predictor | "reverse_diffusion" | Type de predicteur |
| corrector | "ald" | Annealed Langevin Dynamics |

### Alternative : solveur ODE

Au lieu de la SDE stochastique, on peut resoudre l'ODE deterministe associee :
```
dx_t = [f(x_t, y) - 0.5 * g(t)^2 * s_theta(x_t, y, t)] dt
```

- **Plus rapide** : 14 evaluations au lieu de 30 (methode Runge-Kutta d'ordre 5)
- **Deterministe** : meme entree = meme sortie
- **Moins bon** : PESQ 2.78 vs 2.93 (PC sampler)
- RTF (Real-Time Factor) : 0.46 (ODE) vs 1.77 (PC)

### Pipeline d'inference complet

```
Audio bruite WAV (16 kHz)
  |
  v
STFT (n_fft=510, hop=128)
  |
  v
Spectrogramme complexe (B, F, T)
  |
  v
Transformation d'amplitude (alpha=0.5, beta=0.15)
  |
  v
Ajout dimension canal -> (B, 1, F, T)
  |
  v
Padding pour U-Net (T doit etre multiple de 64)
  |
  v
30 pas Predictor-Corrector (appels au score model)
  |
  v
Trim du padding
  |
  v
Transformation inverse (1/alpha, 1/beta)
  |
  v
iSTFT
  |
  v
Audio propre WAV (16 kHz)
```

### Normalisation

Avant le traitement, le signal est normalise par sa valeur absolue maximale. Apres le traitement, cette normalisation est restauree. Cela permet au reseau de toujours travailler sur des signaux de meme echelle.

---

## 8. Resultats experimentaux

### Datasets utilises

| Dataset | Tache | Contenu |
|---------|-------|---------|
| **VoiceBank-DEMAND (VB-DMD)** | Debruitage | Parole + bruit reel (DEMAND) a SNR 0-15 dB |
| **WSJ0-CHiME3** | Debruitage | Wall Street Journal + bruit CHiME3 a SNR 0-20 dB |
| **WSJ0-REVERB** | Dereverberation | WSJ0 + reverberation simulee (T60 0.4-1.0s, DRR -9 dB) |

### Metriques

| Metrique | Mesure | Echelle | Type |
|----------|--------|---------|------|
| POLQA | Qualite perceptuelle (successeur PESQ) | 1-5 | Intrusif |
| PESQ | Qualite perceptuelle | 1-4.5 | Intrusif |
| ESTOI | Intelligibilite | 0-1 | Intrusif |
| SI-SDR | Rapport signal/distorsion | dB | Intrusif |
| SI-SIR | Rapport signal/interference | dB | Intrusif |
| SI-SAR | Rapport signal/artefacts | dB | Intrusif |
| DNSMOS | Qualite perceptuelle (sans reference) | 1-5 | Non intrusif |

(Intrusif = necessite le signal propre de reference. Non intrusif = n'en a pas besoin.)

### Resultats Debruitage - WSJ0-CHiME3 (condition matched = meme dataset train/test)

| Methode | Type | POLQA | PESQ | ESTOI | SI-SDR |
|---------|------|-------|------|-------|--------|
| Melange bruite | - | 2.63 | 1.70 | 0.78 | 10.0 dB |
| SGMSE (original) | G | 2.98 | 2.48 | 0.86 | 14.8 dB |
| **SGMSE+** | **G** | **3.73** | **2.96** | **0.92** | **18.3 dB** |
| MetricGAN+ | D | 3.52 | **3.03** | 0.88 | 10.5 dB |
| Conv-TasNet | D | 3.65 | 2.99 | **0.93** | **19.9 dB** |

### Resultats Debruitage - VoiceBank-DEMAND

| Methode | PESQ | PESQ_nb | ESTOI | SI-SDR | DNSMOS |
|---------|------|---------|-------|--------|--------|
| Melange | 1.97 | 2.88 | 0.79 | 8.4 | 3.09 |
| SGMSE | 2.28 | 3.22 | 0.80 | 16.2 | 3.46 |
| **SGMSE+** | **2.93** | **3.66** | **0.87** | **17.3** | **3.56** |
| MetricGAN+ | 3.13 | 3.63 | 0.83 | 8.5 | 3.37 |
| Conv-TasNet | 2.63 | 3.42 | 0.85 | **19.1** | 3.37 |

**Observation importante** : MetricGAN+ a le PESQ le plus haut mais le DNSMOS le plus bas. Le paper explique que MetricGAN+ a appris a "tricher" sur la metrique PESQ en concentrant l'energie dans les basses/moyennes frequences tout en attenuant les hautes frequences. Les tests d'ecoute MUSHRA confirment que SGMSE+ est prefere par les auditeurs humains.

### Resultats Dereverberation - WSJ0-REVERB

| Methode | Type | POLQA | PESQ | ESTOI | SI-SDR |
|---------|------|-------|------|-------|--------|
| Melange reverberant | - | 1.76 | 1.36 | 0.46 | -7.3 dB |
| SGMSE | G | 1.79 | 1.35 | 0.57 | -7.4 dB |
| **SGMSE+** | **G** | **3.24** | **2.66** | **0.84** | **1.6 dB** |
| Conv-TasNet | D | 2.41 | 1.84 | 0.73 | **1.6 dB** |

SGMSE+ excelle en dereverberation, surpassant tous les modeles discriminatifs.

### Generalisation cross-dataset (train VB-DMD, test WSJ0-CHiME3)

| Methode | Type | POLQA | PESQ | ESTOI |
|---------|------|-------|------|-------|
| **SGMSE+** | **G** | **3.43** | **2.48** | **0.90** |
| MetricGAN+ | D | 2.47 | 2.13 | 0.76 |
| Conv-TasNet | D | 3.13 | 2.40 | 0.88 |

**SGMSE+ generalise le mieux** en condition cross-dataset. Les modeles discriminatifs (MetricGAN+) se degradent fortement. C'est un argument fort pour notre cas d'usage (signaux radio = hors distribution).

### Test d'ecoute MUSHRA (10 participants)

Score median (0-100) sur 12 exemples WSJ0-CHiME3 :
- **SGMSE+ (matched)** : ~75 (meilleur)
- **SGMSE+ (mismatched)** : ~70 (robuste)
- Conv-TasNet (matched) : ~65
- Conv-TasNet (mismatched) : ~45 (forte degradation)
- MetricGAN+ (matched) : ~48 (malgre PESQ eleve !)
- MetricGAN+ (mismatched) : ~25 (tres mauvais)

### Evaluations sur enregistrements reels (DNS Challenge 2020)

| Methode | DNSMOS | SIG | BAK | OVRL | WVMOS |
|---------|--------|-----|-----|------|-------|
| Melange | 3.05 | 3.05 | 2.51 | 2.26 | 1.12 |
| **SGMSE+** | **3.64** | **3.42** | **3.82** | **3.04** | **2.54** |
| MetricGAN+ | 3.26 | 2.88 | 3.39 | 2.45 | 1.52 |

SGMSE+ surpasse tous les autres modeles sur les enregistrements reels bruites.

---

## 9. Forces et limites

### Forces

1. **Meilleure generalisation** : grace a l'approche generative, SGMSE+ se degrade moins que les modeles discriminatifs face a des bruits non vus
2. **Qualite perceptuelle** : prefere par les auditeurs humains (test MUSHRA)
3. **Polyvalent** : meme architecture pour debruitage ET dereverberation
4. **Pas de phase explicite** : travaille directement sur le complexe, pas besoin de reconstruire la phase separement
5. **Robuste en cross-dataset** : performances stables meme quand le bruit de test differe du bruit d'entrainement
6. **EMA** : la moyenne mobile exponentielle des poids stabilise la qualite du modele

### Limites

1. **Lent a l'inference** : 30 pas de diffusion = ~1.77x le temps reel (vs temps reel pour MetricGAN+)
2. **Artefacts de vocalisation** : a tres bas SNR, le modele peut generer des sons de parole la ou il n'y en a pas (hallucination)
3. **Artefacts de respiration** : le modele peut confondre des bruits de respiration avec de la parole
4. **Gourmand en VRAM** : ~65M parametres + spectrogrammes complexes en memoire
5. **Stochastique** : deux inferences sur le meme signal donnent des resultats legerement differents (sauf en mode ODE)

### Limites identifiees par les auteurs (Section VII)

> "We observe that the proposed method sometimes introduces vocalizing and breathing artifacts. We argue that these could be mitigated if some conditioning concerning speech activity and phoneme identity would be added to the score model."

Les auteurs reconnaissent le probleme d'hallucination et suggerent d'ajouter un conditionnement sur l'activite vocale (VAD) et l'identite phonemique.

---

## 10. Application a notre projet

### Pourquoi SGMSE+ pour les signaux radio cockpit ?

1. **Bruit in-band** : le bruit radio et la voix partagent la bande 300-3400 Hz. Le masquage spectral (MetricGAN+) ne peut pas les separer. SGMSE+ regenere la parole en s'appuyant sur sa structure temporelle.

2. **Hors distribution** : tous les modeles pre-entraines echouent sur nos signaux radio. SGMSE+ generalise mieux en cross-dataset (demontre dans le paper).

3. **Bruit non-stationnaire** : le bruit radio (crackling, dropout, interference) est non-stationnaire. Les modeles de masquage supposent un bruit relativement stationnaire.

4. **Qualite perceptuelle** : SGMSE+ est prefere par les auditeurs humains, ce qui est notre critere final (intelligibilite operationnelle).

### Notre configuration de fine-tuning

| Parametre | Valeur paper | Notre valeur | Raison |
|-----------|-------------|-------------|--------|
| GPU | 4x RTX 6000 (24GB) | 1x RX 6800 (16GB) | Contrainte materielle |
| Batch size | 32 | 2 | Limite VRAM |
| LR | 1e-4 | 1e-5 | Fine-tuning (pas from scratch) |
| Epochs | 160 | 30 | Fine-tuning (pas from scratch) |
| Dataset | VB-DMD (14k paires) | 3000+300 paires radio | Domaine specifique |
| Segment | 256 trames STFT | 2s (256 trames) | Equivalent |
| EMA decay | 0.999 | 0.999 | Identique |
| Grad clip | non specifie | 1.0 | Stabilite |

### Risques et mitigations

| Risque | Mitigation |
|--------|-----------|
| Hallucination vocale sur silence radio | Silence gate (bypass si RMS < seuil) |
| Sur-apprentissage (petit dataset) | Early stopping patience=10, LR bas |
| Artefacts HF | Les signaux radio n'ont pas de HF (bande 300-3400 Hz) |
| VRAM insuffisante | Batch size 2, segments 2s, gradient clipping |

---

## 11. Glossaire

| Terme | Definition |
|-------|-----------|
| **Score** | Gradient de la log-densite de probabilite : indique la direction vers les donnees les plus vraisemblables |
| **SDE** | Stochastic Differential Equation : equation differentielle avec un terme de bruit aleatoire |
| **Forward process** | Processus qui transforme progressivement le signal propre en bruit |
| **Reverse process** | Processus inverse qui reconstruit le signal propre a partir du bruit |
| **Ornstein-Uhlenbeck** | Processus stochastique avec retour a la moyenne (ici, vers le signal bruite) |
| **Variance Exploding (VE)** | Schema ou la variance du bruit augmente avec le temps |
| **OUVE** | Ornstein-Uhlenbeck Variance Exploding : combinaison des deux ci-dessus |
| **Predictor-Corrector** | Algorithme d'echantillonnage alternant prediction (solveur SDE) et correction (Langevin) |
| **Langevin Dynamics** | Methode d'echantillonnage iterative qui suit le gradient (score) + bruit |
| **EMA** | Exponential Moving Average : moyenne ponderee des poids du reseau sur l'historique d'entrainement |
| **STFT** | Short-Time Fourier Transform : decomposition temps-frequence du signal |
| **iSTFT** | Inverse STFT : reconstruction du signal temporel depuis le spectrogramme |
| **NCSNpp / NCSN++** | Noise Conditional Score Network ++ : architecture U-Net pour estimer le score |
| **Score Matching** | Technique d'entrainement qui minimise l'erreur entre le score predit et le score exact |
| **Denoising Score Matching** | Variante qui utilise des donnees bruitees pour entrainer le score |
| **RTF** | Real-Time Factor : temps de traitement / duree audio (1.0 = temps reel) |
| **PESQ** | Perceptual Evaluation of Speech Quality (norme ITU-T P.862) |
| **POLQA** | Perceptual Objective Listening Quality Analysis (successeur de PESQ, ITU-T P.863) |
| **ESTOI** | Extended Short-Time Objective Intelligibility |
| **SI-SDR** | Scale-Invariant Signal-to-Distortion Ratio |
| **DNSMOS** | Deep Noise Suppression Mean Opinion Score (metrique non intrusive) |
| **MUSHRA** | MUlti Stimulus test with Hidden Reference and Anchor (test d'ecoute norme) |

---

## 12. References

1. **SGMSE+ (ce paper)** : Richter, Welker, Lemercier, Lay, Gerkmann. "Speech Enhancement and Dereverberation with Diffusion-based Generative Models." IEEE/ACM TASLP, 2023. [arXiv:2208.05830](https://arxiv.org/abs/2208.05830)

2. **SGMSE (original)** : Welker, Richter, Gerkmann. "Speech enhancement with score-based generative models in the complex STFT domain." ISCA Interspeech, 2022. [arXiv:2203.17004](https://arxiv.org/abs/2203.17004)

3. **Framework SDE** : Song, Sohl-Dickstein, Kingma, Kumar, Ermon, Poole. "Score-based generative modeling through stochastic differential equations." ICLR, 2021. [arXiv:2011.13456](https://arxiv.org/abs/2011.13456)

4. **Blog Yang Song** : [yang-song.net/blog/2021/score](https://yang-song.net/blog/2021/score/) - Explication accessible du score matching et des SDE

5. **Code officiel** : [github.com/sp-uhh/sgmse](https://github.com/sp-uhh/sgmse)

6. **Modele SpeechBrain** : [huggingface.co/speechbrain/sgmse-voicebank](https://huggingface.co/speechbrain/sgmse-voicebank)

7. **Page projet** : [uhh.de/inf-sp-sgmse](https://uhh.de/inf-sp-sgmse) - Exemples audio et liens

---

*Documentation generee le 25/03/2026 dans le cadre du projet Audio Super-Resolution (stage Maxime D'Angelo).*
*Basee sur le paper arXiv:2208.05830v3 (version du 13/10/2025).*
