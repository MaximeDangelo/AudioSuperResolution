"""
Generateur de dataset hybride pour fine-tuning DENOISING audio radio.

Deux sources de donnees :
1. Paires REELLES ATC : fichiers originaux radio + versions nettoyees
   depuis Dataset Radio (2)/ (vraie voix ATC/militaire)
2. Paires SYNTHETIQUES : LibriSpeech clean + degradations radio calibrees
   (pour augmenter le volume de donnees)

Strategie DENOISING :
- Le clean ET le raw subissent le meme bandpass + downsample (channel sim)
- Seul le raw recoit les bruits additifs (blanc, rose, crackling, cockpit...)
- Le modele apprend UNIQUEMENT a retirer le bruit, pas a reconstruire les
  frequences manquantes (pas de super-resolution)

Ratio cible : ~40% reel ATC / ~60% synthetique LibriSpeech
Toutes les paires sont resamplees a 16 kHz.

Structure de sortie :
    dataset/
        train/
            clean/  -> WAV 16kHz mono (bandpasse, sans bruit)
            raw/    -> WAV 16kHz mono (bandpasse + bruit)
        val/
            clean/
            raw/
        metadata.csv
"""
import os
import sys
import csv
import json
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from scipy.signal import butter, sosfilt, resample_poly
from math import gcd
from tqdm import tqdm

sys.stdout.reconfigure(encoding="utf-8")

# === Configuration ===

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")

# Profil de degradation calibre (genere par analyze_radio.py)
PROFILE_PATH = os.path.join(SCRIPT_DIR, "output", "analysis", "degradation_profile.json")

# Dossier des vraies paires ATC
ATC_DIR = os.path.join(SCRIPT_DIR, "Dataset Radio (2)")

# Paires ATC reelles : DESACTIVEES
# Analyse a montre que les fichiers "Clean" ne sont pas de vraies references
# propres (correlation ~0.12, SNR ~-2.6 dB). Les inclure pollue l'entrainement.
# Quand de vraies paires clean seront disponibles, les remettre ici.
ATC_PAIRS = []

# Sample rate cible (16 kHz = standard denoising, suffisant pour bande radio < 4 kHz)
TARGET_SR = 16000

# Passer les fichiers degrades dans Demucs avant sauvegarde
# Le SpectralResUNet apprend ainsi a ameliorer la sortie de Demucs (pas le signal brut)
APPLY_DEMUCS_TO_RAW = False

# Nombre d'echantillons synthetiques (LibriSpeech)
# Pour test rapide : 500/50. Pour dataset complet : 3000/300
N_SYNTH_TRAIN = 3000
N_SYNTH_VAL = 300

# Duree des segments decoupes (en secondes)
SEGMENT_MIN_S = 2.0
SEGMENT_MAX_S = 8.0

# Duree min/max pour LibriSpeech
MIN_DURATION_S = 2.0
MAX_DURATION_S = 10.0

# Seed
SEED = 42

# Parametres de degradation calibres sur donnees documentees cockpit chasse
# Sources : NATO RTO-EN-HFM-111, Elie et al. (Interspeech 2021), HIWIRE,
#           STANAG 4591 (MELPe), specs radio militaire UHF/VHF
# IMPORTANT : bandpass + downsample sont appliques au clean ET au raw (channel sim)
# Seules les degradations de BRUIT sont appliquees uniquement au raw
DEGRAD_PARAMS = {
    # --- Channel simulation (appliquee au clean ET au raw) ---
    # VHF AM airband : 300-3400 Hz / UHF militaire : 300-3000 Hz
    # 8.33 kHz channel spacing (Europe) : 350-2500 Hz
    "bandpass_low_range": (280, 400),
    "bandpass_high_range": (2500, 3400),  # mix UHF etroit + VHF standard
    "downsample_rates": [4000, 6000, 8000],  # MELPe = 8 kHz natif
    # --- Degradations bruit (appliquees UNIQUEMENT au raw) ---
    # IMPORTANT : tous les bruits sont filtres dans la bande du canal (channel_bw)
    "max_noise_degradations": 4,  # cockpit chasse = cumul de bruits
    # Bruit blanc electronique du recepteur radio
    "white_noise_prob": 0.8,
    "white_noise_snr_range": (10, 30),  # HIWIRE : SNR 5-10 dB pour cockpit
    # Souffle radio basse frequence (1/f) - pente -3 a -6 dB/octave (documente)
    "pink_noise_prob": 0.7,
    "pink_noise_snr_range": (10, 30),
    # Craquements radio (interferences, commutation, frequency hopping HAVE QUICK)
    "crackling_prob": 0.4,  # plus frequent en UHF militaire
    "crackling_density_range": (0.0003, 0.0012),
    "crackling_amplitude_range": (0.005, 0.025),
    # Interference tonale (dans la bande radio)
    "hf_interference_prob": 0.2,
    "hf_interference_amplitude_range": (0.003, 0.012),
    # Clipping/saturation emetteur - hard limiter militaire (attack 2ms)
    "clipping_prob": 0.3,
    "clipping_threshold_range": (0.5, 0.85),
    # Dropouts (pertes paquets, masquage terrain, manoeuvres G, ECM)
    "dropout_prob": 0.35,
    "dropout_count_range": (1, 5),
    "dropout_duration_ms_range": (20, 200),
    # Bruit cockpit chasse (95-110 dBA - NATO RTO-EN-HFM-111)
    # SNR au micro masque O2 : 5-20 dB (HIWIRE, Lightspeed Aviation)
    # Operationnel : 8-15 dB (seuil intelligibilite 80-90%)
    "cockpit_noise_prob": 0.9,  # quasi-systematique (moteur toujours en marche)
    "cockpit_noise_snr_range": (5, 20),  # documente : SNR masque O2 = 5-20 dB
    # Reverberation cockpit — tres faible : micro DANS le masque O2
    "cockpit_reverb_prob": 0.05,
    # AGC radio militaire (specs : 40 dB range, attack 2 ms, 0.2 dB output variation)
    "agc_prob": 0.6,  # plus frequent : AGC toujours actif sur radio militaire
    "agc_attack_ms_range": (1, 3),  # documente : ~2 ms
    "agc_release_ms_range": (30, 80),
    "agc_max_gain_range": (30, 40),  # documente : 40 dB range
    # --- Simulation masque O2 (Elie, Gauvain, Lamel, Interspeech 2021) ---
    # Attenuation HF > 3 kHz : -7 a -10 dB
    # Attenuation LF < 3 kHz : -0.7 a -2.8 dB (faible)
    # Perturbation formants F1 : +/-15-26%
    "o2_mask_prob": 0.85,  # quasi-systematique en vol chasse
    "o2_mask_hf_attenuation_db_range": (7, 10),  # au-dessus de 3 kHz
    "o2_mask_lf_attenuation_db_range": (0.5, 3.0),  # en-dessous de 3 kHz
    "o2_mask_formant_shift_range": (0.85, 1.15),  # facteur multiplicatif F1
    # --- Rafales de tirs (canon/mitrailleuse) ---
    # M61 Vulcan (F-16/F-18) : 100 coups/s, 150+ dB
    # Autocannon 30mm (A-10, Rafale) : 40-65 coups/s
    # Tres intense, court, repetitif - sature le micro/radio
    "gunfire_prob": 0.50,
    "gunfire_bursts_range": (1, 4),        # nombre de rafales par fichier
    "gunfire_burst_dur_ms_range": (300, 1500),  # duree d'une rafale
    "gunfire_rate_range": (5, 100),        # 5/s (tir espace) a 100/s (Vulcan)
    "gunfire_amplitude_range": (8, 25),    # facteur multiplicatif RMS (TRES fort)
    # --- Autres bruits non stationnaires (contexte combat) ---
    # Explosions, afterburner, alertes, manoeuvres G
    "impulse_noise_prob": 0.50,
    "impulse_noise_count_range": (1, 3),
    "impulse_noise_amplitude_range": (3, 15),
    # --- Bruit residuel sur le clean (realisme canal radio) ---
    # Meme un signal "propre" transmis par radio a un plancher de bruit
    "clean_residual_noise_prob": 0.8,
    "clean_residual_noise_snr_range": (30, 45),
}


def load_calibrated_profile():
    """Charge le profil calibre si disponible et ajuste DEGRAD_PARAMS."""
    if not os.path.exists(PROFILE_PATH):
        print("  Pas de profil calibre trouve, utilisation des parametres par defaut.")
        print(f"  (Lancez analyze_radio.py pour calibrer sur vos fichiers radio)\n")
        return

    with open(PROFILE_PATH, "r", encoding="utf-8") as f:
        profile = json.load(f)

    print(f"  Profil calibre charge depuis {PROFILE_PATH}")

    bp = profile.get("bandpass", {})
    if "low_hz_range" in bp:
        DEGRAD_PARAMS["bandpass_low_range"] = tuple(bp["low_hz_range"])
    if "high_hz_range" in bp:
        DEGRAD_PARAMS["bandpass_high_range"] = tuple(bp["high_hz_range"])

    snr = profile.get("snr", {})
    if "min_db" in snr and "max_db" in snr:
        # Plancher a 10 dB pour le bruit blanc, 15 dB pour le rose
        # En dessous le cumul de bruits rend le signal trop degrade
        DEGRAD_PARAMS["white_noise_snr_range"] = (
            max(10, snr["min_db"]),
            snr["max_db"] + 5,
        )
        DEGRAD_PARAMS["pink_noise_snr_range"] = (
            max(15, snr["min_db"] + 5),
            snr["max_db"] + 10,
        )

    clip = profile.get("clipping", {})
    if clip.get("mean_ratio", 0) > 0.001:
        DEGRAD_PARAMS["clipping_prob"] = 0.6
        DEGRAD_PARAMS["clipping_threshold_range"] = (0.3, 0.7)
    elif clip.get("max_ratio", 0) == 0:
        DEGRAD_PARAMS["clipping_prob"] = 0.05
        DEGRAD_PARAMS["clipping_threshold_range"] = (0.8, 0.95)
    elif clip.get("mean_ratio", 0) < 0.0001:
        DEGRAD_PARAMS["clipping_prob"] = 0.15
        DEGRAD_PARAMS["clipping_threshold_range"] = (0.6, 0.9)

    cutoff = profile.get("channel_cutoff_hz", 4000)
    DEGRAD_PARAMS["downsample_rates"] = [
        max(4000, cutoff - 2000),  # plancher a 4 kHz (2 kHz detruit trop le signal)
        max(4000, cutoff),
        cutoff + 2000,
    ]

    print(f"  Bandpass: {DEGRAD_PARAMS['bandpass_low_range']} - {DEGRAD_PARAMS['bandpass_high_range']} Hz")
    print(f"  SNR blanc: {DEGRAD_PARAMS['white_noise_snr_range']} dB")
    print(f"  Clipping prob: {DEGRAD_PARAMS['clipping_prob']}")
    print(f"  Downsample rates: {DEGRAD_PARAMS['downsample_rates']} Hz\n")


# === Degradations radio ===

def bandpass_filter(data, sr, low_hz=300, high_hz=3400, order=4):
    """Filtre passe-bande simulant la bande passante radio AM."""
    nyquist = sr / 2
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)
    sos = butter(order, [low, high], btype='band', output='sos')
    return sosfilt(sos, data).astype(np.float32)


def add_white_noise(data, snr_db=15, channel_bw=None, sr=16000):
    """Ajoute du bruit blanc gaussien a un SNR donne.

    Si channel_bw=(low_hz, high_hz) est fourni, le bruit est filtre
    dans la meme bande passante que le canal radio (realiste : le bruit
    electronique du recepteur est en bande etroite).
    """
    rms_signal = np.sqrt(np.mean(data ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.randn(len(data)).astype(np.float32)
    if channel_bw is not None:
        noise = bandpass_filter(noise, sr, low_hz=channel_bw[0], high_hz=channel_bw[1])
    noise = noise / (np.sqrt(np.mean(noise ** 2)) + 1e-10) * rms_noise
    return data + noise


def add_pink_noise(data, snr_db=20, channel_bw=None, sr=16000):
    """Ajoute du bruit rose (1/f) simulant le souffle radio.

    Filtre dans la bande du canal si channel_bw est fourni.
    """
    n = len(data)
    white = np.random.randn(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1
    pink_filter = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(np.fft.rfft(white) * pink_filter, n=n).astype(np.float32)
    if channel_bw is not None:
        pink = bandpass_filter(pink, sr, low_hz=channel_bw[0], high_hz=channel_bw[1])
    rms_signal = np.sqrt(np.mean(data ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    pink = pink / (np.sqrt(np.mean(pink ** 2)) + 1e-10) * rms_noise
    return data + pink


def add_crackling(data, sr, density=0.001, amplitude=0.05):
    """Ajoute des craquements aleatoires (interference radio)."""
    result = data.copy()
    n_crackles = int(len(data) * density)
    positions = np.random.randint(0, len(data), n_crackles)
    for pos in positions:
        duration = np.random.randint(int(sr * 0.001), int(sr * 0.003) + 1)
        end = min(pos + duration, len(data))
        impulse = np.random.randn(end - pos).astype(np.float32) * amplitude
        result[pos:end] += impulse
    return result


def add_hf_interference(data, sr, freq_hz=None, amplitude=0.02, channel_bw=None):
    """Ajoute une interference HF (tonalite parasite).

    La frequence est contrainte a rester dans la bande du canal radio.
    """
    max_freq = 3400 if channel_bw is None else channel_bw[1]
    min_freq = 800 if channel_bw is None else max(800, channel_bw[0])
    if freq_hz is None:
        freq_hz = np.random.uniform(min_freq, max_freq)
    freq_hz = min(freq_hz, max_freq)
    t = np.arange(len(data)) / sr
    interference = np.sin(2 * np.pi * freq_hz * t).astype(np.float32) * amplitude
    return data + interference


def apply_clipping(data, threshold=0.7):
    """Simule la distorsion par saturation (clipping radio)."""
    return np.clip(data, -threshold, threshold)


def apply_agc(data, sr, attack_ms=5, release_ms=50, target_rms=0.2, max_gain=30, rng=None):
    """Simule un AGC (Automatic Gain Control) radio militaire.

    Les emetteurs radio militaires compriment fortement la dynamique vocale
    avant transmission pour maximiser l'intelligibilite dans le bruit.
    Effet : dynamique ecrasee, pumping sur les transitoires, souffle amplifie
    dans les silences.

    Parametres :
    - attack_ms : temps de reaction a un pic (rapide = plus de pumping)
    - release_ms : temps de relachement apres un pic
    - target_rms : niveau RMS cible (normalisation)
    - max_gain : gain max en dB (limite l'amplification du bruit de fond)
    """
    if rng is None:
        rng = np.random.default_rng()

    # Taille des fenetres en samples
    attack_samples = max(1, int(attack_ms * sr / 1000))
    release_samples = max(1, int(release_ms * sr / 1000))

    # Enveloppe RMS glissante (fenetrage court)
    frame_len = int(sr * 0.01)  # 10 ms
    hop = frame_len // 2
    n_frames = (len(data) - frame_len) // hop + 1

    if n_frames <= 0:
        return data

    # Calculer l'enveloppe RMS par trame
    rms_env = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        start = i * hop
        frame = data[start:start + frame_len]
        rms_env[i] = np.sqrt(np.mean(frame ** 2)) + 1e-10

    # Gain desire par trame
    max_gain_linear = 10 ** (max_gain / 20)
    gain_env = np.clip(target_rms / rms_env, 1.0 / max_gain_linear, max_gain_linear)

    # Lissage attack/release (simule le circuit analogique)
    smoothed_gain = np.zeros_like(gain_env)
    smoothed_gain[0] = gain_env[0]
    alpha_attack = 1.0 - np.exp(-2.2 / (attack_samples / hop + 1e-10))
    alpha_release = 1.0 - np.exp(-2.2 / (release_samples / hop + 1e-10))

    for i in range(1, len(gain_env)):
        if gain_env[i] < smoothed_gain[i - 1]:
            # Signal monte -> reduire le gain (attack rapide)
            alpha = alpha_attack
        else:
            # Signal descend -> relacher le gain (release lent)
            alpha = alpha_release
        smoothed_gain[i] = smoothed_gain[i - 1] + alpha * (gain_env[i] - smoothed_gain[i - 1])

    # Interpoler le gain sur chaque sample
    frame_centers = np.arange(n_frames) * hop + frame_len // 2
    sample_indices = np.arange(len(data))
    gain_per_sample = np.interp(sample_indices, frame_centers, smoothed_gain).astype(np.float32)

    # Appliquer le gain
    result = data * gain_per_sample

    # Normaliser pour eviter le clipping
    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)

    return result


def apply_dropout(data, sr, n_dropouts=3, duration_ms_range=(20, 200), rng=None):
    """Simule des pertes de paquets radio (segments mis a zero).

    En radio militaire, les pertes sont frequentes : interference,
    manoeuvres, masquage terrain, saturation canal.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = data.copy()
    for _ in range(n_dropouts):
        dur_ms = rng.uniform(*duration_ms_range)
        dur_samples = int(dur_ms * sr / 1000)
        start = rng.integers(0, max(1, len(data) - dur_samples))
        end = min(start + dur_samples, len(data))
        # Fade-out/fade-in court pour eviter les clics
        fade_len = min(int(sr * 0.002), dur_samples // 4)
        result[start:end] = 0.0
        if fade_len > 0 and start >= fade_len:
            fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            result[start - fade_len:start] *= fade_out
        if fade_len > 0 and end + fade_len <= len(data):
            fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
            result[end:end + fade_len] *= fade_in
    return result


def add_cockpit_noise(data, sr, snr_db=15, rng=None, channel_bw=None):
    """Simule le bruit cockpit non stationnaire d'un avion de chasse.

    Calibre sur donnees documentees (NATO RTO-EN-HFM-111) :
    - Bruit cockpit chasse : 95-110 dB(A)
    - Spectre : brownien, pente -3 a -6 dB/octave au-dessus de 500 Hz
    - Pics basse frequence : 63-250 Hz (vibrations structure + moteur)
    - Plateau large bande : 200-4000 Hz (combustion + aerodynamique + turbine)
    - Roll-off au-dessus de 4 kHz

    Composantes :
    1. Bruit moteur/turbine (combustion : 200-2000 Hz)
    2. Vibrations structurelles (< 200 Hz, pics tonaux)
    3. Bruit aerodynamique (> 600 Hz, dominant en croisiere)
    4. Bruit turbine haute frequence (2-5 kHz, tonal + large bande)
    5. Modulation non stationnaire (regime moteur, manoeuvres G)
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(data)
    t = np.arange(n) / sr
    nyquist = sr / 2

    # 1. Bruit moteur/combustion : large bande 200-2000 Hz avec pente brownienne
    white = rng.standard_normal(n).astype(np.float32)
    # Appliquer une pente spectrale -3 a -6 dB/octave (brownien)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    freqs[0] = 1  # eviter division par zero
    slope_db_per_octave = rng.uniform(-6, -3)  # documente
    slope_factor = slope_db_per_octave / 20 * np.log2(freqs / 500 + 1e-10)
    spectral_shape = np.power(10, slope_factor)
    # Booster la bande 200-2000 Hz (combustion chamber)
    combustion_boost = np.exp(-0.5 * ((freqs - 800) / 600) ** 2) * 2.0 + 1.0
    engine_noise = np.fft.irfft(spectrum * spectral_shape * combustion_boost, n=n).astype(np.float32)

    # 2. Vibrations structurelles : pics tonaux sous 200 Hz
    base_freq = rng.uniform(50, 150)
    vibrations = np.zeros(n, dtype=np.float32)
    for harmonic in range(1, 5):
        amp = rng.uniform(0.2, 0.5) / harmonic
        phase = rng.uniform(0, 2 * np.pi)
        vibrations += (amp * np.sin(2 * np.pi * base_freq * harmonic * t + phase)).astype(np.float32)

    # 3. Bruit aerodynamique : dominant > 600 Hz, large bande
    white2 = rng.standard_normal(n).astype(np.float32)
    aero_cutoff = rng.uniform(500, 800) / nyquist
    sos_aero = butter(3, min(aero_cutoff, 0.99), btype='high', output='sos')
    aero_noise = sosfilt(sos_aero, white2).astype(np.float32) * 0.4

    # 4. Bruit turbine HF : 2-5 kHz (tonal + large bande)
    turbine_freq = rng.uniform(2000, 4000)
    turbine_tonal = (0.15 * np.sin(2 * np.pi * turbine_freq * t)).astype(np.float32)
    white3 = rng.standard_normal(n).astype(np.float32)
    if 2000 / nyquist < 0.99 and 5000 / nyquist < 0.99:
        sos_turb = butter(3, [2000 / nyquist, min(5000 / nyquist, 0.99)], btype='band', output='sos')
        turbine_broadband = sosfilt(sos_turb, white3).astype(np.float32) * 0.2
    else:
        turbine_broadband = np.zeros(n, dtype=np.float32)

    # Combiner toutes les composantes
    cockpit = engine_noise * 0.4 + vibrations * 0.2 + aero_noise * 0.25 + \
              turbine_tonal * 0.1 + turbine_broadband * 0.05

    # 4. Enveloppe non stationnaire (simule variations regime moteur)
    env_freq = rng.uniform(0.1, 0.5)
    envelope = (0.7 + 0.3 * np.sin(2 * np.pi * env_freq * t)).astype(np.float32)
    cockpit = cockpit * envelope

    # 5. Filtrer dans la bande du canal radio (le bruit passe par le meme micro/canal)
    if channel_bw is not None:
        cockpit = bandpass_filter(cockpit, sr, low_hz=channel_bw[0], high_hz=channel_bw[1])

    # Ajuster le SNR
    rms_signal = np.sqrt(np.mean(data ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    cockpit = cockpit / (np.sqrt(np.mean(cockpit ** 2)) + 1e-10) * rms_noise

    return data + cockpit


def apply_cockpit_reverb(data, sr, rng=None):
    """Simule la reverberation d'un cockpit d'avion de chasse.

    Cockpit de chasse = espace tres confine (~2-3 m3), surfaces dures
    (verriere, tableau de bord, metal). Caracteristiques :
    - RT60 tres court : 30-120 ms
    - Early reflections fortes et rapprochees (< 10 ms)
    - Absorption HF par le casque/combinaison du pilote
    - Coloration metallique (resonances a certaines frequences)

    Genere une reponse impulsionnelle (RIR) synthetique puis convolue.
    """
    if rng is None:
        rng = np.random.default_rng()

    # RT60 aleatoire typique cockpit chasse (30-120 ms)
    rt60 = rng.uniform(0.03, 0.12)
    rir_len = int(rt60 * sr)

    # 1. Early reflections (surfaces proches : verriere, tableau de bord)
    n_early = rng.integers(4, 10)
    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0  # impulsion directe

    for i in range(n_early):
        # Reflexions entre 1 et 8 ms (cockpit tres petit)
        delay_s = rng.uniform(0.001, 0.008)
        delay_samples = int(delay_s * sr)
        if delay_samples < rir_len:
            # Amplitude decroissante avec le delai, signe aleatoire
            amp = rng.uniform(0.2, 0.6) * (1.0 - delay_s / 0.01)
            amp = max(amp, 0.05)
            if rng.random() > 0.5:
                amp = -amp
            rir[delay_samples] += amp

    # 2. Queue diffuse (decroissance exponentielle)
    t_rir = np.arange(rir_len) / sr
    decay = np.exp(-6.9 * t_rir / rt60).astype(np.float32)  # -60 dB a RT60
    diffuse = rng.standard_normal(rir_len).astype(np.float32) * decay * 0.15
    # Pas de diffuse dans les premieres 2 ms (zone early reflections)
    diffuse[:int(0.002 * sr)] = 0
    rir += diffuse

    # 3. Absorption HF (casque, combinaison, mousse)
    nyquist = sr / 2
    hf_cutoff = rng.uniform(4000, 8000) / nyquist
    sos_abs = butter(2, min(hf_cutoff, 0.99), btype='low', output='sos')
    rir = sosfilt(sos_abs, rir).astype(np.float32)

    # Normaliser la RIR (energie unitaire pour ne pas changer le volume)
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-10)
    rir[0] = max(rir[0], 0.8)  # garder le direct dominant
    rir = rir / (np.sqrt(np.sum(rir ** 2)) + 1e-10)

    # 4. Convolution
    from scipy.signal import fftconvolve
    result = fftconvolve(data, rir, mode='full')[:len(data)].astype(np.float32)

    # Dry/wet mix : cockpit = beaucoup de direct, peu de reverb
    wet_ratio = rng.uniform(0.05, 0.20)  # micro dans masque O2 = tres peu de reverb
    result = (1.0 - wet_ratio) * data + wet_ratio * result

    # Normaliser pour eviter le clipping
    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)

    return result


def apply_o2_mask(data, sr, rng, params=None):
    """Simule l'effet acoustique du masque a oxygene militaire.

    Basé sur Elie, Gauvain, Lamel (Interspeech 2021) :
    - Attenuation HF (> 3 kHz) : -7 a -10 dB (masque filtre les aigus)
    - Attenuation LF (< 3 kHz) : -0.7 a -2.8 dB (faible)
    - Perturbation formants : F1 decale de +/-15-26% (cavite du masque)
    Le masque agit comme un filtre passe-bas + resonateur qui modifie
    la reponse frequentielle et les formants de la voix.
    """
    if params is None:
        params = DEGRAD_PARAMS

    n = len(data)
    # Travailler dans le domaine frequentiel
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # 1. Attenuation HF au-dessus de 3 kHz
    hf_atten_db = rng.uniform(*params["o2_mask_hf_attenuation_db_range"])
    hf_mask = np.ones_like(freqs)
    hf_zone = freqs > 3000
    # Transition progressive entre 2500 et 3500 Hz
    transition = (freqs >= 2500) & (freqs <= 3500)
    hf_mask[transition] = 1.0 - (freqs[transition] - 2500) / 1000 * (1.0 - 10 ** (-hf_atten_db / 20))
    hf_mask[hf_zone] = 10 ** (-hf_atten_db / 20)

    # 2. Attenuation LF legere en-dessous de 3 kHz
    lf_atten_db = rng.uniform(*params["o2_mask_lf_attenuation_db_range"])
    lf_mask = np.ones_like(freqs)
    lf_zone = freqs < 3000
    lf_mask[lf_zone] = 10 ** (-lf_atten_db / 20)

    # Combiner les deux masques
    combined_mask = hf_mask * lf_mask
    spectrum = spectrum * combined_mask

    # Note : la perturbation formants (Elie et al.) est retiree car
    # l'interpolation spectrale creait des artefacts d'echo.
    # L'attenuation HF/LF suffit a simuler l'effet du masque O2.

    result = np.fft.irfft(spectrum, n=n).astype(np.float32)

    # Normaliser pour garder le meme niveau
    rms_in = np.sqrt(np.mean(data ** 2)) + 1e-10
    rms_out = np.sqrt(np.mean(result ** 2)) + 1e-10
    result = result * (rms_in / rms_out)

    return result


def add_gunfire(data, sr, rng, params=None, channel_bw=None):
    """Ajoute des rafales de tirs (canon/mitrailleuse) au signal.

    Simule le bruit capte par le micro du masque O2 lors de tirs :
    - M61 Vulcan (F-16, F-18) : 100 coups/s, 6 canons rotatifs
    - DEFA 30mm (Rafale) : 2500 coups/min (~42/s)
    - GAU-8 Avenger (A-10) : 65 coups/s

    Chaque coup est une impulsion tres courte (2-5ms) avec attaque instantanee
    et decroissance rapide. La repetition rapide cree un bruit de type "buzz"
    ou "brrrrt" tres caracteristique. Le son sature le micro et le canal radio.
    """
    if params is None:
        params = DEGRAD_PARAMS

    result = data.copy()
    rms_signal = np.sqrt(np.mean(data ** 2)) + 1e-10
    n_bursts = rng.integers(*params["gunfire_bursts_range"])

    for _ in range(n_bursts):
        # Position aleatoire de la rafale
        pos = rng.integers(0, max(1, len(data) - int(sr * 0.1)))
        burst_dur_ms = rng.uniform(*params["gunfire_burst_dur_ms_range"])
        burst_samples = int(burst_dur_ms * sr / 1000)
        rate = rng.uniform(*params["gunfire_rate_range"])
        amp_factor = rng.uniform(*params["gunfire_amplitude_range"])

        burst = np.zeros(burst_samples, dtype=np.float32)
        interval = int(sr / rate)  # samples entre deux coups

        # Generer chaque coup de la rafale
        shot_pos = 0
        while shot_pos < burst_samples:
            # Un coup = impulsion courte (2-5ms), forte attaque, decay exponentiel
            shot_dur = int(sr * rng.uniform(0.002, 0.005))
            shot_end = min(shot_pos + shot_dur, burst_samples)
            actual_dur = shot_end - shot_pos

            if actual_dur > 0:
                t_shot = np.arange(actual_dur) / sr
                # Impulsion : bruit blanc module par enveloppe exponentielle
                shot = rng.standard_normal(actual_dur).astype(np.float32)
                decay_rate = rng.uniform(400, 800)  # decay rapide
                envelope = np.exp(-t_shot * decay_rate).astype(np.float32)
                # Pic initial tres fort
                envelope[0] = 1.0
                shot = shot * envelope

                burst[shot_pos:shot_end] += shot

            # Intervalle entre coups (avec leger jitter pour realisme)
            jitter = rng.integers(-max(1, interval // 6), max(2, interval // 6 + 1))
            shot_pos += interval + jitter

        # Normaliser la rafale par rapport au signal
        burst_rms = np.sqrt(np.mean(burst ** 2)) + 1e-10
        burst = burst / burst_rms * rms_signal * amp_factor

        # Filtrer dans la bande du canal (le son passe par le micro + radio)
        if channel_bw is not None and len(burst) > 50:
            burst = bandpass_filter(burst, sr, low_hz=channel_bw[0], high_hz=channel_bw[1])
            # Renormaliser apres filtrage
            filt_rms = np.sqrt(np.mean(burst ** 2)) + 1e-10
            burst = burst / filt_rms * rms_signal * amp_factor

        # Saturation du micro (les tirs saturent systematiquement)
        clip_level = rng.uniform(0.3, 0.7)
        peak_burst = np.max(np.abs(burst))
        if peak_burst > 0:
            burst_norm = burst / peak_burst
            burst = np.clip(burst_norm, -clip_level, clip_level) * peak_burst

        # Inserer dans le signal
        end = min(pos + len(burst), len(data))
        actual_len = end - pos
        if actual_len > 0:
            result[pos:end] += burst[:actual_len]

    # Eviter le clipping global
    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)

    return result


def add_impulse_noise(data, sr, rng, params=None, channel_bw=None):
    """Ajoute des bruits non stationnaires forts (explosions, afterburner, alertes).

    En contexte combat, le signal est perturbe par des evenements intenses :
    - Rafale de canon (M61 Vulcan : 100 coups/s, 150+ dB)
    - Explosions proches (missile, DCA)
    - Afterburner / acceleration moteur (spool-up sur 1-3s)
    - Alertes cockpit "bitching betty" (tonalites repetees)
    - Manoeuvres G (bruit aerodynamique transitoire)

    Ces bruits sont transmis par le micro du masque O2 et passent par le canal radio.
    Ils sont donc filtres dans la bande du canal.
    """
    if params is None:
        params = DEGRAD_PARAMS

    result = data.copy()
    n_events = rng.integers(*params["impulse_noise_count_range"])
    rms_signal = np.sqrt(np.mean(data ** 2)) + 1e-10

    for _ in range(n_events):
        pos = rng.integers(0, len(data))
        # Amplitude relative au signal (ces bruits sont FORTS : souvent plus forts que la voix)
        amp_factor = rng.uniform(*params["impulse_noise_amplitude_range"])
        amplitude = rms_signal * amp_factor

        event_type = rng.choice(["explosion", "afterburner", "alert", "g_maneuver"])

        if event_type == "explosion":
            # Explosion : 200-1000 ms, attaque rapide, decroissance lente, basse freq
            dur_ms = rng.uniform(200, 1000)
            dur_samples = int(dur_ms * sr / 1000)
            impulse = rng.standard_normal(dur_samples).astype(np.float32)
            # Filtre passe-bas dominant (explosions = BF)
            nyquist = sr / 2
            cutoff = rng.uniform(400, 1200) / nyquist
            if cutoff < 0.99:
                sos = butter(3, cutoff, btype='low', output='sos')
                impulse = sosfilt(sos, impulse).astype(np.float32)
            t = np.arange(dur_samples) / sr
            # Attaque rapide (10ms) + decroissance lente
            attack = np.minimum(t / 0.01, 1.0)
            decay = np.exp(-t * rng.uniform(3, 10))
            impulse = impulse * attack * decay * amplitude * 5

        elif event_type == "afterburner":
            # Acceleration moteur / afterburner : spool-up progressif sur 0.5-2s
            dur_s = rng.uniform(0.5, 2.0)
            dur_samples = int(dur_s * sr)
            # Bruit large bande avec montee progressive en amplitude
            noise = rng.standard_normal(dur_samples).astype(np.float32)
            # Enveloppe : montee progressive (spool-up)
            t = np.arange(dur_samples) / sr
            envelope = (t / dur_s) ** rng.uniform(1.5, 3.0)  # montee convexe
            # Ajouter des harmoniques basses (vibration turbine qui monte)
            base_freq_start = rng.uniform(80, 200)
            base_freq_end = base_freq_start * rng.uniform(1.5, 3.0)
            freq_sweep = np.linspace(base_freq_start, base_freq_end, dur_samples)
            phase = np.cumsum(2 * np.pi * freq_sweep / sr)
            tonal = np.sin(phase).astype(np.float32) * 0.4
            impulse = (noise * 0.6 + tonal) * envelope * amplitude * 4

        elif event_type == "alert":
            # Alerte cockpit "bitching betty" : tonalite repetee, 0.5-2s
            dur_s = rng.uniform(0.5, 2.0)
            dur_samples = int(dur_s * sr)
            t = np.arange(dur_samples) / sr
            freq = rng.uniform(500, 2000)
            # Tonalite pulsee (on/off a 2-5 Hz)
            pulse_rate = rng.uniform(2, 5)
            pulse_env = (0.5 + 0.5 * np.sign(np.sin(2 * np.pi * pulse_rate * t))).astype(np.float32)
            impulse = (np.sin(2 * np.pi * freq * t) * pulse_env * amplitude * 2).astype(np.float32)
            # Fade in/out
            fade = min(int(sr * 0.01), dur_samples // 4)
            if fade > 0:
                impulse[:fade] *= np.linspace(0, 1, fade)
                impulse[-fade:] *= np.linspace(1, 0, fade)

        else:  # g_maneuver
            # Manoeuvre G : montee breve du bruit aerodynamique, 0.3-1.5s
            dur_s = rng.uniform(0.3, 1.5)
            dur_samples = int(dur_s * sr)
            noise = rng.standard_normal(dur_samples).astype(np.float32)
            # Filtre passe-haut (bruit aero = HF)
            nyquist = sr / 2
            cutoff = rng.uniform(600, 1500) / nyquist
            if cutoff < 0.99:
                sos = butter(3, cutoff, btype='high', output='sos')
                noise = sosfilt(sos, noise).astype(np.float32)
            # Enveloppe en cloche (monte puis redescend)
            t = np.arange(dur_samples) / sr
            center = dur_s / 2
            sigma = dur_s / 4
            envelope = np.exp(-0.5 * ((t - center) / sigma) ** 2)
            impulse = noise * envelope * amplitude * 3

        # Filtrer dans la bande du canal radio
        if channel_bw is not None and len(impulse) > 50:
            impulse = bandpass_filter(impulse, sr, low_hz=channel_bw[0], high_hz=channel_bw[1])
            # Renormaliser apres filtrage (le bandpass attenue beaucoup)
            imp_rms = np.sqrt(np.mean(impulse ** 2)) + 1e-10
            if imp_rms > 1e-8:
                impulse = impulse / imp_rms * amplitude * 3

        # Inserer dans le signal
        end = min(pos + len(impulse), len(data))
        actual_len = end - pos
        if actual_len > 0:
            result[pos:end] += impulse[:actual_len]

    # Eviter le clipping
    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)

    return result


def downsample_upsample(data, sr, low_sr=8000):
    """Simule la perte de qualite par downsampling puis upsampling."""
    g = gcd(sr, low_sr)
    down = resample_poly(data, low_sr // g, sr // g).astype(np.float32)
    up = resample_poly(down, sr // g, low_sr // g).astype(np.float32)
    if len(up) > len(data):
        up = up[:len(data)]
    elif len(up) < len(data):
        up = np.pad(up, (0, len(data) - len(up)))
    return up


def apply_channel_simulation(data, sr, rng, params=None):
    """Applique la simulation du canal radio (bandpass + downsample).

    Cette fonction est appliquee au clean ET au raw pour que les deux
    aient la meme bande passante. Le modele n'a pas besoin d'apprendre
    a reconstruire les frequences filtrees.
    """
    if params is None:
        params = DEGRAD_PARAMS
    result = data.copy()

    low_hz = rng.uniform(*params["bandpass_low_range"])
    high_hz = rng.uniform(*params["bandpass_high_range"])
    result = bandpass_filter(result, sr, low_hz=low_hz, high_hz=high_hz)

    low_sr = rng.choice(params["downsample_rates"])
    result = downsample_upsample(result, sr, low_sr=low_sr)

    return result.astype(np.float32), low_hz, high_hz, low_sr


def apply_noise_degradations(data, sr, rng, params=None, channel_bw=None):
    """Applique des degradations de bruit (UNIQUEMENT au raw).

    Tire aleatoirement quelles degradations appliquer, avec un maximum
    de max_noise_degradations pour eviter un cumul excessif.

    channel_bw : tuple (low_hz, high_hz) de la bande du canal radio.
    Tous les bruits sont filtres dans cette bande car en realite le bruit
    cockpit est capte par le micro et transmis par le meme canal radio.
    """
    if params is None:
        params = DEGRAD_PARAMS
    result = data.copy()

    max_degs = params.get("max_noise_degradations", 4)

    # Liste des degradations possibles avec leur probabilite
    degradations = []

    if rng.random() < params["white_noise_prob"]:
        degradations.append("white_noise")
    if rng.random() < params["pink_noise_prob"]:
        degradations.append("pink_noise")
    if rng.random() < params["crackling_prob"]:
        degradations.append("crackling")
    if rng.random() < params["hf_interference_prob"]:
        degradations.append("hf_interference")
    if rng.random() < params["clipping_prob"]:
        degradations.append("clipping")
    if rng.random() < params.get("cockpit_noise_prob", 0):
        degradations.append("cockpit_noise")
    if rng.random() < params.get("cockpit_reverb_prob", 0):
        degradations.append("cockpit_reverb")
    if rng.random() < params.get("agc_prob", 0):
        degradations.append("agc")
    if rng.random() < params.get("dropout_prob", 0):
        degradations.append("dropout")
    if rng.random() < params.get("impulse_noise_prob", 0):
        degradations.append("impulse_noise")
    if rng.random() < params.get("gunfire_prob", 0):
        degradations.append("gunfire")

    # Limiter le nombre de degradations (sauf gunfire qui est un evenement dominant)
    has_gunfire = "gunfire" in degradations
    if has_gunfire:
        degradations.remove("gunfire")
    if len(degradations) > max_degs:
        rng.shuffle(degradations)
        degradations = degradations[:max_degs]
    if has_gunfire:
        degradations.append("gunfire")  # toujours ajouter apres la limite

    # Appliquer les degradations selectionnees
    for deg in degradations:
        if deg == "white_noise":
            snr = rng.uniform(*params["white_noise_snr_range"])
            result = add_white_noise(result, snr_db=snr, channel_bw=channel_bw, sr=sr)
        elif deg == "pink_noise":
            snr = rng.uniform(*params["pink_noise_snr_range"])
            result = add_pink_noise(result, snr_db=snr, channel_bw=channel_bw, sr=sr)
        elif deg == "crackling":
            density = rng.uniform(*params["crackling_density_range"])
            amplitude = rng.uniform(*params["crackling_amplitude_range"])
            result = add_crackling(result, sr, density=density, amplitude=amplitude)
        elif deg == "hf_interference":
            amplitude = rng.uniform(*params["hf_interference_amplitude_range"])
            result = add_hf_interference(result, sr, amplitude=amplitude, channel_bw=channel_bw)
        elif deg == "clipping":
            threshold = rng.uniform(*params["clipping_threshold_range"])
            result = apply_clipping(result, threshold=threshold)
        elif deg == "cockpit_noise":
            snr = rng.uniform(*params["cockpit_noise_snr_range"])
            result = add_cockpit_noise(result, sr, snr_db=snr, rng=rng, channel_bw=channel_bw)
        elif deg == "cockpit_reverb":
            result = apply_cockpit_reverb(result, sr, rng=rng)
        elif deg == "agc":
            attack = rng.uniform(*params["agc_attack_ms_range"])
            release = rng.uniform(*params["agc_release_ms_range"])
            max_gain = rng.uniform(*params["agc_max_gain_range"])
            result = apply_agc(result, sr, attack_ms=attack, release_ms=release,
                               max_gain=max_gain, rng=rng)
        elif deg == "dropout":
            n_drops = rng.integers(*params["dropout_count_range"])
            result = apply_dropout(result, sr, n_dropouts=n_drops,
                                   duration_ms_range=params["dropout_duration_ms_range"],
                                   rng=rng)
        elif deg == "impulse_noise":
            result = add_impulse_noise(result, sr, rng=rng, params=params,
                                       channel_bw=channel_bw)
        elif deg == "gunfire":
            result = add_gunfire(result, sr, rng=rng, params=params,
                                 channel_bw=channel_bw)

    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)

    return result.astype(np.float32)


MIN_SNR_DB = 3.0  # SNR plancher pour eviter les paires inutilisables


def apply_radio_degradation(data, sr, rng, params=None):
    """Applique la chaine complete : channel sim + bruit.

    Retourne (raw, clean_channeled) : les deux ont le meme bandpass/downsample,
    mais seul raw a le bruit.
    """
    if params is None:
        params = DEGRAD_PARAMS

    # 0. Simulation masque O2 (appliquee a la voix AVANT le canal radio)
    # Le masque modifie la voix du pilote avant qu'elle n'entre dans le micro
    # Elie, Gauvain, Lamel (Interspeech 2021) : attenuation HF + shift formants
    if rng.random() < params.get("o2_mask_prob", 0):
        data = apply_o2_mask(data, sr, rng, params)

    # 1. Channel simulation (partagee clean/raw)
    channeled, low_hz, high_hz, low_sr = apply_channel_simulation(
        data, sr, rng, params
    )

    # 2. Le clean = signal bandpasse + bruit residuel tres faible (realisme)
    # Un signal radio "propre" a toujours un plancher de bruit du canal
    clean_channeled = channeled.copy()
    if rng.random() < params.get("clean_residual_noise_prob", 0):
        residual_snr = rng.uniform(*params["clean_residual_noise_snr_range"])
        residual = rng.standard_normal(len(clean_channeled)).astype(np.float32)
        residual = bandpass_filter(residual, sr, low_hz=low_hz, high_hz=high_hz)
        rms_sig = np.sqrt(np.mean(clean_channeled ** 2)) + 1e-10
        rms_target = rms_sig / (10 ** (residual_snr / 20))
        residual = residual / (np.sqrt(np.mean(residual ** 2)) + 1e-10) * rms_target
        clean_channeled = clean_channeled + residual

    # 3. Le raw = signal bandpasse + bruit (bruit filtre dans la meme bande)
    channel_bw = (low_hz, high_hz)
    raw = apply_noise_degradations(channeled, sr, rng, params, channel_bw=channel_bw)

    # 4. Garantir un SNR minimum pour que la paire reste exploitable
    diff = raw - clean_channeled
    rms_clean = np.sqrt(np.mean(clean_channeled ** 2)) + 1e-10
    rms_diff = np.sqrt(np.mean(diff ** 2)) + 1e-10
    snr = 20 * np.log10(rms_clean / rms_diff)
    if snr < MIN_SNR_DB:
        # Attenuer le bruit pour remonter au SNR plancher
        target_rms_diff = rms_clean / (10 ** (MIN_SNR_DB / 20))
        noise_only = raw - clean_channeled
        noise_only = noise_only * (target_rms_diff / rms_diff)
        raw = clean_channeled + noise_only
        peak = np.max(np.abs(raw))
        if peak > 0.95:
            raw = raw * (0.95 / peak)

    return raw, clean_channeled


# === Demucs pre-processing ===

def apply_demucs_batch(raw_files, sr):
    """Passe un batch de fichiers WAV dans Demucs pour debruitage.

    Utilise Demucs en mode batch : ecrit tous les fichiers dans un dossier temp,
    lance Demucs une seule fois, puis recupere les sorties.
    Retourne un dict {filepath: demucs_output_array}.
    """
    if not APPLY_DEMUCS_TO_RAW:
        return {}

    import shutil
    tmp_dir = os.path.join(tempfile.gettempdir(), "demucs_batch_in")
    out_dir = os.path.join(tempfile.gettempdir(), "demucs_batch_out")
    os.makedirs(tmp_dir, exist_ok=True)

    # Ecrire tous les fichiers dans le dossier temp
    for fpath in raw_files:
        shutil.copy2(fpath, tmp_dir)

    # Lancer Demucs sur tout le dossier
    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", out_dir,
    ] + [os.path.join(tmp_dir, os.path.basename(f)) for f in raw_files]

    subprocess.run(cmd, capture_output=True)

    # Recuperer les sorties
    results = {}
    for fpath in raw_files:
        base = os.path.splitext(os.path.basename(fpath))[0]
        vocals_path = os.path.join(out_dir, "htdemucs", base, "vocals.wav")
        if os.path.exists(vocals_path):
            data, _ = sf.read(vocals_path, dtype="float32")
            if data.ndim > 1:
                data = data.mean(axis=1)
            results[fpath] = data
        else:
            # Fallback : garder l'original
            data, _ = sf.read(fpath, dtype="float32")
            results[fpath] = data

    # Nettoyage
    shutil.rmtree(tmp_dir, ignore_errors=True)
    shutil.rmtree(out_dir, ignore_errors=True)

    return results


def apply_demucs_single(data, sr):
    """Passe un signal audio dans Demucs et retourne le resultat."""
    tmp_in = os.path.join(tempfile.gettempdir(), "demucs_single_in.wav")
    tmp_out_dir = os.path.join(tempfile.gettempdir(), "demucs_single_out")
    sf.write(tmp_in, data, sr)

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems", "vocals",
        "-n", "htdemucs",
        "-o", tmp_out_dir,
        tmp_in,
    ]
    result = subprocess.run(cmd, capture_output=True)

    vocals_path = os.path.join(tmp_out_dir, "htdemucs", "demucs_single_in", "vocals.wav")
    if os.path.exists(vocals_path):
        out_data, _ = sf.read(vocals_path, dtype="float32")
        if out_data.ndim > 1:
            out_data = out_data.mean(axis=1)
        # Aligner la longueur
        if len(out_data) > len(data):
            out_data = out_data[:len(data)]
        elif len(out_data) < len(data):
            out_data = np.pad(out_data, (0, len(data) - len(out_data)))
    else:
        out_data = data  # Fallback

    # Nettoyage
    import shutil
    if os.path.exists(tmp_in):
        os.remove(tmp_in)
    shutil.rmtree(tmp_out_dir, ignore_errors=True)

    return out_data.astype(np.float32)


# === Chargement audio ===

def load_audio_file(filepath, target_sr=TARGET_SR):
    """Charge un fichier audio, convertit en mono au SR cible via ffmpeg."""
    tmp_wav = os.path.join(tempfile.gettempdir(), "ds_load_tmp.wav")
    cmd = ["ffmpeg", "-y", "-i", filepath, "-ac", "1", "-ar", str(target_sr), tmp_wav]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        return None, 0
    data, sr = sf.read(tmp_wav, dtype="float32")
    os.remove(tmp_wav)
    return data, sr


def segment_pair(raw_data, clean_data, sr, rng, segment_min=SEGMENT_MIN_S, segment_max=SEGMENT_MAX_S):
    """Decoupe une paire (raw, clean) en segments alignes.

    Les deux signaux doivent avoir la meme longueur et le meme SR.
    Retourne une liste de tuples (raw_segment, clean_segment).
    """
    min_len = min(len(raw_data), len(clean_data))
    raw_data = raw_data[:min_len]
    clean_data = clean_data[:min_len]

    min_samples = int(segment_min * sr)
    max_samples = int(segment_max * sr)

    segments = []
    pos = 0
    while pos + min_samples <= min_len:
        # Duree aleatoire pour chaque segment
        seg_len = rng.integers(min_samples, min(max_samples, min_len - pos) + 1)
        raw_seg = raw_data[pos:pos + seg_len]
        clean_seg = clean_data[pos:pos + seg_len]

        # Verifier que le segment contient de l'energie (pas que du silence)
        rms = np.sqrt(np.mean(raw_seg ** 2))
        if rms > 0.005:
            segments.append((raw_seg, clean_seg))

        pos += seg_len

    return segments


# === Partie 1 : Paires ATC reelles ===

def generate_atc_pairs(rng):
    """Charge les vraies paires ATC et les segmente."""
    if not os.path.isdir(ATC_DIR):
        print(f"  ATTENTION: {ATC_DIR} introuvable, pas de paires ATC reelles.")
        return []

    all_segments = []

    for raw_name, clean_name in ATC_PAIRS:
        raw_path = os.path.join(ATC_DIR, raw_name)
        clean_path = os.path.join(ATC_DIR, clean_name)

        if not os.path.exists(raw_path) or not os.path.exists(clean_path):
            print(f"    SKIP: {raw_name} (fichier manquant)")
            continue

        print(f"    Chargement: {raw_name}")
        raw_data, sr_raw = load_audio_file(raw_path, TARGET_SR)
        clean_data, sr_clean = load_audio_file(clean_path, TARGET_SR)

        if raw_data is None or clean_data is None:
            print(f"    ERREUR: impossible de charger {raw_name}")
            continue

        # Pour 07 (Heathrow) : le clean est plus court car les silences ont ete coupes.
        # On ne peut aligner que sur la longueur du plus court.
        min_len = min(len(raw_data), len(clean_data))
        raw_data = raw_data[:min_len]
        clean_data = clean_data[:min_len]

        dur_raw = len(raw_data) / TARGET_SR
        dur_clean = len(clean_data) / TARGET_SR
        print(f"      Raw: {dur_raw:.1f}s | Clean: {dur_clean:.1f}s (aligne: {min_len/TARGET_SR:.1f}s)")

        # Decouper en segments
        segments = segment_pair(raw_data, clean_data, TARGET_SR, rng)
        print(f"      -> {len(segments)} segments extraits")
        all_segments.extend(segments)

    return all_segments


# === Partie 2 : Paires synthetiques LibriSpeech ===

def generate_synthetic_pairs(rng, n_train, n_val):
    """Genere des paires synthetiques depuis LibriSpeech (via torchaudio)."""
    import torchaudio

    # Telecharge uniquement train-clean-100 (~6 Go) dans ./librispeech_data/
    ls_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "librispeech_data")
    print(f"  Chargement de LibriSpeech train-clean-100 (torchaudio)...")
    dataset = torchaudio.datasets.LIBRISPEECH(root=ls_dir, url="train-clean-100", download=True)
    n_samples = len(dataset)
    print(f"  {n_samples} samples disponibles\n")

    indices = list(range(n_samples))
    rng.shuffle(indices)

    pairs = {"train": [], "val": []}
    counts = {"train": 0, "val": 0}
    targets = {"train": n_train, "val": n_val}
    current_split = "train"

    for idx in tqdm(indices, desc="  Paires synthetiques"):
        if counts["train"] >= n_train and counts["val"] >= n_val:
            break

        if counts[current_split] >= targets[current_split]:
            current_split = "val" if current_split == "train" else "train"
            if counts[current_split] >= targets[current_split]:
                break

        waveform, sr, _, _, _, _ = dataset[idx]
        data = waveform.squeeze(0).numpy()
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        duration = len(data) / sr
        if duration < MIN_DURATION_S or duration > MAX_DURATION_S:
            continue

        # Resample vers TARGET_SR
        if sr != TARGET_SR:
            gg = gcd(sr, TARGET_SR)
            data = resample_poly(data, TARGET_SR // gg, sr // gg).astype(np.float32)

        # Normaliser le clean a un RMS realiste (vrais fichiers radio : 0.01-0.04)
        # Avant : peak * 0.9 donnait RMS ~0.10 (4.8x trop fort vs reel)
        target_rms = rng.uniform(0.01, 0.04)
        current_rms = np.sqrt(np.mean(data ** 2)) + 1e-10
        data = data * (target_rms / current_rms)
        # Eviter le clipping
        peak = np.max(np.abs(data))
        if peak > 0.95:
            data = data * (0.95 / peak)

        # Generer la paire (raw degrade, clean bandpasse)
        # Les deux partagent le meme bandpass/downsample, seul raw a le bruit
        data_raw, data_clean = apply_radio_degradation(data, TARGET_SR, rng)

        pairs[current_split].append((data_raw, data_clean))
        counts[current_split] += 1

    print(f"  -> Train: {counts['train']} | Val: {counts['val']}")
    return pairs


# === Pipeline principal ===

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("  GENERATION DU DATASET DENOISING (ATC reel + LibriSpeech)")
    print(f"  Mode: denoising pur @ {TARGET_SR} Hz")
    print(f"  Clean = bandpasse (meme canal que raw, sans bruit)")
    print("=" * 60)

    # Charger le profil calibre
    load_calibrated_profile()

    # Creer les repertoires
    for split in ["train", "val"]:
        for sub in ["clean", "raw"]:
            os.makedirs(os.path.join(DATASET_DIR, split, sub), exist_ok=True)

    metadata = []

    # === Partie 1 : Paires ATC reelles ===
    print("\n--- Partie 1 : Paires ATC reelles ---")
    atc_segments = generate_atc_pairs(rng)
    rng.shuffle(atc_segments)

    # 90% train / 10% val pour les paires ATC
    n_atc_val = max(1, len(atc_segments) // 10)
    n_atc_train = len(atc_segments) - n_atc_val
    atc_train = atc_segments[:n_atc_train]
    atc_val = atc_segments[n_atc_train:]

    print(f"\n  ATC total: {len(atc_segments)} segments")
    print(f"  ATC train: {n_atc_train} | ATC val: {n_atc_val}")

    # Sauvegarder les paires ATC
    atc_counts = {"train": 0, "val": 0}
    for split, segments in [("train", atc_train), ("val", atc_val)]:
        for raw_seg, clean_seg in segments:
            file_id = f"atc_{split}_{atc_counts[split]:05d}"
            raw_path = os.path.join(DATASET_DIR, split, "raw", f"{file_id}.wav")
            clean_path = os.path.join(DATASET_DIR, split, "clean", f"{file_id}.wav")
            sf.write(raw_path, raw_seg, TARGET_SR)
            sf.write(clean_path, clean_seg, TARGET_SR)
            metadata.append({
                "id": file_id,
                "split": split,
                "source": "atc_real",
                "duration_s": round(len(raw_seg) / TARGET_SR, 2),
                "sr": TARGET_SR,
            })
            atc_counts[split] += 1

    # === Partie 2 : Paires synthetiques LibriSpeech ===
    print("\n--- Partie 2 : Paires synthetiques LibriSpeech ---")
    synth_pairs = generate_synthetic_pairs(rng, N_SYNTH_TRAIN, N_SYNTH_VAL)

    synth_counts = {"train": 0, "val": 0}
    for split in ["train", "val"]:
        for raw_seg, clean_seg in synth_pairs[split]:
            file_id = f"synth_{split}_{synth_counts[split]:05d}"
            raw_path = os.path.join(DATASET_DIR, split, "raw", f"{file_id}.wav")
            clean_path = os.path.join(DATASET_DIR, split, "clean", f"{file_id}.wav")
            sf.write(raw_path, raw_seg, TARGET_SR)
            sf.write(clean_path, clean_seg, TARGET_SR)
            metadata.append({
                "id": file_id,
                "split": split,
                "source": "librispeech_synth",
                "duration_s": round(len(raw_seg) / TARGET_SR, 2),
                "sr": TARGET_SR,
            })
            synth_counts[split] += 1

    # === Etape optionnelle : passer les raw dans Demucs ===
    if APPLY_DEMUCS_TO_RAW:
        print("\n--- Post-traitement : Demucs sur les fichiers raw ---")
        import glob as g
        for split in ["train", "val"]:
            raw_dir = os.path.join(DATASET_DIR, split, "raw")
            raw_files = sorted(g.glob(os.path.join(raw_dir, "*.wav")))
            print(f"  {split}: {len(raw_files)} fichiers a traiter avec Demucs...")
            for i, fpath in enumerate(tqdm(raw_files, desc=f"  Demucs {split}")):
                data, file_sr = sf.read(fpath, dtype="float32")
                demucs_data = apply_demucs_single(data, file_sr)
                sf.write(fpath, demucs_data, file_sr)  # Ecrase le raw avec la version Demucs

    # Sauvegarder metadata
    csv_path = os.path.join(DATASET_DIR, "metadata.csv")
    if metadata:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)

    # Resume
    total_train = atc_counts["train"] + synth_counts["train"]
    total_val = atc_counts["val"] + synth_counts["val"]
    atc_pct = atc_counts["train"] / total_train * 100 if total_train > 0 else 0

    print(f"\n{'=' * 60}")
    print(f"  DATASET GENERE")
    print(f"{'=' * 60}")
    print(f"  Train : {total_train} paires ({atc_counts['train']} ATC + {synth_counts['train']} synth)")
    print(f"  Val   : {total_val} paires ({atc_counts['val']} ATC + {synth_counts['val']} synth)")
    print(f"  Ratio ATC : {atc_pct:.0f}%")
    print(f"  Repertoire : {DATASET_DIR}")
    print(f"  Metadata : {csv_path}")


if __name__ == "__main__":
    main()
