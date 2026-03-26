"""
Generateur de dataset hybride pour fine-tuning super-resolution audio.

Deux sources de donnees :
1. Paires REELLES ATC : fichiers originaux radio + versions nettoyees
   depuis Dataset Radio (2)/ (vraie voix ATC/militaire)
2. Paires SYNTHETIQUES : LibriSpeech clean + degradations radio calibrees
   (pour augmenter le volume de donnees)

Ratio cible : ~40% reel ATC / ~60% synthetique LibriSpeech
Toutes les paires sont resamplees a 44.1 kHz.

Structure de sortie :
    dataset/
        train/
            clean/  -> WAV 44.1kHz mono
            raw/    -> WAV 44.1kHz mono (degradee)
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

# Paires ATC reelles : (fichier original, fichier clean)
# Seulement les fichiers qui ont une version clean
ATC_PAIRS = [
    ("01__kwahmah_atc001.flac", "01__kwahmah_atc001-Clean.wav"),
    ("02__kwahmah_atc002.flac", "02__kwahmah_atc002-Clean.wav"),
    ("03_kwahmah_atc003.flac", "03_kwahmah_atc003-Clean.wav"),
    ("04-voicebot_pilotes-pendant-un-vol.mp3", "04-voicebot_pilotes-pendant-un-vol-Clean.wav"),
    ("05__kwahmah_atc005.flac", "05__kwahmah_atc005-Clean.wav"),
    ("07-wahmah_heathrow-air-traffic-control.flac", "07-wahmah_heathrow-air-traffic-control-Clean.wav"),
]

# Sample rate cible
TARGET_SR = 44100

# Passer les fichiers degrades dans Demucs avant sauvegarde
# Le SpectralResUNet apprend ainsi a ameliorer la sortie de Demucs (pas le signal brut)
APPLY_DEMUCS_TO_RAW = True

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

# Parametres de degradation par defaut (ecrases par le profil si disponible)
DEGRAD_PARAMS = {
    "bandpass_low_range": (200, 400),
    "bandpass_high_range": (2800, 3800),
    "downsample_rates": [4000, 6000, 8000],
    "white_noise_prob": 0.8,
    "white_noise_snr_range": (5, 25),
    "pink_noise_prob": 0.7,
    "pink_noise_snr_range": (10, 30),
    "crackling_prob": 0.5,
    "crackling_density_range": (0.0005, 0.003),
    "crackling_amplitude_range": (0.02, 0.1),
    "hf_interference_prob": 0.3,
    "hf_interference_amplitude_range": (0.005, 0.03),
    "clipping_prob": 0.4,
    "clipping_threshold_range": (0.4, 0.8),
    # Dropouts (pertes de paquets radio)
    "dropout_prob": 0.5,
    "dropout_count_range": (1, 5),
    "dropout_duration_ms_range": (20, 200),
    # Bruit cockpit non stationnaire (moteur, turbine, flux d'air)
    "cockpit_noise_prob": 0.6,
    "cockpit_noise_snr_range": (5, 25),
    # Reverberation cockpit (espace confine, surfaces dures)
    "cockpit_reverb_prob": 0.5,
    # AGC radio militaire (compression dynamique emetteur)
    "agc_prob": 0.6,
    "agc_attack_ms_range": (2, 15),
    "agc_release_ms_range": (30, 100),
    "agc_max_gain_range": (15, 35),
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
        DEGRAD_PARAMS["white_noise_snr_range"] = (
            max(1, snr["min_db"] - 5),
            snr["max_db"] + 5,
        )
        DEGRAD_PARAMS["pink_noise_snr_range"] = (
            max(1, snr["min_db"]),
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
        max(2000, cutoff - 2000),
        cutoff,
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


def add_white_noise(data, snr_db=15):
    """Ajoute du bruit blanc gaussien a un SNR donne."""
    rms_signal = np.sqrt(np.mean(data ** 2)) + 1e-10
    rms_noise = rms_signal / (10 ** (snr_db / 20))
    noise = np.random.randn(len(data)).astype(np.float32) * rms_noise
    return data + noise


def add_pink_noise(data, snr_db=20):
    """Ajoute du bruit rose (1/f) simulant le souffle radio."""
    n = len(data)
    white = np.random.randn(n).astype(np.float32)
    freqs = np.fft.rfftfreq(n, d=1.0)
    freqs[0] = 1
    pink_filter = 1.0 / np.sqrt(freqs)
    pink = np.fft.irfft(np.fft.rfft(white) * pink_filter, n=n).astype(np.float32)
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


def add_hf_interference(data, sr, freq_hz=None, amplitude=0.02):
    """Ajoute une interference HF (tonalite parasite)."""
    if freq_hz is None:
        freq_hz = np.random.uniform(3000, 6000)
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


def add_cockpit_noise(data, sr, snr_db=15, rng=None):
    """Simule le bruit cockpit non stationnaire d'un avion de chasse.

    Combine :
    - Bruit moteur/turbine basse frequence (< 500 Hz), module en amplitude
    - Bruit de flux d'air large bande (souffle aerodynamique)
    - Vibrations mecaniques (harmoniques basses)
    Le tout module par une enveloppe lente pour simuler les variations
    de regime moteur et les manoeuvres.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(data)
    t = np.arange(n) / sr

    # 1. Bruit moteur/turbine : bruit filtre passe-bas (< 500 Hz)
    white = rng.standard_normal(n).astype(np.float32)
    nyquist = sr / 2
    engine_cutoff = rng.uniform(200, 500) / nyquist
    sos_engine = butter(4, min(engine_cutoff, 0.99), btype='low', output='sos')
    engine_noise = sosfilt(sos_engine, white).astype(np.float32)

    # 2. Vibrations mecaniques : quelques harmoniques basses
    base_freq = rng.uniform(50, 120)
    vibrations = np.zeros(n, dtype=np.float32)
    for harmonic in range(1, 4):
        amp = 0.3 / harmonic
        vibrations += (amp * np.sin(2 * np.pi * base_freq * harmonic * t)).astype(np.float32)

    # 3. Souffle aerodynamique : bruit large bande filtre passe-haut (> 1 kHz)
    white2 = rng.standard_normal(n).astype(np.float32)
    aero_cutoff = rng.uniform(800, 1500) / nyquist
    sos_aero = butter(3, min(aero_cutoff, 0.99), btype='high', output='sos')
    aero_noise = sosfilt(sos_aero, white2).astype(np.float32) * 0.3

    # Combiner les composantes
    cockpit = engine_noise + vibrations + aero_noise

    # 4. Enveloppe non stationnaire (simule variations regime moteur)
    env_freq = rng.uniform(0.1, 0.5)
    envelope = (0.7 + 0.3 * np.sin(2 * np.pi * env_freq * t)).astype(np.float32)
    cockpit = cockpit * envelope

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
    wet_ratio = rng.uniform(0.15, 0.45)
    result = (1.0 - wet_ratio) * data + wet_ratio * result

    # Normaliser pour eviter le clipping
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


def apply_radio_degradation(data, sr, rng, params=None):
    """Applique une combinaison aleatoire de degradations radio."""
    if params is None:
        params = DEGRAD_PARAMS
    result = data.copy()

    low_hz = rng.uniform(*params["bandpass_low_range"])
    high_hz = rng.uniform(*params["bandpass_high_range"])
    result = bandpass_filter(result, sr, low_hz=low_hz, high_hz=high_hz)

    low_sr = rng.choice(params["downsample_rates"])
    result = downsample_upsample(result, sr, low_sr=low_sr)

    if rng.random() < params["white_noise_prob"]:
        snr = rng.uniform(*params["white_noise_snr_range"])
        result = add_white_noise(result, snr_db=snr)

    if rng.random() < params["pink_noise_prob"]:
        snr = rng.uniform(*params["pink_noise_snr_range"])
        result = add_pink_noise(result, snr_db=snr)

    if rng.random() < params["crackling_prob"]:
        density = rng.uniform(*params["crackling_density_range"])
        amplitude = rng.uniform(*params["crackling_amplitude_range"])
        result = add_crackling(result, sr, density=density, amplitude=amplitude)

    if rng.random() < params["hf_interference_prob"]:
        amplitude = rng.uniform(*params["hf_interference_amplitude_range"])
        result = add_hf_interference(result, sr, amplitude=amplitude)

    if rng.random() < params["clipping_prob"]:
        threshold = rng.uniform(*params["clipping_threshold_range"])
        result = apply_clipping(result, threshold=threshold)

    # Bruit cockpit non stationnaire (moteur, turbine, flux d'air)
    if rng.random() < params.get("cockpit_noise_prob", 0):
        snr = rng.uniform(*params["cockpit_noise_snr_range"])
        result = add_cockpit_noise(result, sr, snr_db=snr, rng=rng)

    # Reverberation cockpit (espace confine avion de chasse)
    if rng.random() < params.get("cockpit_reverb_prob", 0):
        result = apply_cockpit_reverb(result, sr, rng=rng)

    # AGC radio militaire (compression dynamique avant transmission)
    if rng.random() < params.get("agc_prob", 0):
        attack = rng.uniform(*params["agc_attack_ms_range"])
        release = rng.uniform(*params["agc_release_ms_range"])
        max_gain = rng.uniform(*params["agc_max_gain_range"])
        result = apply_agc(result, sr, attack_ms=attack, release_ms=release,
                           max_gain=max_gain, rng=rng)

    # Dropouts / pertes de paquets radio
    if rng.random() < params.get("dropout_prob", 0):
        n_drops = rng.integers(*params["dropout_count_range"])
        result = apply_dropout(result, sr, n_dropouts=n_drops,
                               duration_ms_range=params["dropout_duration_ms_range"],
                               rng=rng)

    peak = np.max(np.abs(result))
    if peak > 0.95:
        result = result * (0.95 / peak)

    return result.astype(np.float32)


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

        # Normaliser le clean
        peak = np.max(np.abs(data))
        if peak > 0:
            data = data / peak * 0.9

        # Generer la version degradee
        data_raw = apply_radio_degradation(data, TARGET_SR, rng)

        pairs[current_split].append((data_raw, data))
        counts[current_split] += 1

    print(f"  -> Train: {counts['train']} | Val: {counts['val']}")
    return pairs


# === Pipeline principal ===

def main():
    rng = np.random.default_rng(SEED)

    print("=" * 60)
    print("  GENERATION DU DATASET HYBRIDE (ATC reel + LibriSpeech)")
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
