"""
Fine-tuning de SGMSE+ (SpeechBrain) pour le debruitage radio cockpit.

Charge le modele pre-entraine sgmse-voicebank et le fine-tune
sur les paires (raw, clean) du dataset radio synthetique.

SGMSE+ est un modele de diffusion qui regenere le signal propre
au lieu de masquer le bruit (contrairement a MetricGAN+).
Adapte au bruit in-band ou voix et bruit partagent la meme bande (300-3400 Hz).

Le modele travaille a 16 kHz dans le domaine STFT (spectrogramme complexe).
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "sgmse")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "sgmse-voicebank")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparametres
BATCH_SIZE = 2          # Max pour 16GB VRAM avec SGMSE+ (65M params)
LEARNING_RATE = 1e-5    # LR bas pour fine-tuning diffusion
EPOCHS = 30
SEGMENT_LENGTH_S = 2.0  # secondes
SR_MODEL = 16000        # SGMSE+ travaille a 16 kHz
SEGMENT_LENGTH = int(SEGMENT_LENGTH_S * SR_MODEL)
NUM_WORKERS = 0 if sys.platform == "win32" else 4
PATIENCE = 10
GRAD_CLIP = 1.0
RESUME_FROM = os.path.join(OUTPUT_DIR, "best_model.pt")  # None pour repartir de zero

# STFT params (doivent matcher le modele pre-entraine)
N_FFT = 510
HOP_LENGTH = 128


# Device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = get_device()


# === Dataset ===
class RadioEnhancementDataset(Dataset):
    """Dataset de paires (raw, clean) a 16 kHz pour SGMSE+."""

    def __init__(self, dataset_dir, split="train", segment_length=SEGMENT_LENGTH):
        self.segment_length = segment_length
        raw_dir = os.path.join(dataset_dir, split, "raw")
        clean_dir = os.path.join(dataset_dir, split, "clean")

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
        self.pairs = []
        for rf in raw_files:
            cf = os.path.join(clean_dir, os.path.basename(rf))
            if os.path.exists(cf):
                self.pairs.append((rf, cf))

        print(f"  {split}: {len(self.pairs)} paires chargees (16kHz)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, clean_path = self.pairs[idx]
        try:
            raw, _ = sf.read(raw_path, dtype="float32")
            clean, _ = sf.read(clean_path, dtype="float32")
        except Exception:
            # Fichier corrompu : retourner du silence
            raw = np.zeros(self.segment_length, dtype=np.float32)
            clean = np.zeros(self.segment_length, dtype=np.float32)

        if raw.ndim > 1:
            raw = np.mean(raw, axis=1)
        if clean.ndim > 1:
            clean = np.mean(clean, axis=1)

        # Aligner les longueurs
        mn = min(len(raw), len(clean))
        raw, clean = raw[:mn], clean[:mn]

        # Decouper un segment aleatoire
        if mn > self.segment_length:
            start = np.random.randint(0, mn - self.segment_length)
            raw = raw[start:start + self.segment_length]
            clean = clean[start:start + self.segment_length]
        else:
            pad = self.segment_length - mn
            raw = np.pad(raw, (0, pad))
            clean = np.pad(clean, (0, pad))

        return torch.tensor(raw), torch.tensor(clean)


# === STFT helpers (meme que le modele pre-entraine) ===
def get_window(device):
    return torch.hann_window(N_FFT, periodic=True).to(device)


def stft(sig, window):
    """(B, T) -> (B, F, L) complex"""
    return torch.stft(sig, n_fft=N_FFT, hop_length=HOP_LENGTH,
                      window=window, center=True, return_complex=True)


def istft(spec, window, length=None):
    """(B, F, L) complex -> (B, T)"""
    return torch.istft(spec, n_fft=N_FFT, hop_length=HOP_LENGTH,
                       window=window, center=True, length=length)


def spec_fwd(S, factor=0.15, exponent=0.5):
    """Forward spectral transform (magnitude^exponent * factor)."""
    if exponent != 1.0:
        mag = S.abs() ** exponent
        ph = S.angle()
        S = mag * torch.exp(1j * ph)
    return S * factor


def spec_back(S, factor=0.15, exponent=0.5):
    """Inverse spectral transform."""
    S = S / factor
    if exponent != 1.0:
        mag = S.abs() ** (1.0 / exponent)
        ph = S.angle()
        S = mag * torch.exp(1j * ph)
    return S


def pad_spec(spec):
    """Pad spectrogram pour les contraintes U-Net (multiple de 64 en temps).
    Travaille sur des tenseurs complexes (B, 1, F, T) via view_as_real."""
    T = spec.shape[-1]
    num_pad = (64 - T % 64) % 64
    if num_pad > 0:
        # ZeroPad2d ne supporte pas les complex tensors directement
        # On split real/imag, pad, puis recombine
        real = spec.real
        imag = spec.imag
        real = F.pad(real, (0, num_pad))
        imag = F.pad(imag, (0, num_pad))
        spec = torch.complex(real, imag)
    return spec


# === Charger le modele SGMSE+ ===
def load_sgmse_model():
    """Charge SGMSE+ depuis les poids pre-entraines."""
    from speechbrain.inference.enhancement import SGMSEEnhancement

    model = SGMSEEnhancement.from_hparams(
        source="speechbrain/sgmse-voicebank",
        savedir=PRETRAINED_DIR,
    )
    return model


# === Entrainement diffusion ===
def train_one_epoch(score_model, loader, optimizer, device, window):
    """Une epoch d'entrainement score matching."""
    score_model.train()
    total_loss = 0
    n_batches = 0

    for raw, clean in loader:
        raw, clean = raw.to(device), clean.to(device)

        # Normaliser par max abs
        raw_norm = torch.clamp(raw.abs().amax(dim=-1, keepdim=True), min=1e-8)
        clean_norm = torch.clamp(clean.abs().amax(dim=-1, keepdim=True), min=1e-8)
        raw_n = raw / raw_norm
        clean_n = clean / clean_norm

        # STFT -> spectrogramme complexe -> transform
        Y = spec_fwd(stft(raw_n, window)).unsqueeze(1)    # (B, 1, F, L) noisy
        X = spec_fwd(stft(clean_n, window)).unsqueeze(1)   # (B, 1, F, L) clean

        # Pad pour U-Net
        T_orig = Y.shape[-1]
        Y = pad_spec(Y)
        X = pad_spec(X)

        # Diffusion training step: sample t, ajouter bruit, calculer loss
        B = X.shape[0]
        t = (
            torch.rand(B, device=device) * (score_model.sde.T - score_model.t_eps)
            + score_model.t_eps
        )
        mean, std = score_model.sde.marginal_prob(X, Y, t)
        z = torch.randn_like(X)
        sigma = std[:, None, None, None]
        x_t = mean + sigma * z

        # Forward + loss
        optimizer.zero_grad()
        forward_out = score_model(x_t, Y, t)
        loss = score_model.compute_loss(forward_out, x_t, z, t, mean, X)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(score_model.dnn.parameters(), GRAD_CLIP)
        optimizer.step()

        # Mise a jour EMA (critique pour SGMSE+)
        score_model.update_ema()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(score_model, loader, device, window):
    """Validation : calcule la loss score matching."""
    score_model.eval()
    total_loss = 0
    n_batches = 0

    with torch.no_grad():
        for raw, clean in loader:
            raw, clean = raw.to(device), clean.to(device)

            raw_norm = torch.clamp(raw.abs().amax(dim=-1, keepdim=True), min=1e-8)
            clean_norm = torch.clamp(clean.abs().amax(dim=-1, keepdim=True), min=1e-8)
            raw_n = raw / raw_norm
            clean_n = clean / clean_norm

            Y = spec_fwd(stft(raw_n, window)).unsqueeze(1)
            X = spec_fwd(stft(clean_n, window)).unsqueeze(1)

            T_orig = Y.shape[-1]
            Y = pad_spec(Y)
            X = pad_spec(X)

            B = X.shape[0]
            t = (
                torch.rand(B, device=device) * (score_model.sde.T - score_model.t_eps)
                + score_model.t_eps
            )
            mean, std = score_model.sde.marginal_prob(X, Y, t)
            z = torch.randn_like(X)
            sigma = std[:, None, None, None]
            x_t = mean + sigma * z

            forward_out = score_model(x_t, Y, t)
            loss = score_model.compute_loss(forward_out, x_t, z, t, mean, X)

            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


def enhance_audio(score_model, raw_wav, device, window):
    """Enhance un signal audio avec SGMSE+ (sampling complet)."""
    score_model.eval()
    with torch.no_grad():
        raw = raw_wav.unsqueeze(0).to(device)  # (1, T)
        norm = torch.clamp(raw.abs().amax(dim=-1, keepdim=True), min=1e-8)
        y = raw / norm

        Y = spec_fwd(stft(y, window)).unsqueeze(1)  # (1, 1, F, L)
        T_orig = Y.shape[-1]
        Yp = pad_spec(Y)

        # Sampling (30 pas de diffusion)
        x_hat = score_model.enhance(Yp)

        # Trim + inverse
        Xh = x_hat[:, :, :, :T_orig].squeeze(1)
        Xh = spec_back(Xh)
        enh = istft(Xh, window, length=y.shape[-1]) * norm
        return enh.squeeze(0)  # (T,)


def compute_metrics(score_model, val_dataset, device, window, n_samples=5):
    """Calcule PESQ et STOI sur quelques echantillons."""
    from pesq import pesq as pesq_score
    from pystoi import stoi as stoi_score

    # Utiliser les poids EMA pour l'evaluation
    score_model.store_ema()

    pesq_vals, stoi_vals = [], []
    indices = np.random.choice(len(val_dataset), min(n_samples, len(val_dataset)), replace=False)

    for idx in indices:
        raw, clean = val_dataset[idx]
        try:
            est = enhance_audio(score_model, raw, device, window)
            est_np = est.cpu().numpy()
            clean_np = clean.numpy()

            mn = min(len(est_np), len(clean_np))
            est_np, clean_np = est_np[:mn], clean_np[:mn]

            try:
                p = pesq_score(SR_MODEL, clean_np, est_np, "wb")
                pesq_vals.append(p)
            except Exception:
                pass
            try:
                s = stoi_score(clean_np, est_np, SR_MODEL, extended=False)
                stoi_vals.append(s)
            except Exception:
                pass
        except Exception as e:
            print(f"    [!] Erreur metrics sample {idx}: {e}")

    # Restaurer les poids normaux (non-EMA)
    score_model.restore_ema()

    pesq_avg = np.mean(pesq_vals) if pesq_vals else 0
    stoi_avg = np.mean(stoi_vals) if stoi_vals else 0
    return pesq_avg, stoi_avg


# === Main ===
def main():
    print("=== Fine-tuning SGMSE+ pour debruitage radio ===")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Epochs: {EPOCHS}")
    print(f"Segment: {SEGMENT_LENGTH_S}s ({SEGMENT_LENGTH} samples @ {SR_MODEL}Hz)")
    print(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}")
    print(f"Modele: ~65.6M params (backbone ncsnpp_v2 + OUVE SDE)")
    print()

    # Dataset
    print("Chargement du dataset...")
    train_dataset = RadioEnhancementDataset(DATASET_DIR, split="train")
    val_dataset = RadioEnhancementDataset(DATASET_DIR, split="val")

    pin = DEVICE.type == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    # Modele
    print("Chargement SGMSE+ pre-entraine...")
    pretrained = load_sgmse_model()
    score_model = pretrained.mods.score_model

    # Reactiver les gradients (SpeechBrain charge en mode eval)
    for param in score_model.parameters():
        param.requires_grad_(True)

    score_model = score_model.to(DEVICE)

    n_params = sum(p.numel() for p in score_model.parameters() if p.requires_grad)
    print(f"  Parametres entrainables: {n_params:,}")
    print()

    # STFT window
    window = get_window(DEVICE)

    # Optimizer (uniquement les params du DNN, pas l'EMA)
    optimizer = torch.optim.Adam(score_model.dnn.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    # Reprise depuis un checkpoint
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Reprise depuis {RESUME_FROM}...")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        score_model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"  Reprise a l'epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")
        print()

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss = train_one_epoch(score_model, train_loader, optimizer, DEVICE, window)
        val_loss = validate(score_model, val_loader, DEVICE, window)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        # Metriques tous les 5 epochs (lent car sampling 30 pas)
        metrics_str = ""
        if epoch % 5 == 0 or epoch == 1:
            pesq_avg, stoi_avg = compute_metrics(
                score_model, val_dataset, DEVICE, window
            )
            metrics_str = f" | PESQ: {pesq_avg:.2f} | STOI: {stoi_avg:.3f}"

        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}{metrics_str} | "
            f"LR: {lr:.2e}"
        )

        # Sauvegarde meilleur modele (avec EMA)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": score_model.state_dict(),
                "ema_state_dict": score_model.ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {
                    "sr": SR_MODEL, "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                    "segment_length": SEGMENT_LENGTH,
                },
            }, ckpt_path)
            print(f"  -> Meilleur modele sauvegarde ({ckpt_path})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping a l'epoch {epoch}")
                break

        # Checkpoint tous les 10 epochs
        if epoch % 10 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": score_model.state_dict(),
                "ema_state_dict": score_model.ema.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

    print(f"\nEntrainement termine ! Meilleur val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {OUTPUT_DIR}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Score Matching Loss")
    ax.set_title("SGMSE+ fine-tuning - Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "sgmse_training_loss.png"), dpi=100)
    plt.close()


if __name__ == "__main__":
    main()
