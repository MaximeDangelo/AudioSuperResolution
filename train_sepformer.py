"""
Fine-tuning de SepFormer (SpeechBrain) pour le debruitage radio cockpit.

Charge le modele pre-entraine sepformer-wham16k-enhancement et le fine-tune
sur les paires (raw, clean) du dataset radio.

Le modele travaille a 16 kHz (resample automatique depuis 44.1 kHz).
"""
import os
import sys
import glob
import json
import csv
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.signal import resample_poly
from math import gcd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")

# === Configuration ===
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(SCRIPT_DIR, "dataset")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "checkpoints", "sepformer")
LOG_DIR = os.path.join(SCRIPT_DIR, "logs")
PRETRAINED_DIR = os.path.join(SCRIPT_DIR, "pretrained_models", "sepformer-wham16k-enhancement")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Hyperparametres
BATCH_SIZE = 2          # Max pour 16GB VRAM avec SepFormer
LEARNING_RATE = 1e-5    # LR bas pour fine-tuning
EPOCHS = 30
SEGMENT_LENGTH_S = 2.0  # secondes (reduit pour tenir en VRAM)
SR_MODEL = 16000        # SepFormer travaille a 16 kHz
SR_DATASET = 44100      # Dataset genere a 44.1 kHz
SEGMENT_LENGTH = int(SEGMENT_LENGTH_S * SR_MODEL)
NUM_WORKERS = 0 if sys.platform == "win32" else 4
PATIENCE = 10

# Device
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()


# === Dataset ===
class RadioEnhancementDataset(Dataset):
    """Dataset de paires (raw, clean) resamplees a 16 kHz pour SepFormer."""

    def __init__(self, dataset_dir, split="train", segment_length=SEGMENT_LENGTH):
        self.segment_length = segment_length
        # Utiliser les fichiers pre-resamples a 16kHz
        raw_dir = os.path.join(dataset_dir, split, "raw_16k")
        clean_dir = os.path.join(dataset_dir, split, "clean_16k")

        raw_files = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
        self.pairs = []
        for rf in raw_files:
            cf = os.path.join(clean_dir, os.path.basename(rf))
            if os.path.exists(cf):
                self.pairs.append((rf, cf))

        print(f"  {split}: {len(self.pairs)} paires chargees (16kHz pre-resample)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, clean_path = self.pairs[idx]
        raw, _ = sf.read(raw_path, dtype="float32")
        clean, _ = sf.read(clean_path, dtype="float32")
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
            # Padding si trop court
            pad = self.segment_length - mn
            raw = np.pad(raw, (0, pad))
            clean = np.pad(clean, (0, pad))

        return torch.tensor(raw), torch.tensor(clean)


# === Charger le modele SepFormer ===
def load_sepformer_model():
    """Charge SepFormer depuis les poids pre-entraines."""
    from speechbrain.inference.separation import SepformerSeparation

    # Charger le modele pre-entraine
    model = SepformerSeparation.from_hparams(
        source="speechbrain/sepformer-wham16k-enhancement",
        savedir=PRETRAINED_DIR,
    )
    return model


class SepFormerEnhancer(torch.nn.Module):
    """Wrapper autour de SepFormer pour le fine-tuning."""

    def __init__(self, pretrained_model, freeze_encoder_decoder=True):
        super().__init__()
        self.encoder = pretrained_model.mods.encoder
        self.masknet = pretrained_model.mods.masknet
        self.decoder = pretrained_model.mods.decoder
        # Reactiver les gradients sur le masknet (la partie a fine-tuner)
        for param in self.masknet.parameters():
            param.requires_grad_(True)
        if freeze_encoder_decoder:
            # Geler encoder et decoder (deja bons, pas besoin de les modifier)
            for param in self.encoder.parameters():
                param.requires_grad_(False)
            for param in self.decoder.parameters():
                param.requires_grad_(False)
            n_train = sum(p.numel() for p in self.masknet.parameters() if p.requires_grad)
            n_total = sum(p.numel() for p in self.parameters())
            print(f"  Fine-tuning masknet uniquement: {n_train:,} / {n_total:,} params entrainables")
        else:
            for param in self.parameters():
                param.requires_grad_(True)

    def forward(self, mix):
        """
        mix: [B, T] signal bruite
        return: [B, T] signal debruite
        """
        # Encoder attend [B, T]
        if mix.dim() == 3:
            mix = mix.squeeze(1)

        mix_w = self.encoder(mix)  # [B, N, L]

        # Mask estimation
        est_mask = self.masknet(mix_w)  # [n_src, B, N, L]

        # Appliquer le masque de la premiere source (voix)
        sep_h = mix_w * est_mask[0]  # [B, N, L]

        # Decoder
        est_source = self.decoder(sep_h)  # [B, 1, T]
        est_source = est_source.squeeze(1)  # [B, T]

        # Ajuster la longueur
        T_in = mix.shape[-1]
        T_out = est_source.shape[-1]
        if T_out > T_in:
            est_source = est_source[..., :T_in]
        elif T_out < T_in:
            est_source = F.pad(est_source, (0, T_in - T_out))

        return est_source


# === Loss ===
def si_snr_loss(est, target):
    """Scale-Invariant SNR loss (negation pour minimiser)."""
    # Normaliser
    target = target - target.mean(dim=-1, keepdim=True)
    est = est - est.mean(dim=-1, keepdim=True)

    # SI-SNR
    dot = torch.sum(target * est, dim=-1, keepdim=True)
    s_target_energy = torch.sum(target ** 2, dim=-1, keepdim=True) + 1e-8
    proj = dot * target / s_target_energy

    noise = est - proj
    si_snr = 10 * torch.log10(
        torch.sum(proj ** 2, dim=-1) / (torch.sum(noise ** 2, dim=-1) + 1e-8) + 1e-8
    )
    return -si_snr.mean()  # Negation : on minimise


# === Entrainement ===
def train_one_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    n_batches = 0
    use_amp = scaler is not None
    for raw, clean in loader:
        raw, clean = raw.to(device), clean.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            est = model(raw)
            mn = min(est.shape[-1], clean.shape[-1])
            loss = si_snr_loss(est[..., :mn], clean[..., :mn])

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def validate(model, loader, device):
    model.eval()
    total_loss = 0
    n_batches = 0
    with torch.no_grad():
        for raw, clean in loader:
            raw, clean = raw.to(device), clean.to(device)
            est = model(raw)
            mn = min(est.shape[-1], clean.shape[-1])
            loss = si_snr_loss(est[..., :mn], clean[..., :mn])
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def compute_metrics(model, val_dataset, device, n_samples=5):
    """Calcule PESQ et STOI sur quelques echantillons."""
    from pesq import pesq as pesq_score
    from pystoi import stoi as stoi_score

    model.eval()
    pesq_vals, stoi_vals = [], []

    indices = np.random.choice(len(val_dataset), min(n_samples, len(val_dataset)), replace=False)
    for idx in indices:
        raw, clean = val_dataset[idx]
        with torch.no_grad():
            est = model(raw.unsqueeze(0).to(device))  # [1, T]
        est_np = est.squeeze(0).cpu().numpy()
        clean_np = clean.numpy()

        mn = min(len(est_np), len(clean_np))
        est_np, clean_np = est_np[:mn], clean_np[:mn]

        try:
            p = pesq_score(SR_MODEL, clean_np, est_np, "wb")
            pesq_vals.append(p)
        except:
            pass
        try:
            s = stoi_score(clean_np, est_np, SR_MODEL, extended=False)
            stoi_vals.append(s)
        except:
            pass

    pesq_avg = np.mean(pesq_vals) if pesq_vals else 0
    stoi_avg = np.mean(stoi_vals) if stoi_vals else 0
    return pesq_avg, stoi_avg


# === Main ===
def main():
    print("=== Fine-tuning SepFormer pour debruitage radio ===")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Epochs: {EPOCHS}")
    print(f"Segment: {SEGMENT_LENGTH_S}s ({SEGMENT_LENGTH} samples @ {SR_MODEL}Hz)")
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
    print("Chargement SepFormer pre-entraine...")
    pretrained = load_sepformer_model()
    model = SepFormerEnhancer(pretrained).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modele: SepFormerEnhancer ({n_params:,} parametres)\n")

    # Optimiser uniquement les parametres entrainables
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
    )

    # Mixed precision (AMP)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None
    if scaler:
        print("Mixed precision (AMP) active\n")

    # Boucle d'entrainement
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, DEVICE, scaler)
        val_loss = validate(model, val_loader, DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]

        # Metriques tous les 5 epochs
        metrics_str = ""
        if epoch % 5 == 0 or epoch == 1:
            pesq_avg, stoi_avg = compute_metrics(model, val_dataset, DEVICE)
            metrics_str = f" | PESQ: {pesq_avg:.2f} | STOI: {stoi_avg:.3f}"

        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f}{metrics_str} | "
            f"LR: {lr:.2e}"
        )

        # Sauvegarde meilleur modele
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "config": {"sr": SR_MODEL, "segment_length": SEGMENT_LENGTH},
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
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

    print(f"\nEntrainement termine ! Meilleur val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {OUTPUT_DIR}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train")
    ax.plot(val_losses, label="Val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("SI-SNR Loss")
    ax.set_title("SepFormer fine-tuning - Loss")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, "sepformer_training_loss.png"), dpi=100)
    plt.close()


if __name__ == "__main__":
    main()
