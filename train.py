"""
Fine-tuning d'un modele de super-resolution audio pour communications radio.

Strategie : on entraine un ResUNet spectral (STFT domain) qui apprend a
reconstruire le spectrogramme clean a partir du spectrogramme degrade.
Le modele travaille en 44.1 kHz pour reconstruire les frequences jusqu'a ~20 kHz.

Architecture : ResUNet dans le domaine STFT (magnitude + phase)
Loss : L1 spectrale + Multi-Resolution STFT Loss
Dataset : paires (raw, clean) generees par create_dataset.py
"""
import os
import sys
import glob
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib
from pesq import pesq as pesq_score
from pystoi import stoi as stoi_score

sys.stdout.reconfigure(encoding="utf-8")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === DirectML / Device detection ===
try:
    import torch_directml
    DIRECTML_AVAILABLE = True
except ImportError:
    DIRECTML_AVAILABLE = False

# === Configuration ===

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")

# Hyperparametres
BATCH_SIZE = 8
LEARNING_RATE = 3e-4
EPOCHS = 80
SEGMENT_LENGTH = 44100 * 3  # 3 secondes a 44.1kHz
NUM_WORKERS = 0 if sys.platform == "win32" else 4

# STFT parametres (pour le modele spectral)
N_FFT = 2048
HOP_LENGTH = 512
WIN_LENGTH = 2048

# Loss weights
L1_WEIGHT = 1.0
STFT_WEIGHT = 1.0
IDENTITY_WEIGHT = 0.3  # Penalise la sur-transformation du signal (evite la degradation)

# Paires (clean, clean) dans le dataset : apprend au modele a ne rien faire sur signal propre
CLEAN_PAIR_RATIO = 0.15  # 15% des exemples sont des paires identiques (clean, clean)

# Early stopping
PATIENCE = 15  # Nombre d'epochs sans amelioration avant arret
MIN_DELTA = 1e-4  # Amelioration minimale pour compter comme progres

# Reprise depuis un checkpoint (None = entrainement depuis zero)
RESUME_FROM = None

# Sample rate du dataset
SR = 44100

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if DIRECTML_AVAILABLE:
        return torch_directml.device()
    return torch.device("cpu")

DEVICE = get_device()
# DirectML ne supporte pas torch.stft/istft, on les fait sur CPU
STFT_DEVICE = torch.device("cpu") if DIRECTML_AVAILABLE and not torch.cuda.is_available() else DEVICE


# === Dataset ===

class AudioPairDataset(Dataset):
    """Dataset de paires (raw, clean) pour entrainement avec data augmentation."""

    def __init__(self, dataset_dir, split="train", segment_length=SEGMENT_LENGTH, augment=False):
        self.segment_length = segment_length
        self.augment = augment
        self.clean_dir = os.path.join(dataset_dir, split, "clean")
        self.raw_dir = os.path.join(dataset_dir, split, "raw")

        self.files = sorted([
            os.path.basename(f)
            for f in glob.glob(os.path.join(self.clean_dir, "*.wav"))
        ])
        print(f"  {split}: {len(self.files)} paires chargees" +
              (" (augmentation ON)" if augment else ""))

    def __len__(self):
        return len(self.files)

    def _augment(self, raw, clean):
        """Data augmentation coherente (appliquee aux deux signaux)."""
        # Gain aleatoire (-6dB a +6dB)
        if np.random.random() < 0.5:
            gain = 10 ** (np.random.uniform(-6, 6) / 20)
            raw = raw * gain
            clean = clean * gain

        # Time shift aleatoire (+-500 samples)
        if np.random.random() < 0.3:
            shift = np.random.randint(-500, 500)
            raw = np.roll(raw, shift)
            clean = np.roll(clean, shift)

        # Bruit additif leger (uniquement sur raw, SNR 25-40 dB)
        if np.random.random() < 0.4:
            rms = np.sqrt(np.mean(raw ** 2)) + 1e-8
            snr_db = np.random.uniform(25, 40)
            noise_rms = rms / (10 ** (snr_db / 20))
            raw = raw + np.random.randn(len(raw)).astype(np.float32) * noise_rms

        return raw, clean

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean, _ = sf.read(os.path.join(self.clean_dir, fname), dtype="float32")
        raw, _ = sf.read(os.path.join(self.raw_dir, fname), dtype="float32")

        min_len = min(len(clean), len(raw))
        clean = clean[:min_len]
        raw = raw[:min_len]

        if len(clean) >= self.segment_length:
            start = np.random.randint(0, len(clean) - self.segment_length + 1)
            clean = clean[start:start + self.segment_length]
            raw = raw[start:start + self.segment_length]
        else:
            pad = self.segment_length - len(clean)
            clean = np.pad(clean, (0, pad))
            raw = np.pad(raw, (0, pad))

        if self.augment:
            raw, clean = self._augment(raw, clean)

        # Paires (clean, clean) : apprend au modele l'identite sur signal propre
        if self.augment and np.random.random() < CLEAN_PAIR_RATIO:
            raw = clean.copy()

        return (
            torch.from_numpy(raw).float().unsqueeze(0),
            torch.from_numpy(clean).float().unsqueeze(0),
        )


# === ResUNet Spectral ===

class ResBlock(nn.Module):
    """Bloc residuel avec 2 convolutions."""

    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(channels)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)


class EncoderBlock(nn.Module):
    """Encoder : Conv2D stride 2 + ResBlock."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)
        self.res = ResBlock(out_ch)

    def forward(self, x):
        x = self.act(self.bn(self.conv(x)))
        x = self.res(x)
        return x


class DecoderBlock(nn.Module):
    """Decoder : ConvTranspose2D stride 2 + ResBlock + skip connection."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, stride=2, padding=1, output_padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.LeakyReLU(0.2)
        # Apres concat avec skip : out_ch * 2 -> out_ch
        self.conv1x1 = nn.Conv2d(out_ch * 2, out_ch, 1)
        self.res = ResBlock(out_ch)

    def forward(self, x, skip):
        x = self.act(self.bn(self.deconv(x)))
        # Ajuster taille si necessaire
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv1x1(x)
        x = self.res(x)
        return x


class SpectralResUNet(nn.Module):
    """ResUNet operant sur les spectrogrammes STFT.

    Entree : signal audio [B, 1, T]
    -> STFT -> magnitude spectrogramme [B, 1, F, T']
    -> ResUNet encode/decode
    -> masque spectral [B, 1, F, T']
    -> applique le masque + phase originale
    -> iSTFT -> signal reconstruit [B, 1, T]
    """

    def __init__(self, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

        # Encoder (1 -> 32 -> 64 -> 128 -> 256)
        self.enc1 = EncoderBlock(1, 32)
        self.enc2 = EncoderBlock(32, 64)
        self.enc3 = EncoderBlock(64, 128)
        self.enc4 = EncoderBlock(128, 256)

        # Bottleneck
        self.bottleneck = ResBlock(256)

        # Decoder (256 -> 128 -> 64 -> 32)
        self.dec4 = DecoderBlock(256, 128)
        self.dec3 = DecoderBlock(128, 64)
        self.dec2 = DecoderBlock(64, 32)
        self.dec1 = DecoderBlock(32, 16)

        # Sortie : masque spectral
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, 1),
            nn.Sigmoid(),  # Masque entre 0 et 1
        )

        # Projection de l'entree (1 ch -> 16 ch) pour le skip de dec1
        self.input_proj = nn.Conv2d(1, 16, 1)

        # Couche pour estimer les hautes frequences (au-dela du masque)
        self.hf_estimator = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
        )

    def forward(self, x):
        # x: [B, 1, T] sur DEVICE (GPU ou CPU)
        B, _, T = x.shape
        model_device = x.device

        # STFT sur CPU (DirectML ne supporte pas torch.stft)
        x_cpu = x.squeeze(1).to(STFT_DEVICE)  # [B, T]
        window = torch.hann_window(self.win_length, device=STFT_DEVICE)
        stft = torch.stft(x_cpu, self.n_fft, self.hop_length, self.win_length,
                          window, return_complex=True)  # [B, F, T']
        mag = torch.abs(stft).unsqueeze(1)    # [B, 1, F, T']
        phase = torch.angle(stft).unsqueeze(1)  # [B, 1, F, T']

        # Transferer sur le device GPU pour les convolutions
        mag_gpu = mag.to(model_device)
        log_mag = torch.log1p(mag_gpu)

        # Encoder
        e1 = self.enc1(log_mag)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d4 = self.dec4(b, e3)
        d3 = self.dec3(d4, e2)
        d2 = self.dec2(d3, e1)
        d1 = self.dec1(d2, self.input_proj(log_mag))

        # Masque spectral
        mask = self.final(d1)  # [B, 1, F, T']

        # Estimation HF additive
        hf = self.hf_estimator(d1)  # [B, 1, F, T']

        # Gate HF par l'energie locale du signal : evite l'hallucination sur silence
        # Si le signal est silencieux, energy_gate -> 0, donc aucune HF ajoutee
        input_energy = mag_gpu.mean(dim=[2, 3], keepdim=True).clamp(min=1e-8)
        energy_gate = torch.sigmoid(torch.log(input_energy) + 4)  # ~0 si silence, ~1 si signal

        # Appliquer masque sur GPU puis transferer pour iSTFT
        enhanced_mag = mask * mag_gpu + F.relu(hf) * energy_gate
        enhanced_mag_cpu = enhanced_mag.squeeze(1).to(STFT_DEVICE)
        phase_cpu = phase.squeeze(1)  # deja sur CPU

        # Reconstruire le complexe avec la phase originale
        enhanced_complex = enhanced_mag_cpu * torch.exp(1j * phase_cpu)

        # iSTFT sur CPU
        output = torch.istft(enhanced_complex, self.n_fft, self.hop_length,
                             self.win_length, window)  # [B, T']

        # Ajuster la longueur
        if output.shape[1] > T:
            output = output[:, :T]
        elif output.shape[1] < T:
            output = F.pad(output, (0, T - output.shape[1]))

        return output.unsqueeze(1).to(model_device)  # [B, 1, T] retour sur GPU


# === Loss functions ===

class MultiResolutionSTFTLoss(nn.Module):
    """Multi-Resolution STFT Loss."""

    def __init__(self, fft_sizes=[512, 1024, 2048], hop_sizes=[128, 256, 512], win_sizes=[512, 1024, 2048]):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_sizes = win_sizes

    def _stft_loss(self, x, y, fft_size, hop_size, win_size):
        # STFT sur CPU (DirectML ne supporte pas torch.stft)
        orig_device = x.device
        x_cpu = x.to(STFT_DEVICE)
        y_cpu = y.to(STFT_DEVICE)
        window = torch.hann_window(win_size, device=STFT_DEVICE)
        x_stft = torch.stft(x_cpu, fft_size, hop_size, win_size, window, return_complex=True)
        y_stft = torch.stft(y_cpu, fft_size, hop_size, win_size, window, return_complex=True)

        x_mag = torch.abs(x_stft).to(orig_device)
        y_mag = torch.abs(y_stft).to(orig_device)

        sc_loss = torch.norm(y_mag - x_mag, p="fro") / (torch.norm(y_mag, p="fro") + 1e-8)
        mag_loss = F.l1_loss(torch.log(x_mag + 1e-8), torch.log(y_mag + 1e-8))

        return sc_loss + mag_loss

    def forward(self, x, y):
        loss = 0
        for fft_size, hop_size, win_size in zip(self.fft_sizes, self.hop_sizes, self.win_sizes):
            loss += self._stft_loss(x, y, fft_size, hop_size, win_size)
        return loss / len(self.fft_sizes)


# === Entrainement ===

def train_one_epoch(model, dataloader, optimizer, l1_loss_fn, stft_loss_fn, device):
    model.train()
    total_loss = 0
    total_l1 = 0
    total_stft = 0
    total_identity = 0

    for raw, clean in dataloader:
        raw = raw.to(device)
        clean = clean.to(device)

        optimizer.zero_grad()
        output = model(raw)

        loss_l1 = l1_loss_fn(output, clean)
        loss_stft = stft_loss_fn(output.squeeze(1), clean.squeeze(1))

        # Identity loss : penalise la sur-transformation du signal d'entree
        # Apprend au modele a ne pas modifier ce qui n'a pas besoin de l'etre
        loss_identity = l1_loss_fn(output, raw)

        loss = L1_WEIGHT * loss_l1 + STFT_WEIGHT * loss_stft + IDENTITY_WEIGHT * loss_identity
        loss.backward()

        # Gradient clipping pour stabilite
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        total_l1 += loss_l1.item()
        total_stft += loss_stft.item()
        total_identity += loss_identity.item()

    n = len(dataloader)
    return total_loss / n, total_l1 / n, total_stft / n, total_identity / n


def compute_perceptual_metrics(output, clean, sr=SR, max_samples=8):
    """Calcule PESQ et STOI sur un sous-ensemble du batch."""
    pesq_vals = []
    stoi_vals = []
    out_np = output.squeeze(1).cpu().numpy()
    clean_np = clean.squeeze(1).cpu().numpy()

    n = min(len(out_np), max_samples)
    for i in range(n):
        try:
            # PESQ attend du 16kHz - on resample
            from scipy.signal import resample
            length_16k = int(len(out_np[i]) * 16000 / sr)
            out_16k = resample(out_np[i], length_16k).astype(np.float32)
            clean_16k = resample(clean_np[i], length_16k).astype(np.float32)
            p = pesq_score(16000, clean_16k, out_16k, "wb")
            pesq_vals.append(p)
        except Exception:
            pass
        try:
            s = stoi_score(clean_np[i], out_np[i], sr, extended=False)
            stoi_vals.append(s)
        except Exception:
            pass

    avg_pesq = np.mean(pesq_vals) if pesq_vals else 0.0
    avg_stoi = np.mean(stoi_vals) if stoi_vals else 0.0
    return avg_pesq, avg_stoi


def validate(model, dataloader, l1_loss_fn, stft_loss_fn, device, compute_metrics=False):
    model.eval()
    total_loss = 0
    all_pesq = []
    all_stoi = []

    with torch.no_grad():
        for raw, clean in dataloader:
            raw = raw.to(device)
            clean = clean.to(device)
            output = model(raw)

            loss_l1 = l1_loss_fn(output, clean)
            loss_stft = stft_loss_fn(output.squeeze(1), clean.squeeze(1))
            loss = L1_WEIGHT * loss_l1 + STFT_WEIGHT * loss_stft
            total_loss += loss.item()

            if compute_metrics:
                p, s = compute_perceptual_metrics(output, clean)
                if p > 0:
                    all_pesq.append(p)
                if s > 0:
                    all_stoi.append(s)

    avg_loss = total_loss / len(dataloader)
    avg_pesq = np.mean(all_pesq) if all_pesq else 0.0
    avg_stoi = np.mean(all_stoi) if all_stoi else 0.0
    return avg_loss, avg_pesq, avg_stoi


def plot_losses(train_losses, val_losses, output_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(train_losses, label="Train", linewidth=1)
    ax.plot(val_losses, label="Validation", linewidth=1)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss - SpectralResUNet")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_metrics(pesq_hist, stoi_hist, output_path):
    """Graphique PESQ et STOI au fil des epochs."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    epochs = range(1, len(pesq_hist) + 1)
    # Filtrer les zeros (epochs sans calcul de metriques)
    pesq_valid = [(e, v) for e, v in zip(epochs, pesq_hist) if v > 0]
    stoi_valid = [(e, v) for e, v in zip(epochs, stoi_hist) if v > 0]

    if pesq_valid:
        ax1.plot(*zip(*pesq_valid), "o-", color="tab:blue", linewidth=1)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("PESQ")
    ax1.set_title("PESQ (qualite percue, max 4.5)")
    ax1.grid(True, alpha=0.3)

    if stoi_valid:
        ax2.plot(*zip(*stoi_valid), "o-", color="tab:green", linewidth=1)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("STOI")
    ax2.set_title("STOI (intelligibilite, max 1.0)")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_spectrogram_sample(model, val_dataset, device, output_path, sr=SR):
    """Genere un spectrogramme avant/apres sur un sample de validation."""
    model.eval()
    raw, clean = val_dataset[0]
    raw = raw.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(raw)

    raw_np = raw.squeeze().cpu().numpy()
    clean_np = clean.squeeze().numpy()
    out_np = output.squeeze().cpu().numpy()

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    for ax, data, title in [
        (axes[0], raw_np, "Entree (degradee)"),
        (axes[1], out_np, "Sortie (modele)"),
        (axes[2], clean_np, "Cible (clean)"),
    ]:
        ax.specgram(data, NFFT=2048, Fs=sr, noverlap=1024, cmap="inferno")
        ax.set_title(title)
        ax.set_ylabel("Freq (Hz)")
        ax.set_ylim(0, sr // 2)
    axes[2].set_xlabel("Temps (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=== Entrainement SpectralResUNet - Super-Resolution Audio ===")
    print(f"Device: {DEVICE}")
    print(f"Batch size: {BATCH_SIZE} | LR: {LEARNING_RATE} | Epochs: {EPOCHS}")
    print(f"Segment: {SEGMENT_LENGTH / SR:.1f}s ({SEGMENT_LENGTH} samples @ {SR}Hz)")
    print(f"STFT: n_fft={N_FFT}, hop={HOP_LENGTH}\n")

    # Dataset
    print("Chargement du dataset...")
    train_dataset = AudioPairDataset(DATASET_DIR, split="train", segment_length=SEGMENT_LENGTH, augment=True)
    val_dataset = AudioPairDataset(DATASET_DIR, split="val", segment_length=SEGMENT_LENGTH, augment=False)

    pin = DEVICE.type == "cuda" if hasattr(DEVICE, "type") else False
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=pin,
    )

    # Modele
    model = SpectralResUNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Modele: SpectralResUNet ({n_params:,} parametres)\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    l1_loss_fn = nn.L1Loss()
    stft_loss_fn = MultiResolutionSTFTLoss()

    # Boucle d'entrainement
    train_losses = []
    val_losses = []
    val_pesq_hist = []
    val_stoi_hist = []
    best_val_loss = float("inf")
    patience_counter = 0
    start_epoch = 1

    # Reprise depuis un checkpoint
    if RESUME_FROM and os.path.exists(RESUME_FROM):
        print(f"Reprise depuis : {RESUME_FROM}")
        ckpt = torch.load(RESUME_FROM, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        # Restaurer le meilleur val_loss connu (depuis best_model.pt)
        best_ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
        if os.path.exists(best_ckpt_path):
            best_ckpt = torch.load(best_ckpt_path, map_location=DEVICE)
            best_val_loss = best_ckpt["val_loss"]
            best_epoch = best_ckpt.get("epoch", start_epoch - 1)
            patience_counter = (start_epoch - 1) - best_epoch
            print(f"  Meilleur val_loss connu : {best_val_loss:.4f} (epoch {best_epoch})")
            print(f"  Patience actuelle : {patience_counter}/{PATIENCE}")
        # Avancer le scheduler jusqu'a l'epoch de reprise
        for _ in range(start_epoch - 1):
            scheduler.step()
        print(f"  Reprise a l'epoch {start_epoch} | LR : {optimizer.param_groups[0]['lr']:.2e}\n")

    for epoch in range(start_epoch, EPOCHS + 1):
        train_loss, train_l1, train_stft, train_identity = train_one_epoch(
            model, train_loader, optimizer, l1_loss_fn, stft_loss_fn, DEVICE,
        )

        # Metriques perceptuelles tous les 5 epochs (couteux en calcul)
        compute_metrics = (epoch % 5 == 0) or (epoch == 1)
        val_loss, val_pesq, val_stoi = validate(
            model, val_loader, l1_loss_fn, stft_loss_fn, DEVICE,
            compute_metrics=compute_metrics,
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_pesq_hist.append(val_pesq)
        val_stoi_hist.append(val_stoi)
        scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        metrics_str = ""
        if compute_metrics:
            metrics_str = f" | PESQ: {val_pesq:.2f} | STOI: {val_stoi:.3f}"
        print(
            f"  Epoch {epoch:3d}/{EPOCHS} | "
            f"Train: {train_loss:.4f} (L1={train_l1:.4f}, STFT={train_stft:.4f}, ID={train_identity:.4f}) | "
            f"Val: {val_loss:.4f}{metrics_str} | LR: {lr:.2e}"
        )

        # Sauvegarde du meilleur modele
        if val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = os.path.join(OUTPUT_DIR, "best_model.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "pesq": val_pesq,
                "stoi": val_stoi,
                "config": {
                    "n_fft": N_FFT, "hop_length": HOP_LENGTH,
                    "win_length": WIN_LENGTH, "sr": SR,
                },
            }, ckpt_path)
            print(f"  -> Meilleur modele sauvegarde ({ckpt_path})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping a l'epoch {epoch} (pas d'amelioration depuis {PATIENCE} epochs)")
                break

        if epoch % 10 == 0:
            ckpt_path = os.path.join(OUTPUT_DIR, f"checkpoint_epoch{epoch:03d}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            }, ckpt_path)

            # Spectrogramme echantillon
            plot_spectrogram_sample(
                model, val_dataset, DEVICE,
                os.path.join(LOG_DIR, f"sample_epoch{epoch:03d}.png"),
            )

    # Graphiques finaux
    plot_losses(train_losses, val_losses, os.path.join(LOG_DIR, "training_loss.png"))
    plot_metrics(val_pesq_hist, val_stoi_hist, os.path.join(LOG_DIR, "training_metrics.png"))
    print(f"\nEntrainement termine ! Meilleur val_loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
