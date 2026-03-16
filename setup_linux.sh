#!/bin/bash
# Installation automatique du projet Audio Super-Resolution sous Linux
# GPU cible : AMD RX 6800 via ROCm

set -e

echo "=== Audio Super-Resolution - Installation Linux ==="
echo ""

# Verifier Python
PYTHON=${PYTHON:-python3}
PY_VERSION=$($PYTHON --version 2>&1)
echo "Python: $PY_VERSION"

# Verifier ROCm
if command -v rocminfo &> /dev/null; then
    echo "ROCm: detecte"
    rocminfo | grep "Marketing Name" | head -1 || true
else
    echo "ATTENTION: ROCm non detecte."
    echo "Installer ROCm : https://rocm.docs.amd.com/projects/install-on-linux/en/latest/"
    echo "Continuer sans GPU ? (Ctrl+C pour annuler)"
    read -r
fi

# Creer le venv
echo ""
echo "--- Creation du venv ---"
$PYTHON -m venv venv
source venv/bin/activate

# Installer PyTorch ROCm
echo ""
echo "--- Installation PyTorch ROCm ---"
pip install --upgrade pip
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# Verifier que PyTorch voit le GPU
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA/ROCm disponible: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
else:
    print('ATTENTION: GPU non detecte par PyTorch')
"

# Installer les autres dependances
echo ""
echo "--- Installation des dependances ---"
pip install -r requirements-linux.txt

# Verifier SpeechBrain
python -c "import speechbrain; print(f'SpeechBrain {speechbrain.__version__}')"

# Installer ffmpeg si necessaire
if ! command -v ffmpeg &> /dev/null; then
    echo ""
    echo "ATTENTION: ffmpeg non trouve. Installer avec :"
    echo "  sudo apt install ffmpeg"
fi

# Installer libsndfile si necessaire
if ! python -c "import soundfile" 2>/dev/null; then
    echo ""
    echo "ATTENTION: libsndfile manquant. Installer avec :"
    echo "  sudo apt install libsndfile1"
fi

echo ""
echo "=== Installation terminee ==="
echo ""
echo "Commandes disponibles :"
echo "  source venv/bin/activate"
echo "  python create_dataset.py   # Generer le dataset"
echo "  python train.py            # Entrainer le modele"
echo "  python inference.py        # Inference complete"
echo "  python pipeline.py         # Pipeline rapide (Demucs + VoiceFixer)"
