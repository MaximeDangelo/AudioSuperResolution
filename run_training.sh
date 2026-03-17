#!/bin/bash
# Script d'entrainement complet - a lancer demain
# Usage: source venv/bin/activate && bash run_training.sh

set -e

echo "============================================"
echo "  PIPELINE D'ENTRAINEMENT COMPLET"
echo "============================================"
echo ""

# Etape 1: Regenerer le dataset
echo "=== Etape 1/2: Generation du dataset ==="
python3 create_dataset.py
echo ""

# Etape 2: Entrainement
echo "=== Etape 2/2: Entrainement SpectralResUNet ==="
python3 train.py
echo ""

echo "============================================"
echo "  ENTRAINEMENT TERMINE"
echo "  Checkpoint: checkpoints/best_model.pt"
echo "============================================"
