#!/bin/bash
# =====================================================================
#  Préparation de la clé USB de livraison - Stage Maxime D'Angelo 2026
#  Usage : ./preparer_cle_usb.sh /media/maxime/NOM_CLE
# =====================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ---- Destination ----
if [ -z "$1" ]; then
    echo ""
    echo "Usage : $0 /media/maxime/NOM_DE_LA_CLE"
    echo ""
    echo "Clés USB détectées :"
    lsblk -o NAME,SIZE,MOUNTPOINT | grep -E "sd|usb" | grep "/" || echo "  (aucune montée)"
    echo ""
    echo "Monte ta clé USB, puis relance ce script avec le bon chemin."
    exit 1
fi

USB="$1/LIVRAISON_STAGE_DANGELO_2026"
mkdir -p "$USB"

echo ""
echo "============================================================"
echo "  Préparation clé USB → $USB"
echo "============================================================"
echo ""

# ---- Structure ----
mkdir -p "$USB/1_CODE_SOURCE"
mkdir -p "$USB/2_MODELES/sgmse"
mkdir -p "$USB/2_MODELES/whisper_atco2"
mkdir -p "$USB/2_MODELES/whisper_base"
mkdir -p "$USB/3_DOCUMENTATION"
mkdir -p "$USB/4_RAPPORTS_HEBDO"
mkdir -p "$USB/5_PIPELINE_DEPLOIEMENT"

echo "[1/6] Code source..."
cp "$SCRIPT_DIR"/*.py "$USB/1_CODE_SOURCE/"
cp "$SCRIPT_DIR/requirements"*.txt "$USB/1_CODE_SOURCE/" 2>/dev/null || true
cp "$SCRIPT_DIR/README.md"              "$USB/1_CODE_SOURCE/"
cp "$SCRIPT_DIR/DOCUMENTATION_PROJET.md" "$USB/1_CODE_SOURCE/"
# DOCUMENTATION_PROJET.md exclu (fichier interne de configuration)
echo "      Scripts Python, README, DOCUMENTATION_PROJET.md -> OK"

echo "[2/6] Modèles entraînés (peut prendre plusieurs minutes)..."

# SGMSE+ best model
SGMSE="$SCRIPT_DIR/checkpoints/sgmse/best_model.pt"
if [ -f "$SGMSE" ]; then
    echo "      Copie SGMSE+ best_model.pt (1.3 Go)..."
    cp "$SGMSE" "$USB/2_MODELES/sgmse/"
    echo "      SGMSE+ → OK"
else
    echo "      [ABSENT] SGMSE+ best_model.pt non trouvé dans checkpoints/sgmse/"
fi

# Whisper ATCO2 fine-tune
WATCO="$SCRIPT_DIR/checkpoints/whisper_atco2/best"
if [ -d "$WATCO" ]; then
    echo "      Copie Whisper ATCO2 (925 Mo)..."
    cp -r "$WATCO" "$USB/2_MODELES/whisper_atco2/"
    echo "      Whisper ATCO2 → OK"
else
    echo "      [ABSENT] checkpoints/whisper_atco2/best/ non trouvé"
fi

# Whisper small base
WSMALL="$HOME/.cache/whisper/small.pt"
if [ -f "$WSMALL" ]; then
    echo "      Copie Whisper small.pt (462 Mo)..."
    cp "$WSMALL" "$USB/2_MODELES/whisper_base/"
    echo "      Whisper small → OK"
else
    echo "      [ABSENT] ~/.cache/whisper/small.pt non trouvé"
fi

echo "[3/6] Documentation technique..."
cp -r "$SCRIPT_DIR/docs/"*.docx "$USB/3_DOCUMENTATION/" 2>/dev/null || true
cp -r "$SCRIPT_DIR/docs/"*.md   "$USB/3_DOCUMENTATION/" 2>/dev/null || true
echo "      docs/ → OK"

echo "[4/6] Rapports hebdomadaires..."
cp "$SCRIPT_DIR/rapport hebdo/"*.docx "$USB/4_RAPPORTS_HEBDO/" 2>/dev/null || true
echo "      rapport hebdo/ → OK"

echo "[5/6] Package déploiement (scripts + small.pt)..."
cp "$SCRIPT_DIR/pipeline_denoise_transcribe.py" "$USB/5_PIPELINE_DEPLOIEMENT/"
cp "$SCRIPT_DIR/demo_pipeline.py"               "$USB/5_PIPELINE_DEPLOIEMENT/"
cp "$SCRIPT_DIR/generate_demo_figures.py"       "$USB/5_PIPELINE_DEPLOIEMENT/"
cp "$SCRIPT_DIR/requirements_deploy.txt"        "$USB/5_PIPELINE_DEPLOIEMENT/"
if [ -f "$WSMALL" ]; then
    cp "$WSMALL" "$USB/5_PIPELINE_DEPLOIEMENT/"
fi
echo "      Pipeline prêt à déployer → OK"

echo "[6/6] README de livraison..."
cat > "$USB/LIRE_MOI.txt" << 'README'
=====================================================================
  LIVRAISON STAGE - MAXIME D'ANGELO - THALES BELGIUM 2026
  Projet ACE4ACES - Pipeline audio radio aéronautique militaire
=====================================================================

STRUCTURE DE LA CLÉ
-------------------
  1_CODE_SOURCE/          Tous les scripts Python + README + DOCUMENTATION_PROJET.md
  2_MODELES/              Modèles entraînés (.pt / dossier HuggingFace)
    sgmse/                SGMSE+ fine-tuné v5 (débruitage, PESQ 3.31)
    whisper_atco2/        Whisper fine-tuné ATCO2 (transcription, WER 5.3%)
    whisper_base/         Whisper-small base (requis pour inference)
  3_DOCUMENTATION/        Rapports scientifiques (SGMSE+, ASR, TFE)
  4_RAPPORTS_HEBDO/       6 rapports de progression hebdomadaires
  5_PIPELINE_DEPLOIEMENT/ Scripts + modèle prêts à copier sur un autre PC

CODE SOURCE COMPLET (GitHub)
-----------------------------
  https://github.com/MaximeDangelo/AudioSuperResolution
  (repo privé - demander accès à Maxime D'Angelo)

LANCEMENT RAPIDE
----------------
  1. Copier 5_PIPELINE_DEPLOIEMENT/ sur le PC cible
  2. pip install -r requirements_deploy.txt
  3. pip install torch torchaudio  (adapter selon GPU)
  4. mkdir -p ~/.cache/whisper && cp small.pt ~/.cache/whisper/
  5. python demo_pipeline.py mon_audio.flac

RÉSULTATS CLÉS
--------------
  Débruitage SGMSE+ v5   : PESQ 3.31 / STOI 0.91 (validation synthétique)
  Transcription militaire : WER 1.7% (prompt ATC/NATO)
  Transcription ATC civil : WER 5.3% (fine-tune ATCO2, 2000 samples)

CONCLUSION PRINCIPALE
---------------------
  Niveau TRL 3-4 (prototype laboratoire).
  Le bottleneck est le dataset d'entraînement, pas l'architecture.
  Prérequis déploiement : données cockpit réelles + temps réel + DO-178C.

=====================================================================
README

echo "      LIRE_MOI.txt → OK"

# ---- Résumé ----
echo ""
echo "============================================================"
echo "  Clé USB prête !"
echo "============================================================"
echo ""
du -sh "$USB/"*/  2>/dev/null
echo ""
du -sh "$USB"
echo ""
echo "  Chemin : $USB"
echo "  Éjecter la clé : udisksctl power-off -b /dev/sdX"
echo ""
README
