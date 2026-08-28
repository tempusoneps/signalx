#!/usr/bin/env bash

# Exit on error, pipe failures, and unset variables
set -euo pipefail

# Determine script & project directories to ensure relative path works from any working directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
DATASET_DIR="${PROJECT_DIR}/datasets"

# Default dataset paths (output to datasets/ directory)
OHLCV_DATASET="${OHLCV_DATASET:-${DATASET_DIR}/VN30F1M_5m.csv}"
SIGNALS_DATASET="${SIGNALS_DATASET:-${DATASET_DIR}/VN30F1M_5m_signals.parquet}"
OHLCV_URL="https://raw.githubusercontent.com/tempusoneps/vn-stock-data/refs/heads/main/VN30F1M/data_ohlcv/VN30F1M_5m.csv"

FORCE=false
GENERATE_SIGNALS=false

# Parse command line options
while [[ $# -gt 0 ]]; do
  case "$1" in
    -f|--force)
      FORCE=true
      shift
      ;;
    -g|--generate)
      GENERATE_SIGNALS=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo "Download VN30F1M 5-minute OHLCV dataset from GitHub into datasets/ directory."
      echo ""
      echo "Options:"
      echo "  -f, --force       Force re-download even if dataset file already exists"
      echo "  -g, --generate    Automatically run 'signalx generate' after downloading"
      echo "  -h, --help        Display this help message"
      exit 0
      ;;
    *)
      echo "[ERROR] Invalid option: $1" >&2
      echo "Run '$0 --help' for available options." >&2
      exit 1
      ;;
  esac
done

cd "${PROJECT_DIR}"

echo "=== START VN30F1M DATASET PREPARATION ==="
echo "[INFO] Target dataset directory: ${DATASET_DIR}"

# 0. Ensure datasets/ directory exists
mkdir -p "${DATASET_DIR}"

# 1. Check & Download OHLCV_DATASET (CSV)
if [ "$FORCE" = true ] || [ ! -f "$OHLCV_DATASET" ]; then
  echo "[INFO] Downloading VN30F1M 5m dataset from repository..."
  if command -v curl &>/dev/null; then
    curl -fsSL -o "$OHLCV_DATASET" "$OHLCV_URL"
  elif command -v wget &>/dev/null; then
    wget -q -O "$OHLCV_DATASET" "$OHLCV_URL"
  else
    echo "[ERROR] curl or wget is required to download dataset." >&2
    exit 1
  fi
  echo "[PASS] Successfully downloaded OHLCV dataset: ${OHLCV_DATASET}"
else
  echo "[SKIP] OHLCV dataset already exists: ${OHLCV_DATASET} (use --force to re-download)"
fi

# 2. Optional: Generate trading signals via SignalX CLI
if [ "$GENERATE_SIGNALS" = true ]; then
  echo ""
  echo "[INFO] Generating 114 trading signals -> ${SIGNALS_DATASET}..."
  uv run signalx generate "${OHLCV_DATASET}" -o "${SIGNALS_DATASET}" --stats-report
  echo "[PASS] Signals generated successfully: ${SIGNALS_DATASET}"
fi

echo "=== PREPARATION COMPLETE ==="
