#!/usr/bin/env bash
# ==============================================================================
# SignalX: Sample Signal Generation Runner
# Demonstrates end-to-end signal extraction via the SignalX CLI
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

INPUT_FILE="datasets/sample_ohlcv.parquet"
OUTPUT_FILE="datasets/sample_signals.parquet"

# Ensure dataset exists, otherwise create it
if [ ! -f "${INPUT_FILE}" ]; then
    echo "Sample dataset not found. Generating sample dataset..."
    uv run python scripts/prepare_sample_dataset.py
fi

echo "================================================================="
echo "Running SignalX Generation Pipeline"
echo "Input:  ${INPUT_FILE}"
echo "Output: ${OUTPUT_FILE}"
echo "================================================================="

uv run signalx generate "${INPUT_FILE}" -o "${OUTPUT_FILE}" --stats-report

echo ""
echo "================================================================="
echo "Inspecting Signal Output Statistics"
echo "================================================================="
uv run signalx stats "${OUTPUT_FILE}"

echo ""
echo "Pipeline execution completed successfully!"
