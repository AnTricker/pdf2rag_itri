#!/usr/bin/env bash
set -euo pipefail

TOOLKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${TOOLKIT_ROOT}/environments/embedding/environment.yml"
ENV_NAME="digital-pdf-embedding"

if [[ -z "${PYTORCH_ROCM_INDEX_URL:-}" ]]; then
  echo "ERROR: set PYTORCH_ROCM_INDEX_URL to the ROCm wheel index supported by this host" >&2
  echo "Example: https://download.pytorch.org/whl/rocm6.4" >&2
  exit 1
fi

if conda env list | awk '{print $1}' | grep -Fxq "${ENV_NAME}"; then
  conda env update --name "${ENV_NAME}" --file "${ENV_FILE}" --prune
else
  conda env create --file "${ENV_FILE}"
fi
conda run -n "${ENV_NAME}" python -m pip install torch torchvision --index-url "${PYTORCH_ROCM_INDEX_URL}"
conda run -n "${ENV_NAME}" python -m pip install -r "${TOOLKIT_ROOT}/environments/embedding/requirements.txt"
conda run -n "${ENV_NAME}" python -m pip install -e "${TOOLKIT_ROOT}"

