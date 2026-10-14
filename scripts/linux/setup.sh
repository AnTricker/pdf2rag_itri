#!/usr/bin/env bash
set -euo pipefail

TOOLKIT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda is not available in PATH" >&2
  exit 1
fi

for component in core surya marker; do
  env_file="${TOOLKIT_ROOT}/environments/${component}/environment.yml"
  env_name="$(awk '/^name:/ {print $2; exit}' "${env_file}")"
  if conda env list | awk '{print $1}' | grep -Fxq "${env_name}"; then
    echo "[setup] UPDATE ${env_name}"
    conda env update --name "${env_name}" --file "${env_file}" --prune
  else
    echo "[setup] CREATE ${env_name}"
    conda env create --file "${env_file}"
  fi
  conda run -n "${env_name}" python -m pip install \
    -r "${TOOLKIT_ROOT}/environments/${component}/requirements.txt"
done

conda run -n digital-pdf-core python -m pip install -e "${TOOLKIT_ROOT}"
chmod +x "${TOOLKIT_ROOT}"/scripts/linux/*.sh
echo "[setup] DONE"
