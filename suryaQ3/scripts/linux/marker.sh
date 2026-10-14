#!/usr/bin/env bash
set -euo pipefail
conda run --no-capture-output -n digital-pdf-core python -m digital_pdf_toolkit marker "$@"
