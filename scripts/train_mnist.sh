#!/usr/bin/env bash
set -euo pipefail
python -m training.train --config configs/mnist.yaml "$@"
