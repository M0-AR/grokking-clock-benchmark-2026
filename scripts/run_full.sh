#!/bin/bash
set -e
cd "$(dirname "$0")/.."
export PYTHONPATH=/work:${PYTHONPATH:-}:.
python3 -m src.task
python3 -c "from src.model import TinyClockTransformer; m=TinyClockTransformer(); print(m.count_params())"
mkdir -p results
echo "quick verify: 800-epoch wd=1 vs wd=0"
python3 -m src.train --epochs 800 --wd 1.0 --frac 0.3 --seed 0 2>&1 | tail -5
python3 -m src.train --epochs 800 --wd 0.0 --frac 0.3 --seed 0 2>&1 | tail -5
