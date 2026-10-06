verify:
	python3 -m src.task
	python3 -m pytest tests/ -q
	python3 -m src.train --epochs 800 --wd 1.0 --frac 0.3 --seed 0
	python3 -m src.realdata || true

full:
	python3 -m src.train --epochs 15000 --wd 1.0 --frac 0.3 --seed 0 --out results/grok_15k.json
	python3 -m src.train --epochs 6000 --wd 0.0 --frac 0.3 --seed 0 --out results/mem_only.json

market:
	python3 -m src.realdata

plots:
	python3 scripts/make_plots.py
