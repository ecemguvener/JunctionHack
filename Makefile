.PHONY: test check build policy-env policy-record policy-train policy-evaluate

test:
	uv run pytest

check:
	uv run ruff check src tests scripts policy
	uv run ruff format --check src tests scripts policy

build:
	uv build --out-dir artifacts/dist

policy-env:
	uv sync --project policy --locked --python 3.12

policy-record:
	PYTHONPATH=src MENAGERIE_CACHE_DIR="$(CURDIR)/.cache/menagerie" artifacts/mujoco-venv/bin/python -m brickcell.simulation.record_policy

policy-train:
	PYTHONPATH=src TORCH_HOME="$(CURDIR)/.cache/torch" policy/.venv/bin/python policy/train_act.py

policy-evaluate:
	PYTHONPATH=src policy/.venv/bin/python policy/evaluate_act.py
