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

# Training on Modal (modal.com); see docs/MODAL_TRAINING.md. The jobs use the local working tree.
MODAL ?= modal
DATASET ?= two-bowl
MODEL_NAME ?= patchcore-sim
STEPS ?= 2000
.PHONY: modal-warm modal-pipeline modal-retrain-detector modal-patchcore modal-record modal-act modal-act-evaluate modal-stop

modal-warm:
	$(MODAL) run cloud/modal_train.py::warm

modal-pipeline:
	$(MODAL) run cloud/modal_train.py::pipeline --dataset $(DATASET) --model-name $(MODEL_NAME)

modal-retrain-detector:
	$(MODAL) run cloud/modal_train.py::pipeline --dataset $(DATASET) --model-name $(MODEL_NAME) --overwrite

modal-patchcore:
	$(MODAL) run cloud/modal_train.py::train_patchcore --dataset $(DATASET) --model-name $(MODEL_NAME)

modal-record:
	$(MODAL) run cloud/modal_train.py::record_policy --demos policy-smoke --episodes 6

modal-act:
	$(MODAL) run cloud/modal_train.py::train_act --demos policy-smoke --steps $(STEPS)

modal-act-evaluate:
	$(MODAL) run cloud/modal_train.py::evaluate_act --run-id $(RUN_ID)

modal-stop:
	$(MODAL) app stop -y brickcell-train || true
	$(MODAL) app list
