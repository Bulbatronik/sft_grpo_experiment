MODEL      ?= Qwen/Qwen3-1.7B
# Short name used in paths, e.g. Qwen3-1.7B.
# NOTE: no inline comment here — make keeps the whitespace before a trailing
# comment, which would corrupt every derived path.
MODEL_NAME ?= $(notdir $(MODEL))

# Per-model config files; override with SFT_CONFIG=... or GRPO_CONFIG=...
SFT_CONFIG  ?= $(PWD)/configs/sft_$(MODEL_NAME).yaml
GRPO_CONFIG ?= $(PWD)/configs/grpo_$(MODEL_NAME).yaml

SEED       ?= 42
NPROC      ?= 1

# All artifacts — including seed-dependent data selections — are namespaced by
# (model, seed) so repeated runs never clobber each other. Raw GSM8K parquets
# are tiny and simply re-created per run dir by `make slurm-prepare`.
RUN_DIR    ?= $(MODEL_NAME)/seed$(SEED)
DATA_DIR   ?= $(PWD)/data/$(RUN_DIR)
CKPT_DIR   ?= $(PWD)/checkpoints/$(RUN_DIR)
LOGS_DIR   ?= $(PWD)/logs/$(RUN_DIR)
RESULTS    ?= $(PWD)/results/$(RUN_DIR)

# CPU-only phases use the dataval_env conda environment.
CONDA_PYTHON ?= $(HOME)/.conda/envs/dataval_env/bin/python3

# GPU phases run inside the Singularity container with a writable overlay.
SIF        ?= /home/usemil/orcd/scratch/apptainer/verl.sif
OVERLAY    ?= /home/usemil/orcd/scratch/apptainer/verl_overlay.img
SINGULARITY ?= singularity exec --nv --overlay $(OVERLAY) -B /orcd,/home --env PYTHONNOUSERSITE=1 $(SIF)

.PHONY: all prepare embed sft rollout grpo grpo-dry plot slurm-prepare slurm-embed slurm-sft slurm-rollout slurm-grpo clean help

# ── Direct execution (interactive GPU session) ────────────────────────────────

all: prepare embed sft rollout grpo plot

prepare:
	$(CONDA_PYTHON) scripts/00_prepare_gsm8k.py --seed $(SEED) --data-dir $(DATA_DIR)

embed:
	$(SINGULARITY) python3 scripts/01_embed_and_select_sft.py \
		--seed $(SEED) --data-dir $(DATA_DIR) --results-dir $(RESULTS)

sft:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/02_train_sft.py \
		--config $(SFT_CONFIG) \
		model.name=$(MODEL) \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--checkpoints-dir $(CKPT_DIR)/sft \
		--logs-dir $(LOGS_DIR) \
		--results-dir $(RESULTS)

rollout:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/03_rollout_and_select_grpo.py \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--checkpoints-dir $(CKPT_DIR)/sft \
		--results-dir $(RESULTS) \
		--logs-dir $(LOGS_DIR)

grpo:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/04_train_grpo.py \
		--config $(GRPO_CONFIG) \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--sft-checkpoints-dir $(CKPT_DIR)/sft \
		--grpo-checkpoints-dir $(CKPT_DIR)/grpo \
		--logs-dir $(LOGS_DIR)

grpo-dry:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/04_train_grpo.py \
		--config $(GRPO_CONFIG) \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--sft-checkpoints-dir $(CKPT_DIR)/sft \
		--grpo-checkpoints-dir $(CKPT_DIR)/grpo \
		--logs-dir $(LOGS_DIR) \
		--dry-run

plot:
	$(CONDA_PYTHON) scripts/07_plot_training_curves.py \
		--logs-dir $(LOGS_DIR) \
		--results-dir $(RESULTS)

# ── SLURM submission ─────────────────────────────────────────────────────────
# MODEL, MODEL_NAME, SFT_CONFIG, and GRPO_CONFIG are forwarded as env vars so
# scripts derive their paths and load the right hyperparameter files.
# --output overrides the #SBATCH directive, routing SLURM logs to the
# model-specific subdirectory so the plot script finds them by glob.

slurm-prepare:
	mkdir -p $(LOGS_DIR)
	SEED="$(SEED)" DATA_DIR="$(DATA_DIR)" \
	sbatch --output=$(LOGS_DIR)/%x-%j.out scripts/slurm/submit_prepare.sh

slurm-embed:
	mkdir -p $(LOGS_DIR)
	MODEL="$(MODEL)" MODEL_NAME="$(MODEL_NAME)" SEED="$(SEED)" \
	sbatch --output=$(LOGS_DIR)/%x-%j.out scripts/slurm/submit_embed.sh

slurm-sft:
	mkdir -p $(LOGS_DIR)
	MODEL="$(MODEL)" MODEL_NAME="$(MODEL_NAME)" SEED="$(SEED)" SFT_CONFIG="$(SFT_CONFIG)" \
	sbatch --output=$(LOGS_DIR)/%x-%A_%a.out scripts/slurm/submit_sft.sh

slurm-rollout:
	mkdir -p $(LOGS_DIR)
	MODEL="$(MODEL)" MODEL_NAME="$(MODEL_NAME)" SEED="$(SEED)" \
	sbatch --output=$(LOGS_DIR)/%x-%j.out scripts/slurm/submit_rollout.sh

slurm-grpo:
	mkdir -p $(LOGS_DIR)
	MODEL="$(MODEL)" MODEL_NAME="$(MODEL_NAME)" SEED="$(SEED)" GRPO_CONFIG="$(GRPO_CONFIG)" \
	sbatch --output=$(LOGS_DIR)/%x-%A_%a.out scripts/slurm/submit_grpo.sh

# ── Maintenance ──────────────────────────────────────────────────────────────

clean:
	rm -f $(DATA_DIR)/embeddings.npy $(DATA_DIR)/pca_reduced.npy $(DATA_DIR)/pca_model.pkl
	rm -f $(DATA_DIR)/sft_indices/*.json
	rm -f $(DATA_DIR)/sft_train/*.parquet
	rm -f $(DATA_DIR)/rollouts/*.jsonl $(DATA_DIR)/rollouts/*.json
	rm -f $(DATA_DIR)/grpo_train/*/*.parquet
	rm -f $(RESULTS)/plots/*.png $(RESULTS)/*.json $(RESULTS)/*.csv $(RESULTS)/*.md

help:
	@echo "Usage:  make <target> [MODEL=<hf_model_id>] [SEED=<n>]"
	@echo ""
	@echo "  MODEL defaults to Qwen/Qwen3-1.7B"
	@echo "  SFT_CONFIG and GRPO_CONFIG are auto-derived from MODEL_NAME."
	@echo "  Data, checkpoints, logs, and results are namespaced by"
	@echo "  <MODEL_NAME>/seed<SEED>, so different seeds never clobber."
	@echo ""
	@echo "  Examples:"
	@echo "    make slurm-sft MODEL=Qwen/Qwen3-1.7B"
	@echo "    make slurm-grpo MODEL=Qwen/Qwen3-4B"
	@echo "    make slurm-sft MODEL=Qwen/Qwen3-1.7B SEED=43"
	@echo ""
	@echo "Direct execution targets (interactive GPU session):"
	@echo "  prepare   Phase 0: download GSM8K + write parquets (dataval_env conda)"
	@echo "  embed     Phase 1: embed + PCA + SFT selection (Singularity)"
	@echo "  sft       Phase 2: 4 SFT runs via TRL SFTTrainer (Singularity)"
	@echo "  rollout   Phase 3: rollout scoring + GRPO selection (Singularity)"
	@echo "  grpo      Phase 4: 16 GRPO runs via verl (Singularity)"
	@echo "  grpo-dry  Phase 4 smoke test (--dry-run, 20 steps)"
	@echo "  plot      Phase 5: plot accuracy training curves from logs (dataval_env)"
	@echo "  all       Run all phases end-to-end"
	@echo ""
	@echo "SLURM submission targets (preferred):"
	@echo "  slurm-{prepare,embed,sft,rollout,grpo}"
	@echo "  (grpo uses --array=0-15%%2 for 16 runs, 2 concurrent)"
	@echo ""
	@echo "Variables (override with VAR=val):"
	@echo "  MODEL=$(MODEL)"
	@echo "  MODEL_NAME=$(MODEL_NAME)"
	@echo "  SFT_CONFIG=$(SFT_CONFIG)"
	@echo "  GRPO_CONFIG=$(GRPO_CONFIG)"
	@echo "  SEED=$(SEED)"
	@echo "  RUN_DIR=$(RUN_DIR)"
	@echo "  DATA_DIR=$(DATA_DIR)"
	@echo "  SIF=$(SIF)"
	@echo "  OVERLAY=$(OVERLAY)"
	@echo "  CKPT_DIR=$(CKPT_DIR)"
	@echo "  LOGS_DIR=$(LOGS_DIR)"
	@echo "  RESULTS=$(RESULTS)"
