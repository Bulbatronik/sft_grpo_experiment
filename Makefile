SEED       ?= 42
NPROC      ?= 1
DATA_DIR   ?= $(PWD)/data
CKPT_DIR   ?= /orcd/scratch/orcd/008/gkim27/gsm8k_selection/checkpoints
LOGS_DIR   ?= $(PWD)/logs
RESULTS    ?= $(PWD)/results

# CPU-only phases use the my_env conda environment.
CONDA_PYTHON ?= $(HOME)/.conda/envs/my_env/bin/python3
# GPU phases run inside the Singularity container (the container has python3 + verl + vllm).
SIF        ?= $(HOME)/verl.sif
SINGULARITY ?= singularity exec --nv -B /orcd $(SIF)

.PHONY: all prepare embed sft rollout grpo eval slurm-prepare slurm-embed slurm-sft slurm-rollout slurm-grpo slurm-eval clean help

# ── Direct execution (login node / interactive GPU session) ──────────────────

all: prepare embed sft rollout grpo eval

prepare:
	$(CONDA_PYTHON) scripts/00_prepare_gsm8k.py --seed $(SEED) --data-dir $(DATA_DIR)

embed:
	$(SINGULARITY) python3 scripts/01_embed_and_select_sft.py \
		--seed $(SEED) --data-dir $(DATA_DIR) --results-dir $(RESULTS)

sft:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/02_train_sft.py \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--checkpoints-dir $(CKPT_DIR)/sft \
		--logs-dir $(LOGS_DIR) \
		--results-dir $(RESULTS) \
		--nproc $(NPROC)

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
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--sft-checkpoints-dir $(CKPT_DIR)/sft \
		--grpo-checkpoints-dir $(CKPT_DIR)/grpo \
		--logs-dir $(LOGS_DIR)

grpo-dry:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/04_train_grpo.py \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--sft-checkpoints-dir $(CKPT_DIR)/sft \
		--grpo-checkpoints-dir $(CKPT_DIR)/grpo \
		--logs-dir $(LOGS_DIR) \
		--dry-run

eval:
	CC=/usr/bin/gcc TRITON_CC=/usr/bin/gcc \
	$(SINGULARITY) python3 scripts/05_evaluate.py \
		--seed $(SEED) \
		--data-dir $(DATA_DIR) \
		--sft-checkpoints-dir $(CKPT_DIR)/sft \
		--grpo-checkpoints-dir $(CKPT_DIR)/grpo \
		--results-dir $(RESULTS) \
		--logs-dir $(LOGS_DIR)

# ── SLURM submission ─────────────────────────────────────────────────────────

slurm-prepare:
	sbatch scripts/slurm/submit_prepare.sh

slurm-embed:
	sbatch scripts/slurm/submit_embed.sh

slurm-sft:
	sbatch scripts/slurm/submit_sft.sh

slurm-rollout:
	sbatch scripts/slurm/submit_rollout.sh

slurm-grpo:
	sbatch scripts/slurm/submit_grpo.sh

slurm-eval:
	sbatch scripts/slurm/submit_eval.sh

# ── Maintenance ──────────────────────────────────────────────────────────────

clean:
	rm -f $(DATA_DIR)/embeddings.npy $(DATA_DIR)/pca_reduced.npy $(DATA_DIR)/pca_model.pkl
	rm -f $(DATA_DIR)/sft_indices/*.json
	rm -f $(DATA_DIR)/sft_train/*.parquet
	rm -f $(DATA_DIR)/rollouts/*.jsonl $(DATA_DIR)/rollouts/*.json
	rm -f $(DATA_DIR)/grpo_train/*/*.parquet
	rm -f $(RESULTS)/plots/*.png $(RESULTS)/*.json $(RESULTS)/*.csv $(RESULTS)/*.md

help:
	@echo "Direct execution targets (interactive GPU session):"
	@echo "  prepare   Phase 0: download GSM8K + write parquets (my_env conda)"
	@echo "  embed     Phase 1: embed + PCA + SFT selection (Singularity)"
	@echo "  sft       Phase 2: 4 SFT runs via verl (Singularity)"
	@echo "  rollout   Phase 3: rollout scoring + GRPO selection (Singularity)"
	@echo "  grpo      Phase 4: 16 GRPO runs via verl (Singularity)"
	@echo "  grpo-dry  Phase 4 smoke test (--dry-run, 20 steps)"
	@echo "  eval      Phase 5: evaluate 21 models (Singularity)"
	@echo "  all       Run all phases end-to-end"
	@echo ""
	@echo "SLURM submission targets:"
	@echo "  slurm-{prepare,embed,sft,rollout,grpo,eval}"
	@echo "  (grpo uses --array=0-15%4 for 16 runs, 4 concurrent)"
	@echo ""
	@echo "Variables (override with VAR=val):"
	@echo "  SEED=$(SEED)  NPROC=$(NPROC)"
	@echo "  SIF=$(SIF)"
	@echo "  CONDA_PYTHON=$(CONDA_PYTHON)"
