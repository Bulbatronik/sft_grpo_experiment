# Installation Layout

Where things live on [MIT ORCD Engaging](https://orcd-docs.mit.edu/). Replace `<username>` with your username.

## Filesystem map

| Path | Filesystem | Contents |
|------|-----------|----------|
| `/home/<username>/orcd/scratch/sft_grpo_experiment` | [Scratch (1 TB flash)](https://orcd-docs.mit.edu/filesystems-file-transfer/filesystems/) | This repo (code, data, results) |
| `/home/<username>/orcd/scratch/apptainer/verl.sif` | Scratch | Base container image ([verlai/verl:vllm011.latest](https://hub.docker.com/r/verlai/verl/tags)) |
| `/home/<username>/orcd/scratch/apptainer/verl_overlay.img` | Scratch | Writable overlay holding `verl` + extra pip packages |
| `/home/<username>/orcd/scratch/apptainer/{cache,tmp}` | Scratch | Apptainer pull cache + mount tmp (kept off home) |
| `/home/<username>/orcd/scratch/verl` | Scratch | [verl source](https://github.com/volcengine/verl) (editable `pip install`) |
| `/home/<username>/.conda/envs/dataval_env` | Home |  Env for Phase 0 (data prep) |

> `/home/<username>/orcd/scratch` is a symlink to `/orcd/scratch/orcd/<group_id>/<username>` — both `/orcd` and `/home` must be bound when running the container.

## Persistent shell env

In `~/.bashrc`:

```bash
export APPTAINER_CACHEDIR=/home/<username>/orcd/scratch/apptainer/cache
export APPTAINER_TMPDIR=/home/<username>/orcd/scratch/apptainer/tmp
```

## Container invocation pattern

All GPU phases use:

```bash
module load apptainer/1.4.2
apptainer exec --nv \
    --overlay /home/<username>/orcd/scratch/apptainer/verl_overlay.img \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    /home/<username>/orcd/scratch/apptainer/verl.sif \
    python3 ...
```

- `--overlay` — adds the writable layer (verl + extra pip packages) on top of the read-only SIF
- `-B /orcd,/home` — binds the real scratch root and home so symlinks resolve inside the container
- `--env PYTHONNOUSERSITE=1` — ignores `~/.local` to prevent host packages from leaking into the container
- `--nv` — passes NVIDIA drivers into the container

## Verified stack

`verl 0.8.0.dev` · `vllm 0.11.0` · `torch 2.8.0+cu128` · `sentence-transformers` · `scikit-learn` · `matplotlib` · `tensorboard`

## Step-by-step setup (from scratch)

Run on a [compute node](https://orcd-docs.mit.edu/running-jobs/requesting-resources/) (`salloc -t 1:00:00 -p mit_normal_gpu -G 1 -c 4 --mem=32G`).

### 1. Conda env for Phase 0 (data prep)

```bash
module load miniforge
conda create -n dataval_env python=3.11 -y
conda activate dataval_env
pip install datasets pyarrow tqdm
```

### 2. Redirect apptainer cache to scratch

```bash
mkdir -p /home/<username>/orcd/scratch/apptainer/{cache,tmp}
export APPTAINER_CACHEDIR=/home/<username>/orcd/scratch/apptainer/cache
export APPTAINER_TMPDIR=/home/<username>/orcd/scratch/apptainer/tmp

cat >> ~/.bashrc << 'EOF'

export APPTAINER_CACHEDIR=/home/<username>/orcd/scratch/apptainer/cache
export APPTAINER_TMPDIR=/home/<username>/orcd/scratch/apptainer/tmp
EOF
```

### 3. Pull the base SIF

```bash
module load apptainer/1.4.2
singularity pull /home/<username>/orcd/scratch/apptainer/verl.sif \
    docker://verlai/verl:vllm011.latest
```

### 4. Clone verl source

```bash
cd /home/<username>/orcd/scratch
git clone https://github.com/volcengine/verl.git
```

### 5. Create writable overlay

```bash
apptainer overlay create --size 8192 /home/<username>/orcd/scratch/apptainer/verl_overlay.img
```

### 6. Install verl + extra packages into the overlay

```bash
cd /home/<username>/orcd/scratch/verl
apptainer exec --nv \
    --overlay /home/<username>/orcd/scratch/apptainer/verl_overlay.img \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    /home/<username>/orcd/scratch/apptainer/verl.sif \
    bash -c "pip3 install --no-deps -e . && pip3 install --no-deps scikit-learn joblib threadpoolctl sentence-transformers"
```

> `--no-deps` is required: pip would otherwise try to upgrade CUDA libs (e.g. `nvidia-nccl-cu12`) that live in the read-only SIF and fail.

### 7. Verify

```bash
apptainer exec --nv \
    --overlay /home/<username>/orcd/scratch/apptainer/verl_overlay.img \
    -B /orcd,/home \
    --env PYTHONNOUSERSITE=1 \
    /home/<username>/orcd/scratch/apptainer/verl.sif \
    python3 -c "import verl, vllm, torch, sentence_transformers, sklearn; print('verl:', verl.__version__, '| vllm:', vllm.__version__, '| torch:', torch.__version__)"
```

## Job submission

Use the [`mit_normal_gpu`](https://orcd-docs.mit.edu/running-jobs/requesting-resources/) partition for GPU phases and `mit_normal` for the data prep phase. See [`scripts/slurm/`](scripts/slurm) for ready-to-submit SLURM scripts.
