# Puro Training Guide

This guide describes how to run the three-phase Puro-2B training flow with
Puro-Megatron. It covers the environment, the data layout the recipe expects,
how each phase is launched, how checkpoints and resume work, and which
environment variables and extra arguments you can override.

For the feature-level reference (PROM data, MuonHyperball, power schedule), see
[Puro-Megatron extensions](../puro-megatron.md). For the general Megatron
workflow, see the [Quick Start](quickstart.md) and
[Training Examples](training-examples.md).

## Overview

The Puro-2B flow has three phases, launched from one script:

| Recipe | Purpose | Loads from | Saves to |
| --- | --- | --- | --- |
| `phase1-power` | Phase 1 training with the open-ended power LR schedule | none (fresh start) | `SAVE_PATH` |
| `phase2-transition` | Transition into phase 2: linear decay, scheduler and data progress reset | `LOAD_PATH` (phase-1 checkpoint) | `SAVE_PATH` |
| `phase2-constant-continuation` | Constant-LR continuation from a selected phase-2 checkpoint | `LOAD_PATH` at `CKPT_STEP` | `SAVE_PATH` |

The launcher is [`examples/puro/run_puro_2b.sh`](../../examples/puro/run_puro_2b.sh).
It is launcher-neutral: it sets up the arguments and then runs
`pretrain_gpt.py` as a single command, so the distributed launch is handled by
your cluster's launcher (see [Launching](#launching)).

## 1. Environment

Install Puro-Megatron with the CUDA, PyTorch, Transformer Engine, and Apex
setup required by the pinned Megatron-LM release:

```bash
pip install --no-build-isolation .[mlm,dev]
```

Muon and MuonHyperball use NVIDIA NeMo Emerging Optimizers, pinned in
`pyproject.toml` and `uv.lock`.

The recipe uses FP8 with the `blockwise` recipe (`--fp8-format e4m3
--fp8-recipe blockwise`), so run it on GPUs that support FP8 through
Transformer Engine.

## 2. Data

The recipe does not tokenize text. It expects pre-tokenized PROM packed NPY
datasets, with separate training and validation paths:

```bash
export TRAIN_DATA_PATH=/path/to/prom/train
export VALID_DATA_PATH=/path/to/prom/valid
export DATA_CACHE_PATH=/path/to/data_cache
```

The recipe passes these fixed data settings, so the datasets must match them:

- `--tokenizer-type NullTokenizer` with `--vocab-size 151936`;
- `--tiktoken-pattern v2` and `--eos-token-id 151645`;
- `--seq-length 4096`, with attention type `thd` for packed sequences.

`DATA_CACHE_PATH` is passed as `--data-cache-path`, where the data index files
are cached. The repository does not document how to produce PROM NPY files; use
the process that produced your datasets.

## 3. Checkpoints

Set the checkpoint locations:

```bash
export LOAD_PATH=/path/to/load_ckpt   # phase-1 output for phase2-transition, etc.
export SAVE_PATH=/path/to/save_ckpt
```

All phases use the `torch_dist` checkpoint format. Two kinds of checkpoints are
written:

- **Persistent** checkpoints every `--save-interval` iterations (2500 for
  phase 1 and the transition, 500 for constant continuation).
- **Non-persistent** checkpoints every `--non-persistent-save-interval`
  iterations (500 for phase 1 and the transition, 100 for constant
  continuation), with global type.

`--non-persistent-ckpt-num-to-keep` controls how many non-persistent checkpoints
are kept; the constant-continuation recipe keeps 4.

## 4. Launching

The script takes the recipe name as its first argument. Any further arguments
are appended last to `pretrain_gpt.py`:

```bash
examples/puro/run_puro_2b.sh phase1-power
examples/puro/run_puro_2b.sh phase2-transition
examples/puro/run_puro_2b.sh phase2-constant-continuation
```

Unknown recipe names exit with status 2.

Because the script runs `$PYTHON pretrain_gpt.py` as one command, the
distributed launch must come from your cluster's launcher, which sets the
distributed environment (ranks, world size, rendezvous) before the script runs.
The repository does not contain a multi-node launch command for this recipe.
Use the same launcher you use for other Megatron-LM jobs.

## 5. Phase Details

### Phase 1: `phase1-power`

Fresh training with the power schedule:

- `--train-samples 107271866`
- learning rate warmup over 1,536,000 samples, peak `--lr 0.005`, floor
  `--min-lr 0.0005`
- `--lr-decay-style power`, `--lr-power-decay-samples 1536000`,
  `--lr-power-exponent 0.5`
- pipeline parallel size 2 with layout `Et*18|t*10L`
- `--overlap-param-gather`

See [Power LR schedule](../puro-megatron.md#power-lr-schedule) for the formula.

### Phase 2 transition: `phase2-transition`

Continues from a phase-1 checkpoint and switches to linear decay:

- `--train-samples 341867221`
- `--lr 0.00103847632916`, `--min-lr 1e-5`, `--lr-decay-style linear`,
  `--lr-decay-samples 234596053`
- `--override-opt-param-scheduler`: the recipe's schedule replaces the one in
  the checkpoint
- `--reset-opt-param-scheduler-progress` and
  `--reset-train-dataloader-progress`: scheduler and data progress restart at
  the transition
- `--use-checkpoint-args` and `--no-load-rng`
- pipeline parallel size 4 with layout `Et*9|t*9|t*9|tL`
- `--exit-on-missing-checkpoint`: the run fails rather than starting from
  random weights when `LOAD_PATH` has no checkpoint

### Phase 2 constant continuation: `phase2-constant-continuation`

Continues from a chosen phase-2 checkpoint with a constant learning rate:

- `--train-samples 341867221`
- `--lr 4.077248127291725e-05` and `--min-lr` set to the same value, with
  `--lr-decay-style constant`
- `--ckpt-step`: loads the persistent checkpoint at iteration `CKPT_STEP`,
  default `218000`
- `--phase-transition-iterations 69838`: iteration boundary used by the phase
  transition logic; it requires a fixed global batch size across phases and
  does not support batch-size ramp-up
- pipeline parallel size 4 with layout `Et*9|t*9|t*9|tL`
- `--diag-interval 100`, `--save-interval 500`

`--ckpt-step` selects that persistent checkpoint even when a newer
non-persistent checkpoint exists. To continue from another step, set
`CKPT_STEP`.

## 6. Overrides

Environment variables:

| Variable | Required | Default | Purpose |
| --- | --- | --- | --- |
| `TRAIN_DATA_PATH` | yes | | Training PROM NPY data |
| `VALID_DATA_PATH` | yes | | Validation PROM NPY data |
| `DATA_CACHE_PATH` | yes | | Data index cache |
| `LOAD_PATH` | yes | | Checkpoint to load |
| `SAVE_PATH` | yes | | Checkpoint output directory |
| `PYTHON` | no | `python` | Interpreter used to run `pretrain_gpt.py` |
| `PURO_MEGATRON_ROOT` | no | repository root | Repository location |
| `TENSORBOARD_DIR` | no | `tensorboard_logs/puro_<recipe>` | TensorBoard output |
| `CKPT_STEP` | no | `218000` | Step for `phase2-constant-continuation` |
| `MUON_HYPERBALL_RMS` | no | unset | Adds `--muon-hyperball-rms`; unset keeps each matrix's initialization-time radius |

Extra command-line arguments are appended after the recipe arguments, so they
override earlier values. This is the way to shorten a run for a smoke test
(for example by reducing `--train-samples` or the batch sizes).

## 7. Monitoring

- Logs are printed every iteration (`--log-interval 1`), and TensorBoard
  receives the same values (`--tensorboard-log-interval 1`).
- Validation runs every 200 iterations for 5 iterations (`--eval-interval 200
  --eval-iters 5`).
- `--log-throughput` and `--log-timers-to-tensorboard` add throughput and
  timer metrics.
- `--diag-interval` controls the effective-LR diagnostics (1 during the
  transition, 100 in the other phases).

## Known gaps

- The repository does not document how PROM NPY datasets are generated.
- The repository does not contain a multi-node launch command for this recipe.
- The recipe's example paths and cluster settings are placeholders; replace
  them with your own.
