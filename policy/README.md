# Traj2Action: Policy Training and Serving Codebase

Traj2Action is a PyTorch-based codebase for learning and serving robot policies. It provides:
- Offline training on LeRobot-style datasets (single or combined datasets)
- PI0-style policy variants with trajectory prediction and action prediction
- Evaluation in LIBERO simulator and a WebSocket policy server compatible with OpenPI clients

This document shows how to set up the environment, train, and evaluate with the scripts included in this repository.

## Quick start

1. Create and activate environment (recommended)

```bash
conda env create -f policy/environment.yml python=3.11
conda activate traj2action
```

2. (Optional) Configure Accelerate for multi-GPU

```bash
accelerate config
```

3. Train with Accelerate (edit paths for your machine)

```bash
accelerate launch --num_processes=4 --multi_gpu \
   traj2action/lerobot/scripts/train_accelerate.py \
  --policy.path=/path/to/pretrained_model \
  --dataset.repo_id=/path/to/lerobot_dataset \
  --policy.use_amp=false \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=my_project \
  --base_output_dir=/path/to/outputs \
  --job_name=my_job \
  --batch_size=8 \
  --steps=200000
```

4. Serve the trained policy (WebSocket)

```bash
python  traj2action/lerobot/scripts/serve_torch_policy.py \
  --env.type=libero \
  --policy.path=/path/to/pretrained_model \
  --host 0.0.0.0 \
  --port 8000
```

5. Evaluate in LIBERO via client

```bash
python  traj2action/lerobot/scripts/eval_libero.py \
  --env.type=libero \
  --eval.task_suite_name=libero_spatial \
  --eval.n_episodes=5 \
  --host 127.0.0.1 \
  --port 8000 \
  --output_dir outputs/eval/libero
```

## Installation

- Python environment: use ` traj2action/environment.yml` or ` traj2action/requirements.txt`.
- On macOS without NVIDIA GPUs, you can still run CPU-only for debugging. Keep `--policy.use_amp=false`, use `--num_processes=1`, and small `--batch_size`.
- Optional quality-of-life:
  - Better exceptions: `export BETTER_EXCEPTIONS=1`
  - Weights & Biases offline: `export WANDB_BASE_URL=https://api.bandw.top` and pass `--wandb.mode=offline`.

## Datasets

Training expects LeRobot-style datasets on disk. Point to a dataset folder with `--dataset.repo_id=/path/to/dataset`.

You can also provide multiple datasets by passing a comma-separated list:

```bash
--dataset.repo_id=/path/to/dataset_hand,/path/to/dataset_robot \
--dataset.mixing_mode=max_size_cycle \
--dataset.dataset_weights=0.5,0.5 \
--dataset.normalize_option=true \
--dataset.visual_keys_used=left_image,ego_image
```

Notes:
- `dataset.visual_keys_used` is used to remap camera keys to a unified set during loading.
- If you combine datasets, also set `--policy.traj_sampling_fps` as a comma-separated list so each dataset has a corresponding FPS (see below).

## Training

Main entry points:
- Single/standard training: ` traj2action/lerobot/scripts/train_accelerate.py`
- Combined-dataset training: ` traj2action/lerobot/scripts/train_accelerate_combine_dataset.py`

There is a ready-to-use shell example at ` traj2action/training_scripts/train_pi0_accelerate.sh`. Replace absolute paths with your own.

### Example: single dataset, multi-GPU

```bash
accelerate launch --num_processes=4 --multi_gpu \
   traj2action/lerobot/scripts/train_accelerate.py \
  --policy.path=/path/to/pretrained_model \
  --dataset.repo_id=/path/to/lerobot/libero \
  --policy.use_amp=false \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=libero_train \
  --base_output_dir=/path/to/outputs \
  --job_name=libero_pi0 \
  --batch_size=8 \
  --steps=200000 \
  --num_workers=8
```

### Example: combined datasets (hand + robot)

```bash
accelerate launch --num_processes=4 --multi_gpu \
   traj2action/lerobot/scripts/train_accelerate_combine_dataset.py \
  --policy.path=/path/to/pi0 \
  --policy.optimizer_lr=1e-4 \
  --policy.scheduler_decay_steps=160000 \
  --policy.scheduler_decay_lr=2.5e-06 \
  --policy.load_pretrained_from_pi0=true \
  --policy.train_trajectory_state_proj=true \
  --policy.max_traj_state_dim=32 \
  --policy.max_trajectory_dim=32 \
  --policy.n_trajectory_steps=50 \
  --policy.chunk_size_traj=50 \
  --policy.traj_sampling_fps=15,5 \
  --dataset.repo_id=/path/to/hand_ds,/path/to/robot_ds \
  --dataset.mixing_mode=max_size_cycle \
  --dataset.dataset_weights=0.5,0.5 \
  --dataset.normalize_option=true \
  --dataset.visual_keys_used=left_image,ego_image \
  --policy.randmask_traj2action_prob=0.2 \
  --batch_size=16 \
  --steps=200000 \
  --save_freq=20000 \
  --base_output_dir=/path/to/outputs \
  --job_name=dual_dataset_train \
  --wandb.enable=true \
  --wandb.mode=offline \
  --wandb.project=realdata_train_from_pi0 \
  --num_workers=8
```

Important training flags (from configs):

- Dataset
  - `--dataset.repo_id`: path or comma-separated paths. Strings are auto-split into lists when needed.
  - `--dataset.visual_keys_used`: list or comma-separated string of visual keys to keep/remap.
  - `--dataset.normalize_option`: whether to normalize features during loading.
  - `--dataset.dataset_weights`: sampling weights per dataset when combining datasets.

- Policy
  - `--policy.path`: path to a folder containing pretrained policy config/weights (Hugging Face-style `config.json` layout). The training will load the policy config and set `pretrained_path` accordingly.
  - `--policy.use_amp`: enable Automatic Mixed Precision (AMP). Keep false if running on CPU.
  - Trajectory settings: `--policy.n_trajectory_steps` must equal `--policy.chunk_size_traj` (asserted). Use `--policy.traj_sampling_fps=<int>` or a comma list matching the number of datasets.
  - Finetuning knobs: `--policy.load_pretrained_from_pi0`, `--policy.optimizer_lr`, `--policy.scheduler_decay_steps`, `--policy.scheduler_decay_lr`, etc.

- Logging/outputs
  - `--base_output_dir` and `--job_name` define the run directory. A dated folder will be created automatically.
  - `--save_freq`, `--log_freq`, `--eval_freq` control checkpointing, logging, and evaluation cadence.
  - `--wandb.enable`, `--wandb.mode=[online|offline|disabled]`, `--wandb.project`, `--wandb.entity`.

Resume training:
- Pass `--resume=true` and `--config_path=/path/to/train_config.json` from a previous run's checkpoint directory. The code will restore the optimizer/scheduler and continue training.

## Evaluation

Two ways to evaluate:

1) WebSocket server + client (recommended for LIBERO evaluation)

- Start server (loads the policy and exposes a WebSocket API compatible with `openpi_client`):

```bash
python  traj2action/lerobot/scripts/serve_torch_policy.py \
  --env.type=libero \
  --policy.path=/path/to/pretrained_model \
  --host 0.0.0.0 \
  --port 8000
```

- Run LIBERO evaluation client, which connects to the server and rolls out tasks, saving videos and computing success rate:

```bash
python  traj2action/lerobot/scripts/eval_libero.py \
  --env.type=libero \
  --eval.task_suite_name=libero_spatial \
  --eval.n_episodes=5 \
  --host 127.0.0.1 \
  --port 8000 \
  --output_dir outputs/eval/libero
```

Task suites available for LIBERO (choose one for `--eval.task_suite_name`): `libero_spatial`, `libero_object`, `libero_10`, `libero_goal`.

2) In-training evaluation (for custom loops)

- The script ` traj2action/lerobot/scripts/train_ego.py` contains an example of invoking evaluation inside the training loop (see `eval_libero_in_training.py`). This is useful if you want periodic validation while training on simulation data.

## Key files and where to look

- Training entry points
  - ` traj2action/lerobot/scripts/train_accelerate.py` (single dataset)
  - ` traj2action/lerobot/scripts/train_accelerate_combine_dataset.py` (combined datasets)
- Evaluation
  - ` traj2action/lerobot/scripts/eval_libero.py` (client)
  - ` traj2action/lerobot/scripts/serve_torch_policy.py` (WebSocket server)
- Configs
  - ` traj2action/lerobot/configs/train.py` (TrainPipelineConfig and validation)
  - ` traj2action/lerobot/configs/default.py` (DatasetConfig, EvalConfig, WandBConfig)
  - ` traj2action/lerobot/configs/policies.py` (PreTrainedConfig and policy flags)
  - ` traj2action/lerobot/common/envs/configs.py` (env feature maps; LIBERO variants)
- Dataset factory and transforms
  - ` traj2action/lerobot/common/datasets/factory.py` (delta timestamps, trajectory sampling, combined dataset)

## Tips and troubleshooting

- Replace all absolute paths from sample scripts with paths on your machine: dataset locations, pretrained checkpoints, and `--base_output_dir`.
- If you see an assertion about `n_trajectory_steps` vs `chunk_size_traj`, set them to the same value.
- On CPU or lower VRAM GPUs, use smaller `--batch_size`, fewer `--num_processes`, and keep `--policy.use_amp=false`.
- For W&B offline usage, pass `--wandb.mode=offline`. Later you can sync using `wandb sync /path/to/offline-run`.

## License

This project includes components derived from the Hugging Face LeRobot codebase. Refer to original licenses where applicable.
