# Traj2Action: Policy Training and Serving Codebase

Traj2Action is a PyTorch-based codebase for learning and serving robot policies. It provides:
- Offline training on LeRobot-style datasets (single or combined datasets)
- PI0-style policy variants with trajectory prediction and action prediction
- Evaluation in LIBERO simulator and a WebSocket policy server compatible with OpenPI clients
- Multi-GPU training with Accelerate
  
This document shows how to set up the environment, train, and evaluate with the scripts included in this repository.

## Quick start

### Install dependencies

1. Create and activate environment (recommended)

```bash
conda env create -f policy/environment.yml python=3.11
conda activate traj2action
```

2. (Optional) Configure Accelerate for multi-GPU

```bash
accelerate config
```


### Training with dual datasets (hand + robot)

Open `Traj2Action/policy/scripts/train_pi0_accelerate_dual_combine_dataset.sh` and replace absolute paths with your own.

* PROJECT_DIR: root directory of policy directory (eg. `/path/to/Traj2Action/policy`)
* BASE_OUTPUT_DIR: base output directory for training logs and checkpoints
* DATASET_DIRS: comma-separated list of dataset directories (eg. `/path/to/hand_dataset,/path/to/robot_dataset`)
* MODELS_DIR: directory containing pi0 pretrained models (eg. `/path/to/pretrained_models`), you can download from [here](https://huggingface.co/https://huggingface.co/lerobot/pi0_base)
* GPU_NUM: number of GPUs to use for training

```bash
bash policy/scripts/train_pi0_accelerate_dual_combine_dataset.sh
```
### Serving the trained policy (WebSocket)


## Tips and troubleshooting

- Replace all absolute paths in the scripts with your own paths.
- If you see an assertion about `n_trajectory_steps` vs `chunk_size_traj`, set them to the same value.
- On CPU or lower VRAM GPUs, use smaller `--batch_size`, fewer `--num_processes`, and keep `--policy.use_amp=false`.
- For W&B offline usage, pass `--wandb.mode=offline`. Later you can sync using `wandb sync /path/to/offline-run`.
