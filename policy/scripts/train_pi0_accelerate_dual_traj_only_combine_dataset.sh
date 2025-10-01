# user specific setting
PROJECT_DIR=PLEASE_SET_YOUR_OWN_PATH
base_output_dir=PLEASE_SET_YOUR_OWN_PATH
MODELS_DIR=PLEASE_SET_YOUR_OWN_PATH
DATASET_DIRS=PLEASE_SET_YOUR_OWN_PATH

# main training script
cd $PROJECT_DIR
# conda activate $CONDA_DIR
# --multi_gpu
num_processes=8
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch  --num_processes=$num_processes --multi_gpu  --main_process_port=0 \
    lerobot/scripts/train_accelerate_combine_dataset.py  \
    --policy.path=$MODELS_DIR/pi0  \
    --policy.scheduler_decay_steps=400000 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_decay_lr=2.5e-06 \
    --policy.train_expert_only=false \
    --policy.train_trajectory_expert_only=true \
    --policy.load_pretrained_from_pi0=true \
    --policy.shared_expert=false \
    --policy.train_trajectory_state_proj=true \
    --policy.max_traj_state_dim=32 \
    --policy.max_trajectory_dim=32 \
    --policy.traj_sampling_fps=5 \
    --policy.n_trajectory_steps=50 \
    --policy.chunk_size_traj=50 \
    --batch_size=8 \
    --steps=80000 \
    --policy.use_amp=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=realdata_train_from_pi0 \
    --job_name=200k_374hand_50robot_paral_dual \
    --base_output_dir=$base_output_dir \
    --num_workers 16 \
    --dataset.repo_id=$DATASET_DIRS \
    --dataset.mixing_mode=max_size_cycle \
    --dataset.dataset_weights=0.5,0.5 \
    --dataset.normalize_option=true \
    --dataset.visual_keys_used=left_image,ego_image \
    # --policy.type=pi0_dual \