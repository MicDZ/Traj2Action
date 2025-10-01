######### user specific setting  ################
PROJECT_DIR=/storage/qiguojunLab/zhouhan/codes/humanpi
base_output_dir=/storage/qiguojunLab/zhouhan/codes/humanpi/outputs
DATASET_DIRS=/storage/qiguojunLab/zhouhan/dataset/droid_100/finetune_lerobot/hand_0904_pickup_bottle,/storage/qiguojunLab/zhouhan/dataset/droid_100/finetune_lerobot/pick_up_water_bottle_200
MODELS_DIR=/storage/qiguojunLab/zhouhan/models
##################################################
export TOKENIZERS_PARALLELISM=false

# main training script
cd $PROJECT_DIR

num_processes=8
#CUDA_VISIBLE_DEVICES=7 
accelerate launch  --num_processes=$num_processes --multi_gpu --main_process_port=0 \
    lerobot/scripts/train_accelerate_combine_dataset.py  \
    --policy.path=$MODELS_DIR/pi0  \
    --policy.scheduler_decay_steps=160000 \
    --policy.optimizer_lr=1e-4 \
    --policy.scheduler_decay_lr=2.5e-06 \
    --policy.train_expert_only=false \
    --policy.train_trajectory_expert_only=false \
    --policy.load_pretrained_from_pi0=true \
    --policy.shared_expert=false \
    --policy.train_trajectory_state_proj=true \
    --policy.max_traj_state_dim=32 \
    --policy.max_trajectory_dim=32 \
    --policy.traj_sampling_fps=15,5 \
    --policy.n_trajectory_steps=50 \
    --policy.chunk_size_traj=50 \
    --batch_size=8 \
    --steps=200000 \
    --save_freq=20000 \
    --policy.use_amp=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=traj2action \
    --job_name=demo_job \
    --base_output_dir=$base_output_dir \
    --num_workers 8 \
    --dataset.repo_id=$DATASET_DIRS \
    --dataset.mixing_mode=max_size_cycle \
    --dataset.dataset_weights=0.5,0.5 \
    --dataset.normalize_option=true \
    --dataset.visual_keys_used=left_image,ego_image \
    --policy.randmask_traj2action_prob=0.2 \

# make sure that fps in traj_sampling_fps aligns with repo_ids