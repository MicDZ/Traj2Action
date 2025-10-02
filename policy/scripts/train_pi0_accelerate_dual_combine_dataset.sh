######### user specific setting  ################
# Please change the following paths according to your own setup
##################################################
PROJECT_DIR=PLEASE_SET_YOUR_OWN_PATH_TO_TRAJ2ACTION_REPO/policy
BASE_OUTPUT_DIR=$PROJECT_DIR/outputs
DATASET_DIRS=PLEASE_SET_YOUR_OWN_PATH_TO_HAND_DATASET,PLEASE_SET_YOUR_OWN_PATH_TO_ROBOT_DATASET
MODELS_DIR=PLEASE_SET_YOUR_OWN_PATH_TO_PRETRAINED_MODELS
GPU_NUM=1
##################################################
export TOKENIZERS_PARALLELISM=false

# check if all the directories exist
if [ ! -d "$PROJECT_DIR" ]; then
  echo "Error: PROJECT_DIR $PROJECT_DIR does not exist."
  exit 1
fi
if [ ! -d "$MODELS_DIR" ]; then
  echo "Error: MODELS_DIR $MODELS_DIR does not exist."
  exit 1
fi
if [ ! -d "$BASE_OUTPUT_DIR" ]; then
  echo "Output directory $BASE_OUTPUT_DIR does not exist. Creating it."
  mkdir -p $BASE_OUTPUT_DIR
fi
IFS=',' read -r -a dataset_dirs_array <<< "$DATASET_DIRS"
for dir in "${dataset_dirs_array[@]}"; do
  if [ ! -d "$dir" ]; then
    echo "Error: One of the DATASET_DIRS $dir does not exist."
    exit 1
  fi
done

# check if GPU_NUM is a positive integer
if ! [[ "$GPU_NUM" =~ ^[1-9][0-9]*$ ]]; then
  echo "Error: GPU_NUM must be a positive integer."
  exit 1
fi

# main training script
cd $PROJECT_DIR


#CUDA_VISIBLE_DEVICES=7 
accelerate launch  --num_processes=$GPU_NUM --main_process_port=0 \
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
    --base_output_dir=$BASE_OUTPUT_DIR \
    --num_workers 8 \
    --dataset.repo_id=$DATASET_DIRS \
    --dataset.mixing_mode=max_size_cycle \
    --dataset.dataset_weights=0.5,0.5 \
    --dataset.normalize_option=true \
    --dataset.visual_keys_used=left_image,ego_image \
    --policy.randmask_traj2action_prob=0.2 \

# make sure that fps in traj_sampling_fps aligns with repo_ids