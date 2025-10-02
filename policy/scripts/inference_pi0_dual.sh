##################################################
# Please change the following constants according to your own setup
##################################################
PROJECT_DIR=PLEASE_SET_YOUR_OWN_PATH_TO_TRAJ2ACTION_REPO/policy
MODEL_DIR=PLEASE_SET_YOUR_OWN_PATH_TO_PRETRAINED_MODELS
PORT=5000
HOST=0.0.0.0
##################################################

cd $PROJECT_DIR

CUDA_VISIBLE_DEVICES=0 python lerobot/scripts/serve_torch_policy.py \
    --policy.path=$MODEL_DIR \
    --env.type=libero \
    --eval.batch_size=1 \
    --eval.n_episodes=10 \
    --policy.device=cuda \
    --port=$PORT \
    --host=$HOST 