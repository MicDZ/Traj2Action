# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# keys
import os
from pathlib import Path

from huggingface_hub.constants import HF_HOME

# # danaaubakirova_koch_test
# OBS_ENV = "observation.environment_state"
# OBS_ROBOT = "observation.state"
# OBS_IMAGE = "observation.image"
# OBS_IMAGES = "observation.images"
# ACTION = "action"

# # ego4d_human_hand_trasform
# OBS_ENV = "observation.environment_state"
# OBS_ROBOT = "state"
# OBS_IMAGE = "image"
# OBS_IMAGES = "image"
# ACTION = "hand_action"
# ACTION_IS_PAD = "hand_action_is_pad"

# libero 
OBS_ENV = "observation.environment_state"
OBS_ROBOT = "state"
OBS_TRAJ = "state_trajectory"
OBS_IMAGE = "image"
OBS_IMAGES = "image"
ACTION = "actions"
TRAJ = "trajectory"
ACTION_IS_PAD = "actions_is_pad"
TRAJ_IS_PAD = "trajectory_is_pad"


# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

# cache dir
default_cache_path = Path(HF_HOME) / "lerobot"
HF_LEROBOT_HOME = Path(os.getenv("HF_LEROBOT_HOME", default_cache_path)).expanduser()

if "LEROBOT_HOME" in os.environ:
    raise ValueError(
        f"You have a 'LEROBOT_HOME' environment variable set to '{os.getenv('LEROBOT_HOME')}'.\n"
        "'LEROBOT_HOME' is deprecated, please use 'HF_LEROBOT_HOME' instead."
    )


# for human hand and robot data keys mapping
DATA_KEYS_MAPPING_HAND = {
    "left_image":"main_image",
    "ego_image":"human_image",
    "right_image":"wrist_image",
    "top_image":"top_image"
}

DATA_KEYS_MAPPING_ROBOT = {
    "left_image":"image",
    "ego_image":"wrist_image",
}
