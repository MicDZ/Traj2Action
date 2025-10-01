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

import abc
from dataclasses import dataclass, field

import draccus

from lerobot.common.constants import ACTION, OBS_ENV, OBS_IMAGE, OBS_IMAGES, OBS_ROBOT
from lerobot.configs.types import FeatureType, PolicyFeature


@dataclass
class EnvConfig(draccus.ChoiceRegistry, abc.ABC):
    task: str | None = None
    fps: int = 30
    features: dict[str, PolicyFeature] = field(default_factory=dict)
    features_map: dict[str, str] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractproperty
    def gym_kwargs(self) -> dict:
        raise NotImplementedError()


@EnvConfig.register_subclass("aloha")
@dataclass
class AlohaEnv(EnvConfig):
    task: str = "AlohaInsertion-v0"
    fps: int = 50
    episode_length: int = 400
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "top": f"{OBS_IMAGE}.top",
            "pixels/top": f"{OBS_IMAGES}.top",
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels":
            self.features["top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))
        elif self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(14,))
            self.features["pixels/top"] = PolicyFeature(type=FeatureType.VISUAL, shape=(480, 640, 3))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("pusht")
@dataclass
class PushtEnv(EnvConfig):
    task: str = "PushT-v0"
    fps: int = 10
    episode_length: int = 300
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,)),
            "agent_pos": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "environment_state": OBS_ENV,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(384, 384, 3))
        elif self.obs_type == "environment_state_agent_pos":
            self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }


@EnvConfig.register_subclass("xarm")
@dataclass
class XarmEnv(EnvConfig):
    task: str = "XarmLift-v0"
    fps: int = 15
    episode_length: int = 200
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 384
    visualization_height: int = 384
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(4,)),
            "pixels": PolicyFeature(type=FeatureType.VISUAL, shape=(84, 84, 3)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "action": ACTION,
            "agent_pos": OBS_ROBOT,
            "pixels": OBS_IMAGE,
        }
    )

    def __post_init__(self):
        if self.obs_type == "pixels_agent_pos":
            self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length,
        }



@EnvConfig.register_subclass("libero")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero"
    fps: int = 10
    episode_length: int = 320
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 224
    visualization_height: int = 224
    max_steps: int = 320
    # BUG here
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(8, )),
            "image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "actions": ACTION,
            "state": OBS_ROBOT,
            "image": OBS_IMAGE,
            # BUG here: may be different from the key in training process
            "wrist_image": "wrist_image",
        }
    )
    
    def __post_init__(self):
        pass

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length
        }


@EnvConfig.register_subclass("libero_object")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_object"
    fps: int = 10
    episode_length: int = 480
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 224
    visualization_height: int = 224
    max_steps: int = 480
    # BUG here
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(8, )),
            "image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "actions": ACTION,
            "state": OBS_ROBOT,
            "image": OBS_IMAGE,
            # BUG here: may be different from the key in training process
            "wrist_image": "wrist_image",
        }
    )
    
    def __post_init__(self):
        pass

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length
        }

@EnvConfig.register_subclass("libero_10")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_10"
    fps: int = 10
    episode_length: int = 1024
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 224
    visualization_height: int = 224
    max_steps: int = 1024
    # BUG here
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(8, )),
            "image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "actions": ACTION,
            "state": OBS_ROBOT,
            "image": OBS_IMAGE,
            # BUG here: may be different from the key in training process
            "wrist_image": "wrist_image",
        }
    )
    
    def __post_init__(self):
        pass

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length
        }

@EnvConfig.register_subclass("libero_spatial")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_spatial"
    fps: int = 10
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 224
    visualization_height: int = 224
    max_steps: int = 520
    # BUG here
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(8, )),
            "image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "actions": ACTION,
            "state": OBS_ROBOT,
            "image": OBS_IMAGE,
            # BUG here: may be different from the key in training process
            "wrist_image": "wrist_image",
        }
    )
    
    def __post_init__(self):
        pass

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length
        }

@EnvConfig.register_subclass("libero_goal")
@dataclass
class LiberoEnv(EnvConfig):
    task: str = "libero_goal"
    fps: int = 10
    episode_length: int = 520
    obs_type: str = "pixels_agent_pos"
    render_mode: str = "rgb_array"
    visualization_width: int = 224
    visualization_height: int = 224
    max_steps: int = 520
    # BUG here
    features: dict[str, PolicyFeature] = field(
        default_factory=lambda: {
            "actions": PolicyFeature(type=FeatureType.ACTION, shape=(7,)),
            "state": PolicyFeature(type=FeatureType.STATE, shape=(8, )),
            "image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
            "wrist_image": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        }
    )
    features_map: dict[str, str] = field(
        default_factory=lambda: {
            "actions": ACTION,
            "state": OBS_ROBOT,
            "image": OBS_IMAGE,
            # BUG here: may be different from the key in training process
            "wrist_image": "wrist_image",
        }
    )
    
    def __post_init__(self):
        pass

    @property
    def gym_kwargs(self) -> dict:
        return {
            "obs_type": self.obs_type,
            "render_mode": self.render_mode,
            "visualization_width": self.visualization_width,
            "visualization_height": self.visualization_height,
            "max_episode_steps": self.episode_length
        }