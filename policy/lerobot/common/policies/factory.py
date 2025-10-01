#!/usr/bin/env python

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

import logging
from torch import Tensor
from torch import nn

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.envs.configs import EnvConfig
from lerobot.common.envs.utils import env_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0fast.configuration_pi0fast import PI0FASTConfig
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.tdmpc.configuration_tdmpc import TDMPCConfig
from lerobot.common.policies.vqbet.configuration_vqbet import VQBeTConfig
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType


def get_policy_class(name: str) -> PreTrainedPolicy:
    """Get the policy's class and config class given a name (matching the policy class' `name` attribute)."""
    if name == "tdmpc":
        from lerobot.common.policies.tdmpc.modeling_tdmpc import TDMPCPolicy

        return TDMPCPolicy
    elif name == "diffusion":
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

        return DiffusionPolicy
    elif name == "act":
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        return ACTPolicy
    elif name == "vqbet":
        from lerobot.common.policies.vqbet.modeling_vqbet import VQBeTPolicy

        return VQBeTPolicy
    # elif name == "pi0":
    #     from lerobot.common.policies.pi0.modeling_pi0 import PI0Policy

    #     return PI0Policy
    elif name == "pi0":
        from lerobot.common.policies.pi0.modeling_pi0_dual import PI0Policy

        return PI0Policy
    elif name == "pi0fast":
        from lerobot.common.policies.pi0fast.modeling_pi0fast import PI0FASTPolicy

        return PI0FASTPolicy
    else:
        raise NotImplementedError(f"Policy with name {name} is not implemented.")


def make_policy_config(policy_type: str, **kwargs) -> PreTrainedConfig:
    if policy_type == "tdmpc":
        return TDMPCConfig(**kwargs)
    elif policy_type == "diffusion":
        return DiffusionConfig(**kwargs)
    elif policy_type == "act":
        return ACTConfig(**kwargs)
    elif policy_type == "vqbet":
        return VQBeTConfig(**kwargs)
    elif policy_type == "pi0":
        return PI0Config(**kwargs)
    elif policy_type == "pi0fast":
        return PI0FASTConfig(**kwargs)
    else:
        raise ValueError(f"Policy type '{policy_type}' is not available.")


def make_policy(
    cfg: PreTrainedConfig,
    ds_meta: LeRobotDatasetMetadata | None = None,
    env_cfg: EnvConfig | None = None,
    stats: dict[str, dict[str, Tensor]] | None = None,
    ds_meta_offline: dict | None = None,
) -> PreTrainedPolicy:
    """Make an instance of a policy class.

    This function exists because (for now) we need to parse features from either a dataset or an environment
    in order to properly dimension and instantiate a policy for that dataset or environment.

    Args:
        cfg (PreTrainedConfig): The config of the policy to make. If `pretrained_path` is set, the policy will
            be loaded with the weights from that path.
        ds_meta (LeRobotDatasetMetadata | None, optional): Dataset metadata to take input/output shapes and
            statistics to use for (un)normalization of inputs/outputs in the policy. Defaults to None.
        env_cfg (EnvConfig | None, optional): The config of a gym environment to parse features from. Must be
            provided if ds_meta is not. Defaults to None.

    Raises:
        ValueError: Either ds_meta or env and env_cfg must be provided.
        NotImplementedError: if the policy.type is 'vqbet' and the policy device 'mps' (due to an incompatibility)

    Returns:
        PreTrainedPolicy: _description_
    """
    if bool(ds_meta) == bool(env_cfg):
        raise ValueError("Either one of a dataset metadata or a sim env must be provided.")

    # NOTE: Currently, if you try to run vqbet with mps backend, you'll get this error.
    # TODO(aliberts, rcadene): Implement a check_backend_compatibility in policies?
    # NotImplementedError: The operator 'aten::unique_dim' is not currently implemented for the MPS device. If
    # you want this op to be added in priority during the prototype phase of this feature, please comment on
    # https://github.com/pytorch/pytorch/issues/77764. As a temporary fix, you can set the environment
    # variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the CPU as a fallback for this op. WARNING: this will be
    # slower than running natively on MPS.
    if cfg.type == "vqbet" and cfg.device == "mps":
        raise NotImplementedError(
            "Current implementation of VQBeT does not support `mps` backend. "
            "Please use `cpu` or `cuda` backend."
        )

    policy_cls = get_policy_class(cfg.type)

    kwargs = {}
    if ds_meta_offline is not None:
        features = ds_meta_offline["features"]
        kwargs["dataset_stats"] = ds_meta_offline["stats"]
    elif ds_meta is not None:
        # import ipdb; ipdb.set_trace()
        features = dataset_to_policy_features(ds_meta.features)
        kwargs["dataset_stats"] = ds_meta.stats
    else:
        if stats is not None:
            kwargs["dataset_stats"] = stats
        if not cfg.pretrained_path:
            logging.warning(
                "You are instantiating a policy from scratch and its features are parsed from an environment "
                "rather than a dataset. Normalization modules inside the policy will have infinite values "
                "by default without stats from a dataset."
            )
        features = env_to_policy_features(env_cfg)
    # import ipdb; ipdb.set_trace()
    cfg.output_features = {key: ft for key, ft in features.items() if (ft.type is FeatureType.ACTION or ft.type is FeatureType.TRAJ)}
    cfg.input_features  = {key: ft for key, ft in features.items() if key not in cfg.output_features}
    print(f'features: {cfg.input_features}, {cfg.output_features}')
    kwargs["config"] = cfg
    
    
    if cfg.load_pretrained_from_pi0:
        import os
        from safetensors.torch import load_file
        # 1. 先初始化模型
        policy = policy_cls(**kwargs)
        # 2. 加载原始权重
        # file_paths = [
        #     os.path.join(cfg.pretrained_path, "model-00001-of-00003.safetensors"),
        #     os.path.join(cfg.pretrained_path, "model-00002-of-00003.safetensors"),
        #     os.path.join(cfg.pretrained_path, "model-00003-of-00003.safetensors"),
        # ]
        # all_weights = {}
        # for path in file_paths:
        #     weights = load_file(path)
        #     all_weights.update(weights)
        state_dict = load_file(
            os.path.join(cfg.pretrained_path, "model.safetensors"),
            device="cpu"
        )     
        # 只保留非normalize/unnormalize相关的参数
        norm_keywords = [
            "normalize_inputs", "normalize_targets", "unnormalize_outputs"
        ]
        filtered = {k: v for k, v in state_dict.items() if not any(nk in k for nk in norm_keywords)}
        state_dict = filtered   
        incompatiblekeys = policy.load_state_dict(state_dict, strict=False)
        missing_keys = incompatiblekeys.missing_keys
        # import ipdb;ipdb.set_trace()
        print([key for key in missing_keys if 'traj' not in key])
        # 只加载 gemma_expert 相关参数
        gemma_traj_expert_state_dict = {k.replace("model.paligemma_with_expert.gemma_expert.", ""): v for k, v in state_dict.items() if k.startswith("model.paligemma_with_expert.gemma_expert")}
        policy.model.paligemma_with_expert.gemma_traj_expert.load_state_dict(gemma_traj_expert_state_dict, strict=True)
        # # 3. 将 gemma_expert 权重拷贝到 gemma_traj_expert
        # policy.gemma_traj_expert.load_state_dict(policy.gemma_traj_expert_state_dict.state_dict(), strict=True)
        policy = policy.to(cfg.device)
    else:
        print(f'loading checkpoint from {cfg.pretrained_path}')
        if cfg.pretrained_path:
            # Load a pretrained policy and override the config if needed (for example, if there are inference-time
            # hyperparameters that we want to vary).
            kwargs["pretrained_name_or_path"] = cfg.pretrained_path
            policy = policy_cls.from_pretrained(**kwargs)
        else:
            # Make a fresh policy.
            policy = policy_cls(**kwargs)

    policy.to(cfg.device)
    assert isinstance(policy, nn.Module)

    # policy = torch.compile(policy, mode="reduce-overhead")

    return policy
