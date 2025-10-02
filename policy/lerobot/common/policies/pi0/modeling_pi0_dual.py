#!/usr/bin/env python

# Copyright 2025 Physical Intelligence and The HuggingFace Inc. team. All rights reserved.
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

"""
π0: A Vision-Language-Action Flow Model for General Robot Control

[Paper](https://www.physicalintelligence.company/download/pi0.pdf)
[Jax code](https://github.com/Physical-Intelligence/openpi)

Designed by Physical Intelligence. Ported from Jax by Hugging Face.

Install pi0 extra dependencies:
```bash
pip install -e ".[pi0]"
```

Example of finetuning the pi0 pretrained model (`pi0_base` in `openpi`):
```bash
python lerobot/scripts/train.py \
--policy.path=lerobot/pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of finetuning the pi0 neural network with PaliGemma and expert Gemma
pretrained with VLM default parameters before pi0 finetuning:
```bash
python lerobot/scripts/train.py \
--policy.type=pi0 \
--dataset.repo_id=danaaubakirova/koch_test
```

Example of using the pi0 pretrained model outside LeRobot training framework:
```python
policy = Pi0Policy.from_pretrained("lerobot/pi0")
```

"""

import math
from collections import deque

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from transformers import AutoTokenizer
import numpy as np

from lerobot.common.constants import ACTION, OBS_ROBOT, ACTION_IS_PAD, OBS_TRAJ, TRAJ, TRAJ_IS_PAD
from lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.common.policies.pi0.configuration_pi0 import PI0Config
from lerobot.common.policies.pi0.paligemma_with_dual_expert import (
    PaliGemmaWithExpertConfig,
    PaliGemmaWithDualExpertModel,
)
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.utils import get_safe_dtype

def make_bool_mask(*dims: int) -> tuple[bool, ...]:
    """Make a boolean mask for the given dimensions.

    Example:
        make_bool_mask(2, -2, 2) == (True, True, False, False, True, True)
        make_bool_mask(2, 0, 2) == (True, True, True, True)

    Args:
        dims: The dimensions to make the mask for.

    Returns:
        A tuple of booleans.
    """
    result = []
    for dim in dims:
        if dim > 0:
            result.extend([True] * (dim))
        else:
            result.extend([False] * (-dim))
    return tuple(result)

def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def sample_beta(alpha, beta, bsize, device):
    gamma1 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / alpha)
    gamma2 = torch.empty((bsize,), device=device).uniform_(0, 1).pow(1 / beta)
    return gamma1 / (gamma1 + gamma2)


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by pi0 to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)

class PI0Policy(PreTrainedPolicy):
    """Wrapper class around PI0FlowMatching model to train and run inference within LeRobot."""

    config_class = PI0Config
    name = "pi0"

    def __init__(
        self,
        config: PI0Config,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config
        # import ipdb; ipdb.set_trace()
        self.dataset_stats = dataset_stats
        if self.dataset_stats is not None:
            self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
            self.normalize_targets = Normalize(config.output_features, config.normalization_mapping, dataset_stats)
            self.unnormalize_outputs = Unnormalize(config.output_features, config.normalization_mapping, dataset_stats)

        self.language_tokenizer = AutoTokenizer.from_pretrained("/storage/qiguojunLab/qiguojun/home/Models/google/paligemma-3b-pt-224")

        self.model = PI0FlowMatching(config)
    
        self.IDX = 0

        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        # TODO: whether or not need trajectory reset
        self._action_queue = deque([], maxlen=self.config.n_action_steps)

    def get_optim_params(self) -> dict:
        return self.parameters()

    @torch.no_grad
    def select_action(self, batch: dict[str, Tensor], noise: Tensor | None = None) -> Tensor:
        """Select actions given environment observations.
        Returns all predicted actions at once (batch_size, n_action_steps, action_dim).
        """
        print(f'input features:{self.config.input_features}')
        print(f'output features:{self.config.output_features}')

        self.eval()
        if self.config.adapt_to_pi_aloha:
            assert 0
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
        # BUG: When in inference mode, normalize stats is None! 
        absolute_state = batch[OBS_ROBOT]

        batch = self.normalize_inputs(batch)
        images, img_masks = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        
        state = None
        if not self.config.train_trajectory_expert_only:
            state = self.prepare_state(batch)

        state_traj = self.prepare_state_traj(batch)
        
        trajectory, actions = self.model.sample_actions(
            images, img_masks, lang_tokens, lang_masks, \
            state=state, state_traj=state_traj, noise=noise
        )
        
        original_traj_dim = self.config.traj_feature.shape[0]
        trajectory = trajectory[:, :, :original_traj_dim]
        trajectory = self.unnormalize_outputs({"trajectory": trajectory})["trajectory"]

        if not self.config.train_trajectory_expert_only:
            # BUG: original_action_dim may be different with state
            # Unpad actions
            original_action_dim = self.config.action_feature.shape[0]
            # original_action_dim = batch[f"{ACTION}"].shape[-1]
            actions = actions[:, :, :original_action_dim]
            actions = self.unnormalize_outputs({"actions": actions})["actions"]

            if self.config.use_delta_actions:
                # Recover from delta actions to absolute actions
                # delta_actions shape: [batch_size, seq_len, action_dim]
                # Subsequent items accumulate delta actions
                actions = torch.cumsum(actions, dim=-2) + absolute_state[..., :original_action_dim].unsqueeze(-2)
 
            if self.config.adapt_to_pi_aloha:
                actions = self._pi_aloha_encode_actions(actions)
            return trajectory.squeeze(0), actions.squeeze(0)
        return trajectory.squeeze(0), None

    def forward(self, batch: dict[str, Tensor], noise=None, time=None) -> tuple[Tensor, dict[str, Tensor]]:
        """Do a full training forward pass to compute the loss"""

        if self.config.adapt_to_pi_aloha:
            batch[OBS_ROBOT] = self._pi_aloha_decode_state(batch[OBS_ROBOT])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        # if self.dataset_stats is not None:# data normalization should be done in the dataloader
        #     batch = self.normalize_inputs(batch)
        #     batch = self.normalize_targets(batch)

        images, img_masks, images_keys = self.prepare_images(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        state, actions = None, None
        if not self.config.train_trajectory_expert_only:
            state = self.prepare_state(batch)
            original_action_dim = batch[ACTION].shape[-1]
            actions = self.prepare_action(batch)
            actions_is_pad = batch.get(ACTION_IS_PAD)

        state_traj = self.prepare_state_traj(batch)
        trajectory = self.prepare_trajectory(batch)
        trajectory_is_pad = batch.get(TRAJ_IS_PAD)

        loss_dict = {}
        losses_return_dict = self.model.forward(images, img_masks, lang_tokens, lang_masks, 
                                    state, actions, 
                                    state_traj, trajectory,
                                    noise, time, images_keys=images_keys)

        if not self.config.train_trajectory_expert_only:
            loss_dict["losses_after_forward_action"] = losses_return_dict['action_losses'].clone()
            if actions_is_pad is not None:
                in_episode_bound = ~actions_is_pad
                losses = losses_return_dict["action_losses"] * in_episode_bound.unsqueeze(-1)
                loss_dict["losses_after_in_ep_bound"] = losses.clone()

        loss_dict["losses_after_forward_trajectory"] = losses_return_dict['trajectory_losses'].clone()
        if trajectory_is_pad is not None:
            in_episode_bound = ~trajectory_is_pad
            losses = losses_return_dict['trajectory_losses'] * in_episode_bound.unsqueeze(-1)
            loss_dict["losses_after_in_ep_bound_trajectory"] = losses.clone()

        # For backward pass
        if not self.config.train_trajectory_expert_only:
            # apply masks to the action expert loss 
            # batch['action_expert_loss_mask'] shape: (batch_size) 
            # losses_return_dict['action_losses'] shape: (batch_size, )
            # action_expert_loss_mask is used to select out the batch data needed to be calculated in loss
            if "action_expert_loss_mask" in batch:
                action_expert_loss_mask = batch['action_expert_loss_mask']
                losses = losses_return_dict["action_losses"] * action_expert_loss_mask.view(-1, 1, 1)
                loss_dict["losses_after_action_expert_mask"] = losses.clone()
            loss = losses_return_dict['action_losses'].mean() + losses_return_dict['trajectory_losses'].mean()
            loss_dict["action_losses"] = losses_return_dict['action_losses'].mean()
            loss_dict["trajectory_losses"] = losses_return_dict['trajectory_losses'].mean()
        else:
            loss = losses_return_dict['trajectory_losses'].mean()
            loss_dict["trajectory_losses"] = loss

        # For logging
        loss_dict["l2_loss"] = loss.item()

        return loss, loss_dict

    def prepare_images(self, batch):
        """Apply Pi0 preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key]

            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expacted by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            img = torch.ones_like(img) * -1
            mask = torch.zeros_like(mask)
            images.append(img)
            img_masks.append(mask)
        images_keys = present_img_keys + missing_img_keys
        return images, img_masks, images_keys

    def prepare_language(self, batch) -> tuple[Tensor, Tensor]:
        """Tokenize the text input"""
        device = batch[OBS_ROBOT].device
        tasks = batch["task"]
        # PaliGemma prompt has to end with a new line
        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding="max_length",
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = pad_vector(batch[OBS_ROBOT], self.config.max_state_dim)
        return state
    
    def prepare_state_traj(self, batch):
        """Pad trajectory state"""
        state_traj = pad_vector(batch[OBS_TRAJ], self.config.max_traj_state_dim)
        return state_traj

    def prepare_action(self, batch):
        # NOTE: add delta action in 2025-07-01
        """Pad action, and optionally convert absolute actions to delta actions."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions
    
    def prepare_trajectory(self, batch):
        """Pad trajectory"""
        trajectory = pad_vector(batch[TRAJ], self.config.max_trajectory_dim)
        return trajectory


class PI0FlowMatching(nn.Module):
    """
    π0: A Vision-Language-Action Flow Model for General Robot Control
    [Paper](https://www.physicalintelligence.company/download/pi0.pdf)
    [Jax code](https://github.com/Physical-Intelligence/openpi)
    Designed by Physical Intelligence. Ported from Jax by Hugging Face.
    Modified to include trajectory prediction.
    ┌────────────────────────────────────┐
    │              actions               │
    │              ▲                     │
    │             ┌┴─────┐               │
    │ kv cache    │Gemma │               │
    │ ┌──────────►│Expert│               │
    │ │           │      │               │
    │ │x 10       │x 10  │               │
    │ │           └▲──▲──┘               │
    │ │            │  │                  │
    │ │            │  robot state         │
    │ │            │  │                  │
    │ │            │  ▼                  │
    │ │            │noisy trajectory     │
    │ │            │  ▲                  │
    │ │            │  │                  │
    │ │            │  noise               │
    │ │            │                      │
    │ │PaliGemma   │                      │
    │ │            │                      │
    │ │            │                      │
    │ │            │                      │
    │ │            │                      │
    │ │            │                      │
    │ │            │                      │
    │ │            │                      │
    │ └▲──▲────────┘                      │
    │  │  │                               │
    │  │  image(s)                        │
    │  language tokens                   │
    └────────────────────────────────────┘
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        paligemma_with_export_config = PaliGemmaWithExpertConfig(
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            attention_implementation=self.config.attention_implementation,
        )
        self.paligemma_with_expert = PaliGemmaWithDualExpertModel(paligemma_with_export_config, \
                                                                  shared_expert=self.config.shared_expert)
        # Projections are float32
        self.state_proj = nn.Linear(self.config.max_state_dim, self.config.proj_width)
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.config.proj_width)
        self.action_out_proj = nn.Linear(self.config.proj_width, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.action_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        
        self.trajectory_state_proj = nn.Linear(self.config.max_traj_state_dim, self.config.proj_width)
        self.trajectory_in_proj = nn.Linear(self.config.max_trajectory_dim, self.config.proj_width)
        self.trajectory_out_proj = nn.Linear(self.config.proj_width, self.config.max_trajectory_dim)
        self.trajectory_time_mlp_in = nn.Linear(self.config.proj_width * 2, self.config.proj_width)
        self.trajectory_time_mlp_out = nn.Linear(self.config.proj_width, self.config.proj_width)
        
        self.randmask_traj2action_prob = self.config.randmask_traj2action_prob
        self.randmask_traj2egoimage_prob = self.config.randmask_traj2egoimage_prob
        self.randmask_action2egoimage_prob = self.config.randmask_action2egoimage_prob

        self.set_requires_grad()

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj
        for params in self.trajectory_state_proj.parameters():
            params.requires_grad = self.config.train_trajectory_state_proj
        self.randmask_traj2action_prob = self.config.randmask_traj2action_prob
        # for params in self.trajectory_in_proj.parameters():
        #     params.requires_grad = self.config.train_trajectory_proj
        # for params in self.trajectory_out_proj.parameters():
        #     params.requires_grad = self.config.train_trajectory_proj
        # for params in self.trajectory_time_mlp_in.parameters():
        #     params.requires_grad = self.config.train_trajectory_proj
        # for params in self.trajectory_time_mlp_out.parameters():
        #     params.requires_grad = self.config.train_trajectory_proj
        # -------------------------

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        # Note: Jax version uses (1.0, 3.0) for beta sampling
        time_beta = sample_beta(1.5, 1.0, bsize, device) # Assuming sample_beta is defined elsewhere
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        # TODO: avoid list in python and torch.cat ; prefer pre-allocation with torch.empty
        embs = []
        pad_masks = []
        att_masks = []
        # TODO: remove for loop
        for (
            img,
            img_mask,
        ) in zip(images, img_masks, strict=False):
            img_emb = self.paligemma_with_expert.embed_image(img)
            img_emb = img_emb.to(dtype=torch.bfloat16)
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)
            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)
            embs.append(img_emb)
            pad_masks.append(img_mask)
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs
        lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)
        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def embed_traj(self, state, noisy_trajectories, timestep):
        """Embed state and noisy_trajectories to prepare for Expert Gemma processing.
        Order: [state_emb, trajectory_emb]
        """
        embs = []
        pad_masks = []
        att_masks = []
        bsize = state.shape[0]
        dtype = state.dtype
        device = state.device

        # Embed state
        state_emb = self.trajectory_state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])  # [B, 1, D]
        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        # State can attend to image/language (0) but not to trajectory (1)
        att_masks += [1]

        # Embed timestep for trajectory using sine-cosine positional encoding
        time_emb_traj = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb_traj = time_emb_traj.type(dtype=dtype)

        # Embed noisy_trajectories
        trajectory_emb = self.trajectory_in_proj(noisy_trajectories)  # [B, T_traj, D]
        time_emb_traj_expanded = time_emb_traj[:, None, :].expand_as(trajectory_emb)  # [B, T_traj, D]
        trajectory_time_emb = torch.cat([trajectory_emb, time_emb_traj_expanded], dim=2)  # [B, T_traj, 2*D]
        trajectory_time_emb = self.trajectory_time_mlp_in(trajectory_time_emb)
        trajectory_time_emb = F.silu(trajectory_time_emb)
        trajectory_time_emb = self.trajectory_time_mlp_out(trajectory_time_emb)

        embs.append(trajectory_time_emb)  # [B, T_traj, D]
        trajectory_mask = torch.ones(bsize, self.config.n_trajectory_steps, dtype=torch.bool, device=device)
        pad_masks.append(trajectory_mask)
        # Trajectory can attend to image/language/state (0)
        att_masks += [1] + ([0] * ( self.config.n_trajectory_steps - 1))

        # Final outputs
        embs = torch.cat(embs, dim=1)        # [B, 1 + T_traj, D]
        pad_masks = torch.cat(pad_masks, dim=1)  # [B, 1 + T_traj]
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)  # [1 + T_traj]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))  # [B, 1 + T_traj]

        return embs, pad_masks, att_masks
    
    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Embed state
        state_emb = self.state_proj(state)
        state_emb = state_emb.to(dtype=torch.bfloat16)
        embs.append(state_emb[:, None, :])
        bsize = state_emb.shape[0]
        dtype = state_emb.dtype
        device = state_emb.device

        state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.config.proj_width, min_period=4e-3, max_period=4.0, device=device
        )
        time_emb = time_emb.type(dtype=dtype)

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.n_action_steps - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(
        self, images, img_masks, lang_tokens, lang_masks, state=None, actions=None, 
        state_traj=None, trajectories=None, noise=None, time=None, images_keys=None,
    ) -> Tensor:
        """Do a full training forward pass and compute the combined loss.
           Returns loss tensor of shape (batch_size,)
        """
        ### BUG: assert needed
        # assert (state is not None) and (not self.config.train_trajectory_expert_only)
        # assert actions.shape == trajectories.shape
        if time is None:
            time = self.sample_time(trajectories.shape[0], trajectories.device)
        time_expanded = time[:, None, None]
        
        ##### process actions 
        if not self.config.train_trajectory_expert_only:
            if noise is None:
                noise_actions = self.sample_noise(actions.shape, actions.device)
            else:
                noise_actions = noise


            # Generate noisy samples for actions
            x_t_actions = time_expanded * noise_actions + (1 - time_expanded) * actions
            u_t_actions = noise_actions - actions # Target for actions

            # Pass noisy actions to embed_suffix
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t_actions, time)

        ##### process trajectories
        if noise is None:
            # Sample noise for trajectories
            # Assume trajectories shape is (B, n_trajectory_steps, max_trajectory_dim)
            noise_trajectories = self.sample_noise(trajectories.shape, trajectories.device)
        else:
            # Assume noise is a tuple (noise_actions, noise_trajectories) if provided
            assert 0
            noise_trajectories = noise

        # Generate noisy samples for trajectories
        assert trajectories is not None, "Ground truth trajectories must be provided for trajectory prediction."
        # TODO: whether or not the time of trajectories and actions should be the same?
        x_t_trajectories = \
            time_expanded[:, :self.config.n_trajectory_steps, :self.config.max_trajectory_dim] * noise_trajectories \
            + (1 - time_expanded[:, :self.config.n_trajectory_steps, :self.config.max_trajectory_dim]) * trajectories
        u_t_trajectories = noise_trajectories - trajectories # Target for trajectories

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks)
        
        # Pass noisy trajectories to embed_suffix
        traj_embs, traj_pad_masks, traj_att_masks = self.embed_traj(state_traj, x_t_trajectories, time)

        if not self.config.train_trajectory_expert_only:
            pad_masks = torch.cat([prefix_pad_masks, traj_pad_masks, suffix_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, traj_att_masks, suffix_att_masks], dim=1)
        else:
            # Only use prefix and trajectory embeddings
            pad_masks = torch.cat([prefix_pad_masks, traj_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, traj_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks) # Assuming make_att_2d_masks is defined elsewhere
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # random masking other conditions for actions except for trajectory
        if (not self.config.train_trajectory_expert_only) and (self.randmask_traj2action_prob > 0.0):
            prefix_len = prefix_embs.shape[1] # images+language
            traj_len = traj_embs.shape[1]
            action_len = suffix_embs.shape[1]
            if torch.rand(1).item() < self.randmask_traj2action_prob:
                # mask attention maps from actions to prefix and state_traj
                att_2d_masks[:, prefix_len+traj_len:, :prefix_len+1] = False
        
        # random masking ego_image condition for trajectory prediction
        if self.randmask_traj2egoimage_prob > 0.0:
            if torch.rand(1).item() < self.randmask_traj2egoimage_prob:
                # mask attention maps from trajectory to ego_image
                # 0. determine the ego_image index by images_keys
                assert images_keys is not None
                ego_img_indices = [i for i, key in enumerate(images_keys) if "ego" in key]
                if len(ego_img_indices) > 0:
                    prefix_len = prefix_embs.shape[1] # images+language
                    image_len_total = prefix_len - lang_tokens.shape[1]
                    image_len = image_len_total // len(images_keys)
                    traj_len = traj_embs.shape[1]
                    # 1. determine the token index range of ego_image
                    ego_img_token_start = ego_img_indices[0] * image_len
                    ego_img_token_end = (ego_img_indices[0] + 1) * image_len
                    att_2d_masks[:, prefix_len:prefix_len+traj_len, ego_img_token_start:ego_img_token_end] = False

        # random masking ego_image condition for action prediction
        if (not self.config.train_trajectory_expert_only) and (self.randmask_action2egoimage_prob > 0.0):
            if torch.rand(1).item() < self.randmask_action2egoimage_prob:
                # mask attention maps from actions to ego_image
                # 0. determine the ego_image index by images_keys
                assert images_keys is not None
                ego_img_indices = [i for i, key in enumerate(images_keys) if "ego" in key]
                if len(ego_img_indices) > 0:
                    prefix_len = prefix_embs.shape[1]
                    image_len_total = prefix_len - lang_tokens.shape[1]
                    image_len = image_len_total // len(images_keys)
                    traj_len = traj_embs.shape[1]
                    action_len = suffix_embs.shape[1]
                    # 1. determine the token index range of ego_image
                    ego_img_token_start = ego_img_indices[0] * image_len
                    ego_img_token_end = (ego_img_indices[0] + 1) * image_len
                    att_2d_masks[:, prefix_len+traj_len:prefix_len+traj_len+action_len, \
                                  ego_img_token_start:ego_img_token_end] = False


        # Forward pass through the model
        # The model now receives suffix_embs which includes trajectory embeddings
        expert_out, _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, traj_embs, suffix_embs] if not self.config.train_trajectory_expert_only else [prefix_embs, traj_embs],
            use_cache=False,
            fill_kv_cache=False,
            train_trajectory_expert_only=self.config.train_trajectory_expert_only,
        )

        suffix_out = expert_out[-1] if not self.config.train_trajectory_expert_only else None
        traj_out = expert_out[1]

        # suffix_out shape: [B, Prefix_Len + 1 + T_traj + 1 + T_act, D]
        # We are interested in the last 1 + T_traj + T_act elements
        # state_out = suffix_out[:, -1-self.config.n_trajectory_steps-self.config.n_action_steps:-self.config.n_trajectory_steps-self.config.n_action_steps+1, :]
        loss_dict = {}
        trajectory_out = traj_out[:, -self.config.n_trajectory_steps:, :] # [B, T_traj, D]
        trajectory_out = trajectory_out.to(dtype=torch.float32)
        v_t_trajectories = self.trajectory_out_proj(trajectory_out) # [B, T_traj, max_trajectory_dim]
        trajectory_losses = F.mse_loss(u_t_trajectories, v_t_trajectories, reduction="none") # [B, T_traj, max_trajectory_dim]
        loss_dict["trajectory_losses"] = trajectory_losses

        if not self.config.train_trajectory_expert_only:
            action_out = suffix_out[:, -self.config.n_action_steps:, :]           # [B, T_act, D]
            action_out = action_out.to(dtype=torch.float32)
            v_t_actions = self.action_out_proj(action_out)             # [B, T_act, max_action_dim]
            # Compute losses
            action_losses = F.mse_loss(u_t_actions, v_t_actions, reduction="none") # [B, T_act, max_action_dim]
            loss_dict["action_losses"] = action_losses
        return loss_dict


    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state=None, state_traj=None, noise=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state_traj.shape[0]
        device = state_traj.device

        if noise is None:
            trajectories_shape = (bsize, self.config.n_trajectory_steps, self.config.max_trajectory_dim)
            noise_trajectories = self.sample_noise(trajectories_shape, device)
        else:
            noise_trajectories = noise
        if not self.config.train_trajectory_expert_only:
            if noise is None:
                actions_shape = (bsize, self.config.n_action_steps, self.config.max_action_dim)
                noise_actions = self.sample_noise(actions_shape, device)
            else:
                # Assume noise is a tuple (noise_actions, noise_trajectories) if provided for inference
                noise_actions = noise

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        # Compute image and language key value cache
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None] if self.config.train_trajectory_expert_only else [prefix_embs, None, None],
            use_cache=self.config.use_cache,
            fill_kv_cache=True,
            train_trajectory_expert_only=self.config.train_trajectory_expert_only,
        )

        dt = -1.0 / self.config.num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)
        time = torch.tensor(1.0, dtype=torch.float32, device=device)

        x_t_actions = None
        if not self.config.train_trajectory_expert_only:
            x_t_actions = noise_actions
        else:
            x_t_actions = None
        x_t_trajectories = noise_trajectories
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            # Denoise both trajectory and action
            v_t_trajectories, v_t_actions = self.denoise_step(
                state,
                state_traj,
                prefix_pad_masks,
                past_key_values,
                x_t_trajectories,
                x_t_actions,
                expanded_time,
            )
            # Euler step for both
            x_t_trajectories += dt * v_t_trajectories
            if not self.config.train_trajectory_expert_only:
                x_t_actions += dt * v_t_actions 
            time += dt
        # Return final denoised actions
        return x_t_trajectories, x_t_actions

    def denoise_step(
        self,
        state,
        state_traj,
        prefix_pad_masks,
        past_key_values,
        x_t_trajectories,
        x_t_actions,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        traj_embs, traj_pad_masks, traj_att_masks = self.embed_traj(state_traj, x_t_trajectories, timestep)
        traj_len = traj_pad_masks.shape[1]
        traj_att_2d_masks = make_att_2d_masks(traj_pad_masks, traj_att_masks)
        if not self.config.train_trajectory_expert_only:
            # After traj_att_2d_masks, pad a (batch_size, traj_len, traj_len) tensor of all True
            pad_true = torch.ones((traj_att_2d_masks.shape[0], traj_len, traj_len), dtype=torch.bool, device=traj_att_2d_masks.device)
            traj_att_2d_masks = torch.cat([traj_att_2d_masks, pad_true], dim=1)

        if not self.config.train_trajectory_expert_only:
            suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(state, x_t_actions, timestep)
            suffix_len = suffix_pad_masks.shape[1]
            suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
            pad_false = torch.zeros((suffix_att_2d_masks.shape[0], suffix_len, suffix_len), dtype=torch.bool, device=suffix_att_2d_masks.device)
            suffix_att_2d_masks = torch.cat([pad_false, suffix_att_2d_masks], dim=1)

        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        if not self.config.train_trajectory_expert_only:
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, traj_len+suffix_len, prefix_len)
        else:
            prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, traj_len, prefix_len)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]

        if not self.config.train_trajectory_expert_only:
            # import ipdb; ipdb.set_trace()
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, traj_att_2d_masks, suffix_att_2d_masks], dim=2)
            position_ids = prefix_offsets + torch.cumsum(torch.cat([traj_pad_masks, suffix_pad_masks], dim=1), dim=1) - 1
        else:
            full_att_2d_masks = torch.cat([prefix_pad_2d_masks, traj_att_2d_masks], dim=2)
            position_ids = prefix_offsets + torch.cumsum(traj_pad_masks, dim=1) - 1
        # check 2d mask, position_ids

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, traj_embs, suffix_embs] if not self.config.train_trajectory_expert_only else [None, traj_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
            train_trajectory_expert_only=self.config.train_trajectory_expert_only,
        )

        v_t = None
        if not self.config.train_trajectory_expert_only:
            suffix_out = outputs_embeds[-1]
            suffix_out = suffix_out[:, -self.config.n_action_steps :]
            suffix_out = suffix_out.to(dtype=torch.float32)
            v_t = self.action_out_proj(suffix_out)
        traj_out = outputs_embeds[1]
        traj_out = traj_out[:, -self.config.n_trajectory_steps :]
        traj_out = traj_out.to(dtype=torch.float32)
        v_t_traj = self.trajectory_out_proj(traj_out)

        return v_t_traj, v_t
