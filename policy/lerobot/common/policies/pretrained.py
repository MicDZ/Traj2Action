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
import logging
import os
from pathlib import Path
from typing import Type, TypeVar

import packaging
import safetensors
from huggingface_hub import hf_hub_download
from huggingface_hub.constants import SAFETENSORS_SINGLE_FILE
from huggingface_hub.errors import HfHubHTTPError
from safetensors.torch import load_model as load_model_as_safetensor
from safetensors.torch import save_model as save_model_as_safetensor
from torch import Tensor, nn

from lerobot.common.utils.hub import HubMixin
from lerobot.configs.policies import PreTrainedConfig
SAFETENSORS_MULTI_FILE = [
    "model-00001-of-00003.safetensors", 
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors",
]

T = TypeVar("T", bound="PreTrainedPolicy")

DEFAULT_POLICY_CARD = """
---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

This policy has been pushed to the Hub using [LeRobot](https://github.com/huggingface/lerobot):
- Docs: {{ docs_url | default("[More Information Needed]", true) }}
"""

def filter_norm_keys(state_dict):
    # 只保留非normalize/unnormalize相关的参数
    norm_keywords = [
        "normalize_inputs", "normalize_targets", "unnormalize_outputs"
    ]
    filtered = {k: v for k, v in state_dict.items() if not any(nk in k for nk in norm_keywords)}
    return filtered


class PreTrainedPolicy(nn.Module, HubMixin, abc.ABC):
    """
    Base class for policy models.
    """

    config_class: None
    name: None

    def __init__(self, config: PreTrainedConfig, *inputs, **kwargs):
        super().__init__()
        if not isinstance(config, PreTrainedConfig):
            raise ValueError(
                f"Parameter config in `{self.__class__.__name__}(config)` should be an instance of class "
                "`PreTrainedConfig`. To create a model from a pretrained model use "
                f"`model = {self.__class__.__name__}.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.config = config

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if not getattr(cls, "config_class", None):
            raise TypeError(f"Class {cls.__name__} must define 'config_class'")
        if not getattr(cls, "name", None):
            raise TypeError(f"Class {cls.__name__} must define 'name'")

    def _save_pretrained(self, save_directory: Path) -> None:
        self.config._save_pretrained(save_directory)
        model_to_save = self.module if hasattr(self, "module") else self
        save_model_as_safetensor(model_to_save, str(save_directory / SAFETENSORS_SINGLE_FILE))

    @classmethod
    def from_pretrained(
        cls: Type[T],
        pretrained_name_or_path: str | Path,
        *,
        config: PreTrainedConfig | None = None,
        force_download: bool = False,
        resume_download: bool | None = None,
        proxies: dict | None = None,
        token: str | bool | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        strict: bool = False,
        **kwargs,
    ) -> T:
        """
        The policy is set in evaluation mode by default using `policy.eval()` (dropout modules are
        deactivated). To train it, you should first set it back in training mode with `policy.train()`.
        """
        if config is None:
            config = PreTrainedConfig.from_pretrained(
                pretrained_name_or_path=pretrained_name_or_path,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                token=token,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                revision=revision,
                **kwargs,
            )
        model_id = str(pretrained_name_or_path)
        instance = cls(config, **kwargs)
        if os.path.isdir(model_id):
            print("Loading weights from local directory")
            model_file = os.path.join(model_id, SAFETENSORS_SINGLE_FILE)
            policy = cls._load_as_safetensor(instance, model_file, "cpu", strict)
            if config and hasattr(config, "device") and config.device != "cpu":
                policy = policy.to(config.device)
        else:
            try:
                model_file = hf_hub_download(
                    repo_id=model_id,
                    filename=SAFETENSORS_SINGLE_FILE,
                    revision=revision,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    resume_download=resume_download,
                    token=token,
                    local_files_only=local_files_only,
                )
                policy = cls._load_as_safetensor(instance, model_file, config.device, strict)
            except HfHubHTTPError as e:
                raise FileNotFoundError(
                    f"{SAFETENSORS_SINGLE_FILE} not found on the HuggingFace Hub in {model_id}"
                ) from e

        policy.to(config.device)
        policy.eval()
        return policy

    @classmethod
    def _load_as_safetensor(cls, model: T, model_file: list[str], map_location: str, strict: bool) -> T:
        if len(model_file) == 1:
            model_file = model_file[0]
            if packaging.version.parse(safetensors.__version__) < packaging.version.parse("0.4.3"):
                load_model_as_safetensor(model, model_file, strict=strict)
                if map_location != "cpu":
                    logging.warning(
                        "Loading model weights on other devices than 'cpu' is not supported natively in your version of safetensors."
                        " This means that the model is loaded on 'cpu' first and then copied to the device."
                        " This leads to a slower loading time."
                        " Please update safetensors to version 0.4.3 or above for improved performance."
                    )
                    model.to(map_location)
            else:
                # NOTE: change load_file code, original code is:
                # safetensors.torch.load_model(model, model_file, strict=strict, device=map_location)
                state_dict = safetensors.torch.load_file(model_file, device=map_location)
                state_dict = filter_norm_keys(state_dict)
                model.load_state_dict(state_dict, strict=strict)
        else: 
            all_state_dict = {}
            for file in model_file:
                state_dict = safetensors.torch.load_file(file, device=map_location)
                # state_dict = filter_norm_keys(state_dict)
                all_state_dict.update(state_dict)
            model_parameter_names = list(model.state_dict().keys())
            model_param_index = {name: i for i, name in enumerate(model_parameter_names)}
            
            for i, element in enumerate(list(all_state_dict.keys())):
                if element == "paligemma.language_model.model.embed_tokens.weight": 
                    all_state_dict["model.paligemma_with_expert.paligemma.language_model.lm_head.weight"] = all_state_dict[element]
                matching_params = [param for param in model_parameter_names if param.endswith(element)]
                if matching_params: 
                    corresponding_param = max(matching_params, key=len)
                    all_state_dict[corresponding_param] = all_state_dict[element]
                    all_state_dict.pop(element)
                else: 
                    print(f"No matching parameter found for {element}")
            all_state_dict = filter_norm_keys(all_state_dict) 
            model.load_state_dict(all_state_dict, strict=strict)
        return model

    # def generate_model_card(self, *args, **kwargs) -> ModelCard:
    #     card = ModelCard.from_template(
    #         card_data=self._hub_mixin_info.model_card_data,
    #         template_str=self._hub_mixin_info.model_card_template,
    #         repo_url=self._hub_mixin_info.repo_url,
    #         docs_url=self._hub_mixin_info.docs_url,
    #         **kwargs,
    #     )
    #     return card

    @abc.abstractmethod
    def get_optim_params(self) -> dict:
        """
        Returns the policy-specific parameters dict to be passed on to the optimizer.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self):
        """To be called whenever the environment is reset.

        Does things like clearing caches.
        """
        raise NotImplementedError

    # TODO(aliberts, rcadene): split into 'forward' and 'compute_loss'?
    @abc.abstractmethod
    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        """_summary_

        Args:
            batch (dict[str, Tensor]): _description_

        Returns:
            tuple[Tensor, dict | None]: The loss and potentially other information. Apart from the loss which
                is a Tensor, all other items should be logging-friendly, native Python types.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return one action to run in the environment (potentially in batch mode).

        When the model uses a history of observations, or outputs a sequence of actions, this method deals
        with caching.
        """
        raise NotImplementedError
