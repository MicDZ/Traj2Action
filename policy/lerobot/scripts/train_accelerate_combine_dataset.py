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
import json
import time
from pprint import pformat
from typing import Any
import os
import sys
import torch
from copy import deepcopy
from collections import defaultdict
from accelerate import Accelerator
from accelerate.utils import set_seed as accelerate_set_seed
from termcolor import colored
from torch.optim import Optimizer
import numpy as np
sys.path.insert(0, os.getcwd())
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.combined_dataset import LerobotCombinedDataset, CollectiveDataloader
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.policies.pi0.modeling_pi0_dual import resize_with_pad
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    has_method,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


from datetime import datetime
from pathlib import Path

from lerobot.common.constants import DATA_KEYS_MAPPING_HAND, DATA_KEYS_MAPPING_ROBOT
from lerobot.common.constants import OBS_ROBOT, ACTION, TRAJ, OBS_TRAJ
def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    batch: Any,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    lr_scheduler=None,
) -> tuple[MetricsTracker, dict]:
    start_time = time.perf_counter()
    policy.train()

    # Use accelerator's autocast context if mixed precision is enabled
    with accelerator.autocast():
        loss, output_dict = policy.forward(batch)
        # TODO(rcadene): policy.unnormalize_outputs(out_dict)

    # Use accelerator for backward pass
    accelerator.backward(loss)

    # Gradient clipping - accelerator handles unscaling automatically
    if accelerator.sync_gradients and grad_clip_norm > 0:
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.tensor(0.0)

    optimizer.step()
    lr_scheduler.step() if lr_scheduler is not None else None
    optimizer.zero_grad()

    # Update policy-specific buffers if needed
    if has_method(policy, "update"):
        policy.update()

    # Gather metrics across all processes
    loss_value = accelerator.gather(loss.detach()).mean().item()
    grad_norm_value = accelerator.gather(grad_norm).mean().item()

    train_metrics.loss = loss_value
    train_metrics.grad_norm = grad_norm_value
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics, output_dict

def print_batch_structure(obj, prefix="batch"):
    """
    递归地打印出 batch 对象中每个元素的类型、形状、设备等信息。
    """
    if isinstance(obj, torch.Tensor):
        print(f"{prefix:<50} | {'Type:':<10} {type(obj).__name__:<20} | {'Shape:':<10} {str(obj.shape):<25} | {'Device:':<10} {obj.device}")
    elif isinstance(obj, dict):
        # 打印字典本身的信息（通常是它的键的数量）
        print(f"{prefix:<50} | {'Type:':<10} {type(obj).__name__:<20} | {'Info:':<10} {f'Keys: {len(obj)}'}")
        for k, v in obj.items():
            # 检查 key 是否为字符串，如果不是，则特别标记
            if not isinstance(k, str):
                print(f"!!! WARNING: Non-string key found: {k} (Type: {type(k).__name__})")
            
            new_prefix = f"{prefix}['{k}']"
            print_batch_structure(v, new_prefix)
    elif isinstance(obj, (list, tuple)):
        # 打印列表/元组本身的信息（长度）
        print(f"{prefix:<50} | {'Type:':<10} {type(obj).__name__:<20} | {'Info:':<10} {f'Len: {len(obj)}'}")
        for i, v in enumerate(obj):
            new_prefix = f"{prefix}[{i}]"
            print_batch_structure(v, new_prefix)
    else:
        # 打印其他所有类型的数据
        print(f"{prefix:<50} | {'Type:':<10} {type(obj).__name__:<20} | {'Info:':<10} Value: {str(obj)[:50]}") # 打印部分值以供参考

def normalize(feature, stat, accelerator):
    mean = torch.tensor(stat['mean'], device=accelerator.device)
    std  = torch.tensor(stat['std'], device=accelerator.device)
    return (feature - mean ) / (std + 1e-8)

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    base_output_dir = Path(cfg.base_output_dir)
    current_time = datetime.now()
    project_name = cfg.wandb.project if cfg.wandb.project else "humanpi"
    job_name = cfg.job_name
    cfg.output_dir = base_output_dir / f"{project_name}" / f"{job_name}" / f"{current_time.strftime('%Y-%m-%d')}" / f"{cfg.policy.type}_{current_time.strftime('%H-%M-%S')}"

    # cfg.checkpoint_path = cfg.output_dir / "checkpoints"
    # cfg.output_dir.mkdir(parents=True, exist_ok=True)
    # cfg.checkpoint_path.mkdir(parents=True, exist_ok=True)
    
    log_file = base_output_dir / "training.log"
    os.makedirs(base_output_dir, exist_ok=True)

    init_logging(log_file=str(log_file))


    logging.info(pformat(cfg.to_dict()))



    # Initialize accelerator
    from accelerate.utils import DistributedDataParallelKwargs

    from lerobot.common.utils.wandb_utils import cfg_to_group, get_wandb_run_id_from_filesystem

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16" if cfg.policy.use_amp else "no",
        gradient_accumulation_steps=cfg.policy.gradient_accumulation_steps,
        log_with="wandb" if cfg.wandb.enable else None,
        kwargs_handlers=[ddp_kwargs],
        project_dir=cfg.output_dir,
    )

    accelerator.init_trackers(
        project_name=cfg.wandb.project,
        init_kwargs={
            "wandb": {
                "entity": cfg.wandb.entity,
                "name": cfg.job_name,
                "notes": cfg.wandb.notes,
                "tags": cfg_to_group(cfg, return_list=True),
                "dir": cfg.output_dir,
                "config": cfg.to_dict(),
                "save_code": False,
                "job_type": "train_eval",
                "mode": cfg.wandb.mode if cfg.wandb.mode in ["online", "offline", "disabled"] else "online",
                "resume": "must" if cfg.resume else None,
                "id": cfg.wandb.run_id
                if cfg.wandb.run_id
                else (get_wandb_run_id_from_filesystem(cfg.output_dir) if cfg.resume else None),
            }
        },
    )

    # Set seed for reproducibility
    if cfg.seed is not None:
        accelerate_set_seed(cfg.seed)

    # Setup device - accelerator handles device placement
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # Create dataset
    if accelerator.is_main_process:
        logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    is_multiple_datasets = False
    if not isinstance(cfg.dataset.repo_id, str):
        is_multiple_datasets = True
    #     visual_keys_used = ["left_image", "ego_image", "top_image"]
    #     new_features = {}
    #     for key, ft in dataset.meta.features.items():
    #         if ft["dtype"] not in ["image", "video"]:
    #             new_features[key] = ft
    #         else:

    # Create evaluation environment (only on main process)
    eval_env = None
    # if cfg.eval_freq > 0 and cfg.env is not None and accelerator.is_main_process:
    #     logging.info("Creating env")
    #     eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    # Create policy
    if accelerator.is_main_process:
        logging.info("Creating policy")

    # Use accelerator's device instead of cfg.policy.device
    with accelerator.main_process_first():
        policy = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta, # include dataset stat 
        )

    # Create optimizer and scheduler
    if accelerator.is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, policy)

    step = 0  # number of policy updates

    if cfg.resume:
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

    # Prepare dataloader
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None
    if isinstance(dataset, LerobotCombinedDataset): 
        dataloader_init = CollectiveDataloader(
            accelerator,
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            device=accelerator.device,
            drop_last=True, #TODO: check if need to be True
            mixing_mode=cfg.dataset.mixing_mode,
            dataset_weights=cfg.dataset.dataset_weights,
        )
        dataloader = dataloader_init.train_dataloader()
    else:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,  # Important for distributed training
        )

    # Prepare for distributed training
    if is_multiple_datasets:
        policy, optimizer, lr_scheduler = accelerator.prepare(
            policy, optimizer, lr_scheduler
        )
    else:
        policy, optimizer, dataloader, lr_scheduler = accelerator.prepare(
            policy, optimizer, dataloader, lr_scheduler
        )

    # Log training info (only on main process)
    if accelerator.is_main_process:
        num_learnable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        num_total_params = sum(p.numel() for p in policy.parameters())

        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        if cfg.env is not None:
            logging.info(f"{cfg.env.task=}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")
        logging.info(f"Number of processes: {accelerator.num_processes}")
        logging.info(f"Device: {accelerator.device}")
        logging.info(f"Mixed precision: {accelerator.mixed_precision}")

    # Create metrics trackers
    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size * accelerator.num_processes,  # Account for all processes
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
    )

    # Training loop
    policy.train()
    if accelerator.is_main_process:
        logging.info("Start offline training on a fixed dataset")

    # Create iterator from dataloader
    dl_iter = iter(dataloader)

    DEBUG_DATALOADER = True
    for current_step in range(step, cfg.steps):
        start_time = time.perf_counter()

        # Get next batch, cycling through dataloader if needed
        try:
            batch = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dataloader)
            batch = next(dl_iter)
        # for multiple datasets, concat each element in the batch
        if is_multiple_datasets and isinstance(dataloader_init, CollectiveDataloader):
            visual_keys_used = cfg.dataset.visual_keys_used
            # visual_keys_used = ["left_image", "ego_image", "top_image"]
            batch_data_repo = deepcopy(batch[0])
            dtype = torch.float32
            action_expert_loss_mask = torch.tensor(data=[],dtype=torch.bool)
            needed_keys = [ACTION, OBS_ROBOT, TRAJ, OBS_TRAJ]

            # mapping all visual keys to the same size
            batch_size = batch_data_repo[list(batch_data_repo.keys())[0]]["timestamp"].shape[0]
            for repo_name, batch_dataset in batch_data_repo.items():
                batch_dataset_tmp = {}
                key_mapping = DATA_KEYS_MAPPING_HAND if "human_image" in batch_dataset.keys() else DATA_KEYS_MAPPING_ROBOT
                # mapping keys in batch_dataset to new keys according to key_mapping
                for new_key in visual_keys_used:
                    old_key = key_mapping[new_key] if new_key in key_mapping else new_key
                    if old_key in batch_dataset.keys():
                        batch_dataset_tmp[new_key] = resize_with_pad(batch_dataset[old_key], *cfg.policy.resize_imgs_with_padding, pad_value=0)
                    else:
                        # feature_shape = dataset.meta.features[new_key]['shape']
                        # new_shape = (batch_size, feature_shape[-1], *feature_shape[:-1])
                        new_shape = (batch_size, 3, *cfg.policy.resize_imgs_with_padding)
                        batch_dataset_tmp[new_key] = torch.zeros(new_shape, dtype=dtype)
                
                
                # pad action/obs_robot to same size and construct action_expert_loss_mask
                # action_expert_loss_mask: 1 for robot action, 0 for human action， used to mask the human action loss
                action_expert_key_used = [OBS_ROBOT, ACTION]
                for action_expert_key in action_expert_key_used:
                    if action_expert_key in batch_dataset.keys():
                        # pad all the action/state to the target size
                        target_size = cfg.policy.action_dim if action_expert_key == ACTION else cfg.policy.state_dim
                        current_size = batch_dataset[action_expert_key].shape[-1]
                        # HACK: use the dim to identify whether this action/state is from human or robot
                        if current_size < target_size: # human dataset
                            pad_tensor = torch.zeros((*batch_dataset[action_expert_key].shape[:-1], target_size - current_size), dtype=dtype)
                            batch_dataset_tmp[action_expert_key] = torch.cat([batch_dataset[action_expert_key], pad_tensor], dim=-1)
                            if action_expert_key == ACTION:
                                # print('constructing loss mask with batchsize:',repo_name, batch_dataset[action_expert_key].shape[0])
                                action_expert_loss_mask = torch.cat([action_expert_loss_mask, torch.zeros((batch_dataset[action_expert_key].shape[0]), dtype=torch.bool)], dim=-1)
                        elif current_size == target_size:# robot dataset
                            batch_dataset_tmp[action_expert_key] = batch_dataset[action_expert_key]
                            if action_expert_key == ACTION:
                                # print('constructing loss mask with batchsize:',repo_name, batch_dataset[action_expert_key].shape[0])
                                action_expert_loss_mask = torch.cat([action_expert_loss_mask, torch.ones((batch_dataset[action_expert_key].shape[0]), dtype=torch.bool)], dim=-1)
                        elif current_size > target_size:
                            raise ValueError(f"current_size {current_size} > target_size {target_size} at key {action_expert_key}")
                # batch_data_repo[repo_name] = {**batch_dataset_tmp, **{k:v for k, v in batch_dataset.items() if (k in dataset.meta.features and k not in visual_keys_used + action_expert_key_used)}}
                for k, v in batch_dataset.items():
                    if k not in (visual_keys_used + action_expert_key_used) and 'image' not in k:
                        batch_dataset_tmp[k] = v
                batch_data_repo[repo_name] = batch_dataset_tmp
                
            assert action_expert_loss_mask.shape[0] == cfg.batch_size, f'action_expert_loss_mask.shape[0] {action_expert_loss_mask.shape[0]} != batch_size {cfg.batch_size}'


            # concat multiple datasets in the batch
            batch = defaultdict(list)
            batch['action_expert_loss_mask'] = action_expert_loss_mask.to(accelerator.device)   
            for repo_name, batch_dataset in batch_data_repo.items():
                for k, v in batch_dataset.items():
                    batch[k].append(v)
            # print all shape of the batch
            # batch = {k: torch.cat(v, dim=0) for k, v in batch.items()}
            
            for key, value in batch.items():
                if isinstance(value[0], torch.Tensor):
                    if key == 'action_expert_loss_mask':
                        continue
                    batch[key] = torch.cat(value, dim=0).to(accelerator.device)
                elif key == 'task':
                    batch[key] = sum(value, [])
                else:
                    raise ValueError(f"Unsupported type: {type(value[0])}")
            

            # if accelerator.is_main_process and current_step == 0: # 只在第一个step打印一次
            #     # print("\n" + "="*80)
            #     # print("                INSPECTING BATCH STRUCTURE (step 0)               ")
            #     # print("="*80)
            #     # print_batch_structure(batch)
            #     # print("="*80 + "\n")
            #     # import ipdb; ipdb.set_trace()
            #     for k, v in batch.items():
            #         print(f"{k}:{v}")
 
        else:
            visual_keys_used = cfg.dataset.visual_keys_used
            action_expert_key_used = [OBS_ROBOT, ACTION]
            needed_keys = [ACTION, OBS_ROBOT, TRAJ, OBS_TRAJ]
            # visual_keys_used = ["left_image", "ego_image", "top_image"]
            dtype = torch.float32

            # mapping all visual keys to the same size
            batch_size = batch["timestamp"].shape[0]
            batch_dataset_tmp = {}
            key_mapping = DATA_KEYS_MAPPING_HAND if "human_image" in batch.keys() else DATA_KEYS_MAPPING_ROBOT
            # mapping keys in batch_dataset to new keys according to key_mapping
            for new_key in visual_keys_used:
                old_key = key_mapping[new_key] if new_key in key_mapping else new_key
                if old_key in batch.keys():
                    batch_dataset_tmp[new_key] = resize_with_pad(batch[old_key], *cfg.policy.resize_imgs_with_padding, pad_value=0).to(device=accelerator.device)
                else:
                    # feature_shape = dataset.meta.features[new_key]['shape']
                    # new_shape = (batch_size, feature_shape[-1], *feature_shape[:-1])
                    new_shape = (batch_size, 3, *cfg.policy.resize_imgs_with_padding)
                    batch_dataset_tmp[new_key] = torch.zeros(new_shape, dtype=dtype, device=accelerator.device)

            for k, v in batch.items():
                if k not in visual_keys_used and 'image' not in k:
                    if isinstance(v, torch.Tensor):
                        batch_dataset_tmp[k] = v.to(accelerator.device)
                    else:
                        batch_dataset_tmp[k] = v
            batch = batch_dataset_tmp
            
        
        for key in needed_keys:
                batch[key] = normalize(batch[key], dataset.meta.stats[key], accelerator).to(dtype=torch.float32)
        # Convert the defaultdict to a standard dict to ensure DDP compatibility
        batch = dict(batch)

        # =======================> 在这里添加检查代码 <=======================
        # 只在训练的前几个步骤进行检查，以避免不必要的开销
        if current_step < 2:
            # 1. 从你的 batch 中选择一个有代表性的张量。
            #    最好是像 'index' 或 'frame_index' 这样的索引张量。
            #    如果你的 batch 中没有索引，任何数据张量（如 'actions'）都可以。
            # !!! 请将 'index' 替换为你 batch 中实际存在的键 !!!
            key_to_check = 'actions'
            if key_to_check not in batch:
                # 如果指定的键不存在，自动选择第一个张量进行检查
                key_to_check = next(k for k, v in batch.items() if isinstance(v, torch.Tensor))

            representative_tensor = batch[key_to_check]

            # 2. accelerator.gather() 必须由所有进程调用。
            #    它会收集所有 GPU 上的 representative_tensor 并将它们拼接起来。
            gathered_tensors = accelerator.gather(representative_tensor)

            # 3. 只在主进程 (rank 0) 上进行分析和打印，保持日志干净。
            if accelerator.is_main_process:
                print("\n" + "="*80)
                print(f"                Checking Batches on Step {current_step}                ")
                print(f"Checking uniqueness of tensor batch['{key_to_check}'] across {accelerator.num_processes} GPUs...")

                # 4. 将收集到的张量切分成每个 GPU 的数据块
                chunks = torch.chunk(gathered_tensors, accelerator.num_processes, dim=0)
                
                # 5. 比较每对数据块是否相同
                is_duplicated = False
                for i in range(len(chunks)):
                    for j in range(i + 1, len(chunks)):
                        if torch.equal(chunks[i], chunks[j]):
                            is_duplicated = True
                            print(f"❌ WARNING: Batch on GPU {i} and GPU {j} are IDENTICAL!")
                
                if not is_duplicated:
                    print("✅ OK: Batches on all GPUs are unique.")
                else:
                    print("❌ FAILED: Found duplicate batches. Your data distribution is not working correctly.")
                print("="*80 + "\n")

            # 等待所有进程完成这个检查步骤，再继续训练
            accelerator.wait_for_everyone()

        # ====================================================================

        train_tracker.dataloading_s = time.perf_counter() - start_time
        # Update policy
        train_tracker, output_dict = update_policy(
            train_tracker,
            policy,
            batch,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator,
            lr_scheduler=lr_scheduler,
        )
        
        # Increment step counter
        step += 1
        train_tracker.step()

        # print(f'detailed loss step:{step}, traj loss:{output_dict["trajectory_losses"]}, action loss:{output_dict["action_losses"]}')
        # detailed_loss = []
        # for k, v in output_dict.items():
        #     if "after" not in k:
        #         detailed_loss.append({k:v})
        # print(f'detailed loss step:{step}, {detailed_loss}')

        # Determine if we should log, save, or evaluate
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        # is_saving_step = True
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0
        # Save evaluation at the first step (for debugging purposes)
        # if step == 1:
        #     is_saving_step = True
        
        # Logging (only on main process)
        if is_log_step and accelerator.is_main_process:
            logging.info(train_tracker)
            wandb_log_dict = train_tracker.to_dict()
            if output_dict:
                wandb_log_dict.update(output_dict)
            for k, v in wandb_log_dict.items():
                accelerator.log({f"{'train'}/{k}": v}, step=step)
            train_tracker.reset_averages()

        # Checkpointing (only on main process)
        if cfg.save_checkpoint and is_saving_step:
            # Ensure all processes are synchronized before saving
            accelerator.wait_for_everyone()
            # Save checkpoint if on main process
            if accelerator.is_main_process:
                logging.info(f"Checkpoint policy after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.job_name, step)
                unwrapped_policy = accelerator.unwrap_model(policy)
                save_checkpoint(checkpoint_dir, step, cfg, unwrapped_policy, optimizer, lr_scheduler)
                update_last_checkpoint(checkpoint_dir)

                # save dataset.meta.stats into a json file in the checkpoint_dir
                save_stats = {}
                for k1,v1 in dataset.meta.stats.items():
                    save_stats[k1] = {}
                    for k2,v2 in v1.items():
                        if isinstance(v2, torch.Tensor):
                            save_stats[k1].update({k2: v2.cpu().tolist()})
                        elif isinstance(v2, np.ndarray):
                            save_stats[k1].update({k2: v2.tolist()})
                save_stats_path = os.path.join(checkpoint_dir, "pretrained_model")
                with open(os.path.join(save_stats_path, "dataset_stats.json"), "w") as f:
                    json.dump(save_stats, f, indent=4)


    # Wait for all processes to finish
    accelerator.wait_for_everyone()

    # Cleanup
    if eval_env and accelerator.is_main_process:
        eval_env.close()

    if accelerator.is_main_process:
        logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()