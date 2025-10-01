#!/usr/bin/env python

import json
import logging
import threading
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
import collections

import einops
import gymnasium as gym
import numpy as np
import torch
from termcolor import colored
from torch import Tensor, nn
from tqdm import trange, tqdm

import os 
import sys 
sys.path.insert(0, os.getcwd())

from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

# 导入openpi_client的组件
from openpi_client import image_tools

from lerobot.common.envs.factory import make_env
from lerobot.common.envs.utils import add_envs_task, check_env_attributes_and_types, preprocess_observation
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.io_utils import write_video
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

def _quat2axisangle(quat):
    """四元数转轴角

    Args:
        quat: 四元数
    Returns:
        轴角表示
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if np.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den

def rollout(
    env: OffScreenRenderEnv,
    policy: PreTrainedPolicy,
    task_description: str,
    num_steps_wait: int = 10,
    max_steps: int = 100,
    resize_size: int = 224,
    replan_steps: int = 5,
    return_observations: bool = False,
) -> dict:
    """运行一次策略评估

    Args:
        env: LIBERO环境
        policy: 策略模型
        task_description: 任务描述
        num_steps_wait: 等待物体稳定的步数
        max_steps: 最大步数
        resize_size: 图像调整大小
        replan_steps: 重新规划的步数
        return_observations: 是否返回观察数据
    """
    # 重置环境和策略
    obs = env.reset()
    policy.reset()
    device = next(policy.parameters()).device
    action_plan = collections.deque()
    
    all_observations = []
    all_actions = []
    all_rewards = []
    all_successes = []
    all_dones = []
    replay_images = []

    step = 0
    done = False
    
    while not done and step < max_steps + num_steps_wait:
        try:
            # 等待物体稳定
            if step < num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                step += 1
                continue

            # 处理图像 - 与main.py保持一致
            # IMPORTANT: rotate 180 degrees to match train preprocessing
            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(img, resize_size, resize_size)
            )
            wrist_img = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_img, resize_size, resize_size)
            )
            
            # 保存图像用于回放视频
            replay_images.append(img)
            
            if not action_plan:
                # 按 websocket_policy_server.py 的方式处理 observation
                observation = {}

                # 处理图像
                if img is not None:
                    img_tensor = torch.from_numpy(np.array(img)).float()
                    img_tensor = einops.rearrange(img_tensor, "h w c -> 1 c h w") / 255.0
                    observation["image"] = img_tensor.to(device)
                if wrist_img is not None:
                    wrist_img_tensor = torch.from_numpy(np.array(wrist_img)).float()
                    wrist_img_tensor = einops.rearrange(wrist_img_tensor, "h w c -> 1 c h w") / 255.0
                    observation["wrist_image"] = wrist_img_tensor.to(device)

                # 处理状态
                state_vec = np.concatenate(
                    (
                        obs["robot0_eef_pos"],                      #(3,)
                        _quat2axisangle(obs["robot0_eef_quat"]),    #(4,) -> (3,)
                        obs["robot0_gripper_qpos"],                 #(2,)
                    )
                )
                state_tensor = torch.from_numpy(state_vec).float().unsqueeze(0).to(device)
                observation["state"] = state_tensor

                # 处理提示词
                observation["task"] = [task_description]

                # 查询模型获取动作
                with torch.inference_mode():
                    action_chunk = policy.select_action(observation)
                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[:replan_steps].cpu().numpy())
                
            action = action_plan.popleft()
            # 执行动作 
            obs, reward, done, info = env.step(action)

            all_actions.append(action)
            all_rewards.append(reward)
            # all_successes.append(info.get("is_success", False))
            all_dones.append(done)
            
            if return_observations:
                all_observations.append(deepcopy(obs))

            if done:
                break
            step += 1

        except Exception as e:
            logging.error(f"Caught exception: {e}")
            break

    return {
        "observations": all_observations if return_observations else None,
        "actions": all_actions,
        "rewards": all_rewards,
        # "successes": all_successes,
        "dones": all_dones,
        "replay_images": replay_images,
        "final_success": done
    }

def eval_policy(
    env: OffScreenRenderEnv,
    policy: PreTrainedPolicy,
    n_episodes: int,
    task_description: str,
    max_steps: int,
    max_episodes_rendered: int = 0,
    videos_dir: Path | None = None,
    return_episode_data: bool = False,
    start_seed: int | None = None,
) -> dict:
    """评估策略在LIBERO环境中的表现。

    Args:
        env: LIBERO环境
        policy: 策略模型
        n_episodes: 评估的episode数量
        max_episodes_rendered: 最大渲染视频数量
        videos_dir: 视频保存目录
        return_episode_data: 是否返回episode数据
        start_seed: 起始随机种子
    Returns:
        包含评估结果的字典
    """
    if max_episodes_rendered > 0 and not videos_dir:
        raise ValueError("If max_episodes_rendered > 0, videos_dir must be provided.")

    if not isinstance(policy, PreTrainedPolicy):
        raise ValueError(
            f"Policy of type 'PreTrainedPolicy' is expected, but type '{type(policy)}' was provided."
        )

    start = time.time()
    policy.eval()

    # 记录指标
    sum_rewards = []
    max_rewards = []
    all_successes = []
    all_seeds = []
    threads = []  # 用于视频保存线程
    video_paths = []
    n_episodes_rendered = 0  # 已渲染的视频数量

    # 评估每个episode
    for episode in trange(n_episodes, desc="Running evaluation episodes"):
        if start_seed is not None:
            seed = start_seed + episode
            env.seed(seed)
        else:
            seed = None

        # 运行rollout
        rollout_data = rollout(
            env=env,
            policy=policy,
            task_description=task_description,  # 从环境获取任务描述
            num_steps_wait=10,
            max_steps=max_steps,
            resize_size=224,
            replan_steps=5
        )

        # 计算episode的奖励
        sum_reward = float(sum(rollout_data["rewards"]))
        max_reward = float(max(rollout_data["rewards"]) if rollout_data["rewards"] else 0.0)
        success = bool(rollout_data["final_success"])

        # 记录指标
        sum_rewards.append(sum_reward)
        max_rewards.append(max_reward)
        all_successes.append(success)
        all_seeds.append(seed)

        # 保存视频
        if rollout_data["replay_images"] and n_episodes_rendered < max_episodes_rendered:
            video_path = videos_dir / f"eval_episode_{n_episodes_rendered}.mp4"
            video_paths.append(str(video_path))
            thread = threading.Thread(
                target=write_video,
                args=(str(video_path), rollout_data["replay_images"], 10)
            )
            thread.start()
            threads.append(thread)
            n_episodes_rendered += 1

        # 输出当前进度
        running_success_rate = sum(all_successes) / len(all_successes) * 100
        tqdm.write(
            f"Episode {episode}: Success={success}, "
            f"Reward={sum_reward:.2f}, Steps={len(rollout_data['actions'])}, "
            f"Running success rate: {running_success_rate:.1f}%"
        )

    # 等待所有视频保存完成
    for thread in threads:
        thread.join()

    # 编译评估信息
    eval_duration = time.time() - start
    info = {
        "per_episode": [
            {
                "episode_ix": i,
                "sum_reward": sum_reward,
                "max_reward": max_reward,
                "success": success,
                "seed": seed,
            }
            for i, (sum_reward, max_reward, success, seed) in enumerate(
                zip(sum_rewards, max_rewards, all_successes, all_seeds)
            )
        ],
        "aggregated": {
            "avg_sum_reward": float(np.nanmean(sum_rewards)),
            "avg_max_reward": float(np.nanmean(max_rewards)),
            "pc_success": float(np.nanmean(all_successes) * 100),
            "eval_s": eval_duration,
            "eval_ep_s": eval_duration / n_episodes,
        }
    }
    if video_paths:
        info["video_paths"] = video_paths
    logging.info(f"Video paths: {video_paths}")
    logging.info(f"Evaluation complete in {eval_duration:.1f}s")
    logging.info(f"Average episode duration: {eval_duration / n_episodes:.1f}s")
    logging.info(f"Success rate: {info['aggregated']['pc_success']:.1f}%")
    logging.info(f"Average sum reward: {info['aggregated']['avg_sum_reward']:.2f}")
    
    return info

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    """主评估函数"""
    logging.info(pformat(asdict(cfg)))

    # 设置设备和随机种子
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # 初始化LIBERO任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # 初始化策略模型
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env,
    )
    policy.to(device)
    policy.eval()

    # 创建输出目录
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    # 评估每个任务
    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        for task_id in trange(num_tasks, desc="Evaluating tasks"):
            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            
            # 初始化环境
            env = OffScreenRenderEnv(
                bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
                camera_heights=LIBERO_ENV_RESOLUTION,
                camera_widths=LIBERO_ENV_RESOLUTION,
                camera_depths=True
            )
            env.task_description = task.language  # 添加任务描述到环境中

            # 评估当前任务
            info = eval_policy(
                env=env,
                policy=policy,
                n_episodes=cfg.eval.n_episodes,
                task_description=task.language,
                max_steps=cfg.env.max_steps,
                max_episodes_rendered=10,
                videos_dir=videos_dir,
                start_seed=cfg.seed + task_id * cfg.eval.n_episodes
            )

            # 保存当前任务的评估结果
            task_output_dir = output_dir / f"task_{task_id}"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            with open(task_output_dir / "eval_info.json", "w") as f:
                json.dump(info, f, indent=2)

            env.close()

    print("End of eval")

if __name__ == "__main__":
    init_logging()
    eval_main() 