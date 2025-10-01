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
from tqdm import trange

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
from openpi_client import websocket_client_policy as _websocket_client_policy

import sys
import os

sys.path.insert(0, os.getcwd())
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

def rollout(
    env: OffScreenRenderEnv,
    policy: _websocket_client_policy.WebsocketClientPolicy,
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
        policy: WebSocket客户端策略
        task_description: 任务描述
        num_steps_wait: 等待物体稳定的步数
        max_steps: 最大步数
        resize_size: 图像调整大小
        replan_steps: 重新规划的步数
        return_observations: 是否返回观察数据
    """
    # 重置环境和策略
    obs = env.reset()
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
                # 准备观察数据 - 与main.py完全一致
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate(
                        (
                            obs["robot0_eef_pos"],                      #(3,)
                            _quat2axisangle(obs["robot0_eef_quat"]),    #(4,) -> (3,)
                            obs["robot0_gripper_qpos"],                 #(2,)
                        )
                    ),
                    "prompt": str(task_description),
                }

                # 查询模型获取动作 - 与main.py完全一致

                action_chunk = policy.infer(element)["actions"]

                assert (
                    len(action_chunk) >= replan_steps
                ), f"We want to replan every {replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                action_plan.extend(action_chunk[:replan_steps])
                
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

@parser.wrap()
def eval_main(cfg: EvalPipelineConfig):
    """主评估函数"""
    logging.info(pformat(asdict(cfg)))

    # 设置设备和随机种子
    device = torch.device(cfg.policy.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    # 初始化LIBERO任务套件
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.eval.task_suite_name]()
    num_tasks = task_suite.n_tasks

    # 初始化WebSocket客户端
    client = _websocket_client_policy.WebsocketClientPolicy(
        host=cfg.host,
        port=cfg.port
    )

    # 创建输出目录
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = 0
    total_successes = 0

    # 评估每个任务
    for task_id in trange(num_tasks):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        
        # 初始化环境
        env = OffScreenRenderEnv(
            bddl_file_name=Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file,
            camera_heights=LIBERO_ENV_RESOLUTION,
            camera_widths=LIBERO_ENV_RESOLUTION,
            camera_depths=True
        )
        env.seed(cfg.seed)

        # 运行多个episode
        
        for episode in range(cfg.eval.n_episodes):
            # 设置初始状态
            env.set_init_state(initial_states[episode])
            # 运行rollout
            rollout_data = rollout(
                env=env,
                policy=client,
                task_description=task.language,
                num_steps_wait=10,
                max_steps=cfg.env.max_steps,
                resize_size=224,
                replan_steps=5
            )

            # 保存视频
            if rollout_data["replay_images"]:
                video_path = videos_dir / f"task_{task_id}_ep_{episode}_{rollout_data['final_success']}.mp4"
                write_video(str(video_path), rollout_data["replay_images"], fps=10)

            # 更新统计信息
            total_episodes += 1
            if rollout_data["final_success"]:
                total_successes += 1

            # 输出当前进度
            success_rate = total_successes / total_episodes * 100
            logging.info(f"Episode {total_episodes}: Success={rollout_data['final_success']}")
            logging.info(f"Current success rate: {success_rate:.1f}%")

    # 保存评估结果
    results = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": total_successes / total_episodes * 100
    }
    
    with open(output_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logging.info(f"Evaluation complete. Final success rate: {results['success_rate']:.1f}% | n_episodes={results['total_episodes']} | task_suite_name={cfg.eval.task_suite_name}")

if __name__ == "__main__":
    init_logging()
    eval_main() 