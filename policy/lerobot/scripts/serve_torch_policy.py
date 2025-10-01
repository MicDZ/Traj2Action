#!/usr/bin/env python

import dataclasses
import enum
import json
import logging
import socket
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
from dataclasses import asdict
from pprint import pformat
import asyncio
import traceback

import torch
import tyro
import numpy as np
from torch import nn
import websockets
import websockets.asyncio.server
import websockets.frames
import einops
from termcolor import colored

import os 
import sys 
sys.path.insert(0, os.getcwd())
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs import parser
from lerobot.scripts.websocket.websocket_policy_server import WebsocketPolicyServer
from lerobot.common.ds_meta import ds_meta_offline
# 导入msgpack_numpy用于数据序列化
try:
    from openpi_client import msgpack_numpy
except ImportError:
    # 如果没有openpi_client，使用简单的json序列化
    import json
    class msgpack_numpy:
        @staticmethod
        def Packer():
            return json
        @staticmethod
        def unpackb(data):
            return json.loads(data)

            
    def _json_serializer(self, obj):
        """JSON序列化辅助函数"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return str(obj)

def to_tensor(d):
    if isinstance(d, dict):
        return {k: to_tensor(v) for k, v in d.items()}
    elif isinstance(d, list):
        return torch.tensor(d)
    else:
        return d
    
@parser.wrap()
def main(cfg: EvalPipelineConfig) -> None:
    """主函数"""
    logging.info(pformat(asdict(cfg)))
    
    # 检查设备可用性
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    
    logging.info("Making policy.")

    # 在同步环境中创建模型
    # stats = json.load(open(cfg.stats_path))
    # stats = to_tensor(stats)
    policy = make_policy(
        cfg=cfg.policy,
        env_cfg=cfg.env, 
        ds_meta_offline=ds_meta_offline,
        # stats=stats
    )
    
    logging.info("Policy created successfully.")
    
    # 创建服务器
    server = WebsocketPolicyServer(
        policy=policy,
        host=cfg.host or "127.0.0.1",
        port=cfg.port or 8000,
        device=str(device),
        metadata={"model_type": "torch_policy", "device": str(device)}
    )
    
    logging.info(f"Starting server on {server._host}:{server._port}")
    
    # 启动服务
    server.serve_forever()

if __name__ == "__main__":
    init_logging()
    main()