#!/usr/bin/env python


import json
import logging

from dataclasses import asdict
from pprint import pformat
import os 
import sys 
sys.path.insert(0, os.getcwd())
import torch
from lerobot.common.datasets.factory import make_dataset
import numpy as np
from termcolor import colored

from lerobot.common.ds_meta import ds_meta_offline

from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
    inside_slurm,
)
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.random_utils import set_seed
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs import parser
from lerobot.scripts.websocket.websocket_policy_server import WebsocketPolicyServer
# 导入msgpack_numpy用于数据序列化

import functools

import msgpack
import numpy as np

def pack_array(obj):
    if (isinstance(obj, (np.ndarray, np.generic))) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj

Packer = functools.partial(msgpack.Packer, default=pack_array)
packb = functools.partial(msgpack.packb, default=pack_array)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=unpack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


        
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
    # stats = json.load(open("/mnt/sda/zhouhan/dataset/ours_franka_ok/meta/stats.json", "r"))
    # stats = to_tensor(stats)
    # dataset = make_dataset(cfg)

    # load ds_meta from file
    import json
    # load ds_meta from file
    ds_meta_file = os.path.join(cfg.policy.pretrained_path, "dataset_stats.json") if cfg.policy and cfg.policy.pretrained_path else None
    if ds_meta_file and os.path.exists(ds_meta_file):
        with open(ds_meta_file, "r") as f:
            ds_meta_stats = json.load(f)
        # convert to tensor
        ds_meta_stats = {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in ds_meta_stats.items()}
    ds_meta_offline['stats'] = ds_meta_stats
    print(f'Inference policy config:{cfg.policy}')
    policy = make_policy(
        cfg=cfg.policy,
        # ds_meta=dataset.meta,
        ds_meta_offline=ds_meta_offline
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