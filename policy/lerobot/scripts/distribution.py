import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    """设置分布式训练环境"""
    # 自动检测可用的 GPU 数量
    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No GPUs available for distributed training.")

    # 设置必要的环境变量
    os.environ["MASTER_ADDR"] = "localhost"  # 主节点地址
    os.environ["MASTER_PORT"] = "54321"     # 主节点端口
    os.environ["WORLD_SIZE"] = str(num_gpus)  # 总进程数
    os.environ["RANK"] = os.environ.get("RANK", "0")  # 当前进程的 rank，默认为 0

    # 初始化分布式进程组
    dist.init_process_group(backend="nccl", init_method="env://")

def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()