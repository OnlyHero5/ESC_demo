"""
随机种子工具    
"""


import random
import numpy as np
import torch



def set_seed(seed: int = 42):
    """设置所有随机种子确保可复现

    Args:
        seed (int, optional): 随机种子. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 强制 cuDNN 使用确定性算法
    # 禁用 cuDNN 的自动调优（auto-tuner）
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    print(f"随机种子设置为: {seed}")
