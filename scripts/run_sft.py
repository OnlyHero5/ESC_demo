"""
SFT 训练启动脚本
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.sft_trainer import run_sft_training

def main():
    parser = argparse.ArgumentParser(description="ESConv SFT 训练")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sft_esconv.yaml",
        help="配置文件路径"
    )
    
    args = parser.parse_args()
    
    # 运行训练
    run_sft_training(args.config)


if __name__ == "__main__":
    main()