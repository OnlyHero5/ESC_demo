"""GRPO 训练启动脚本"""
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.grpo_trainer import run_grpo_training

def main():
    parser = argparse.ArgumentParser(description="GRPO 训练")
    parser.add_argument("--config", type=str, default="configs/rl_extes_grpo.yaml")
    args = parser.parse_args()
    
    run_grpo_training(args.config)

if __name__ == "__main__":
    main()
