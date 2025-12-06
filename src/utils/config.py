"""
配置文件加载工具    
"""

import yaml
from pathlib import Path
from typing import Dict, Any

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载YAML配置文件

    Args:
        config_path (str): 配置文件路径
    
    Returns:
        Dict[str, Any]: 配置文件内容
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config =  yaml.safe_load(f)
    
    return config



def save_config(config: Dict[str, Any], config_path: str):
    """
    保存配置文件

    Args:
        config (Dict[str, Any]): 配置文件内容
        config_path (str): 配置文件路径
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    
    print(f"配置文件已保存: {config_path}")