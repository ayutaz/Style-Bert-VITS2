"""パス設定ファイル読み込み"""

import shutil
from pathlib import Path

import yaml

from style_bert_vits2.logging import logger


class PathConfig:
    def __init__(self, dataset_root: str, assets_root: str):
        self.dataset_root = Path(dataset_root)
        self.assets_root = Path(assets_root)


def get_path_config() -> PathConfig:
    path_config_path = Path("configs/paths.yml")
    if not path_config_path.exists():
        shutil.copy(src="configs/default_paths.yml", dst=path_config_path)
        logger.info(
            f"A configuration file {path_config_path} has been generated based on the default configuration file default_paths.yml."
        )
        logger.info(
            "Please do not modify configs/default_paths.yml. Instead, modify configs/paths.yml."
        )
    with open(path_config_path, encoding="utf-8") as file:
        path_config_dict: dict[str, str] = yaml.safe_load(file.read())
    return PathConfig(**path_config_dict)
