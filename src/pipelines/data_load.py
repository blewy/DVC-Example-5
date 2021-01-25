from typing import Text
import yaml

from src.data.dataset import get_dataset


def data_load(config_path: Text) -> None:
    """ Lod Raw Data
    Args:
        config_path {Text} path to config

    """
    config = yaml.safe_load(open(config_path))
    dataset = get_dataset()
    dataset.to_csv(config['data_load']['dataset_csv'], index=False)
