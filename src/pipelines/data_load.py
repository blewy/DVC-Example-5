from typing import Text
import yaml
import argparse

from src.data.dataset import get_dataset


def data_load(config_path: Text) -> None:
    """ Lod Raw Data
    Args:
        config_path {Text} path to config

    """
    config = yaml.safe_load(open(config_path))
    dataset = get_dataset()
    dataset.to_csv(config['data_load']['dataset_csv'], index=False)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest='config', required=True)
    args = args_parser.parse_args()
    
    data_load(config_path=args.config)
