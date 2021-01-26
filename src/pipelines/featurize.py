from typing import Text
import argparse
import pandas as pd
import yaml

from src.features.features import extract_features


def featurize(config_path=Text) -> None:
    """ Lod Raw Data
        Args:
            config_path {Text} path to config

        """
    config = yaml.safe_load(open(config_path))
    dataset = pd.read_csv(config['data_load']['dataset_csv'])
    featured_dataset = extract_features(dataset)
    filepath = config['featurize']['features_path']
    featured_dataset.to_csv(filepath, index=False)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest='config', required=True)
    args = args_parser.parse_args()

    featurize(config_path=args.config)
