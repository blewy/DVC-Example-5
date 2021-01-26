from typing import Text
import argparse
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split


def data_split(config_path: Text) -> None:
    """ Lod Raw Data
        Args:
            config_path {Text} path to config

        """
    config = yaml.safe_load(open(config_path))
    dataset = pd.read_csv(config['data_load']['dataset_csv'])
    train_dataset, test_dataset = train_test_split(dataset,
                                                   test_size=config['data_split']['test_size'],
                                                   random_state=config['base']['random_state'])

    trains_csv_path = config['data_split']['train_path']
    test_csv_path = config['data_split']['test_path']
    train_dataset.to_csv(trains_csv_path, index=False)
    test_dataset.to_csv(test_csv_path, index=False)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest='config', required=True)
    args = args_parser.parse_args()

    data_split(config_path=args.config)
