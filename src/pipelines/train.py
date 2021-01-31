import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from typing import Text, Dict
import yaml
import argparse
import joblib
import os


class UnsupportedClassifier(Exception):

    def __init__(self, estimator_name):
        self.msg = f'Unsupported estimator {estimator_name})'
        super().__init__(self.msg)


def getsupported_estimator() -> Dict:
    """
    Returns
        Dict: Supported classifiers
    """

    return {
        'logreg': LogisticRegression,
        'svm': SVC,
        'knn': KNeighborsClassifier}


def train(df: pd.DataFrame, target_column: Text,
          estimator_name: Text,
          param_grid: Dict, cv: int):
    """:param

    """

    estimators = getsupported_estimator()

    if estimator_name not in estimators.keys():
        raise UnsupportedClassifier(estimator_name)

    estimator = estimators[estimator_name]()
    f1_scorer = make_scorer(f1_score, average='weighted')
    clf = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        verbose=1,
        cv=cv,
        scoring=f1_scorer)

    # Get X and Y
    y_train = df.loc[:, target_column].values.astype('int32')
    x_train = df.drop(target_column, axis=1).values.astype('float32')

    clf.fit(x_train, y_train)

    return clf


def train_lr(config_path: Text) -> None:
    config = yaml.safe_load(open(config_path))
    estimator_name = config['train']['estimator_name']
    param_grid = config['train']['estimators'][estimator_name]['param_grid']
    cv = config['train']['cv']
    target_column = config['featurize']['target_column']
    train_df = pd.read_csv(config['data_split']['train_path'])

    model = train(
        df=train_df,
        target_column=target_column,
        estimator_name=estimator_name,
        param_grid=param_grid,
        cv=cv
    )
    print(model.best_score_)

    model_name = config['base']['model']['model_name']
    model_folder = config['base']['model']['models_folder']

    joblib.dump(
        model,
        os.path.join(model_folder, model_name)
    )


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--config", dest='config', required=True)
    args = args_parser.parse_args()

    train_lr(config_path=args.config)
