from collections import namedtuple
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin

from .data import TrainTestSplit
from .resample import Resample

Performance = namedtuple("Performance", "f1 acc conf_matrix roc_curve roc_auc_score")


class NotEvaluatedError(Exception):
    super().__init__("Call evaluate before trying to obtain performance")


class Model:
    def __init__(self, name: str, model: ClassifierMixin):
        self.name: str = name
        self.model: ClassifierMixin = model
        self.fitted: bool = False
        self.training_performance: Optional[Performance] = None
        self.testing_performance: Optional[Performance] = None
        self.holdout_performance: Optional[Performance] = None

    def get_performance(self, dataset: str, metric: str):
        try:
            if dataset == "training":
                if self.training_performance is None:
                    raise NotEvaluatedError()
            if dataset == "testing":
                if self.testing_performance is None:
                    raise NotEvaluatedError()
            if dataset == "holdout":
                if self.holdout_performance is None:
                    raise NotEvaluatedError()
            raise ValueError("dataset must be one of: 'training', 'testing', or 'holdout'")
        except AttributeError:
            raise AttributeError(f"Invalid metric, must be one of: {Performance._fields}")

    def evaluate(
        self, train_test: TrainTestSplit, holdout: TrainTestSplit
    ) -> List[Dict[str, float], Dict[str, float]]:
        if not self.fitted:
            raise ValueError("Cannot assess the performance of a model that has not been fitted")
        # Todo: compute F1 score (macro)
        # Todo: compute roc curve
        # Todo: compute auc score
        # Todo: compute balanced accuracy
        # Todo: compute confusion matrix
        # self.training_performance = Performance()
        # self.testing_performance = Performance()
        # self.holdout_performance = Performance()

        return [{}, {}]

    def feature_importance(self):
        pass

    def calibration(self):
        pass


class ModelPipeline:
    def __init__(self, data: TrainTestSplit, features: List[str], resampling_method: Resample):
        self._models: Dict[str, ClassifierMixin] = {}
        self._locked: bool = False
        self.data: TrainTestSplit = data
        self.features: List[str] = features
        self.resampling_method: Resample = resampling_method

    def _resample(self):
        pass

    @property
    def models(self) -> Dict[str, ClassifierMixin]:
        return self._models

    def edit_models(self, new_models: Dict[str, ClassifierMixin]):
        pass

    def train(self):
        pass

    def performance(self):
        pass

    def top_performers(self) -> List[Model]:
        pass


def mcnemar_test(model1: Model, model2: Model, x: np.ndarray, y: np.ndarray) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pass


def calibration_plot():
    pass


def roc_plot():
    pass


def box_swarm_plot():
    pass


def confusion_matrix():
    pass
