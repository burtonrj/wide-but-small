from collections import namedtuple
from types import SimpleNamespace
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.model_selection import BaseCrossValidator
from sklearn.model_selection import LeaveOneOut

from .data import TrainTestSplit
from .resample import CrossValidation
from .resample import Resample
from .utils import progress_bar


class Performance:
    def __init__(self):
        self.roc_auc_score_ = []
        self.f1_ = []
        self.roc_curve_ = []
        self.balanced_accuracy_ = []
        self.confusion_matrix_ = []

    def calculate_all(self, y_true: Iterable, y_pred: Iterable, y_score: Optional[Iterable] = None):
        self.balanced_accuracy_.append(balanced_accuracy_score(y_true=y_true, y_pred=y_pred))
        self.f1_.append(f1_score(y_true=y_true, y_pred=y_pred, average="macro"))
        self.confusion_matrix_.append(confusion_matrix(y_true=y_true, y_pred=y_pred))
        if y_score:
            self.roc_auc_score_.append(roc_auc_score(y_true=y_true, y_score=y_score))
            self.roc_curve_.append(roc_curve(y_true=y_true, y_score=y_score))
        return self

    @property
    def fields_(self):
        return self.__dict__.keys()


class NotEvaluatedError(Exception):
    def __init__(self):
        super(self).__init__("Call evaluate before trying to obtain performance")


class Model:
    def __init__(self, name: str, model):
        self.name: str = name
        self.model = model
        self.fitted: bool = False
        self.training_performance = Performance()
        self.testing_performance = Performance()
        self.holdout_performance = Performance()

    def get_performance(self, dataset: str, metric: str):
        try:
            if dataset == "training":
                if self.training_performance is None:
                    raise NotEvaluatedError()
                return getattr(self.training_performance, metric)
            if dataset == "testing":
                if self.testing_performance is None:
                    raise NotEvaluatedError()
                return getattr(self.testing_performance, metric)
            if dataset == "holdout":
                if self.holdout_performance is None:
                    raise NotEvaluatedError()
                return getattr(self.holdout_performance, metric)
            raise ValueError("dataset must be one of: 'training', 'testing', or 'holdout'")
        except AttributeError:
            raise AttributeError(f"Invalid metric, must be one of: {Performance.fields_}")

    def bootstrap_evaluation(self, outer_data: TrainTestSplit):
        pass

    def kfoldcv_evalulation(self, outer_data: TrainTestSplit, cv: BaseCrossValidator):
        pass

    def loocv(self, outer_data: TrainTestSplit, verbose: bool = True):
        self.training_performance = Performance()
        self.testing_performance = Performance()

        y_true_test = []
        y_pred_test = []
        y_score_test = []

        cv = CrossValidation(cross_validator=LeaveOneOut())
        for inner_data in progress_bar(
            cv.resample(data=outer_data), verbose=verbose, total=outer_data.training_data[0].shape[0]
        ):
            # Fit model
            x_train, y_train = inner_data.training_data
            self.model.fit(x_train, y_train)

            # Training score
            y_pred_training = self.model.predict(x_train)
            y_score_training = None
            if hasattr(self.model, "predict_proba"):
                y_score = self.model.predict_proba(x_train)
            self.training_performance.calculate_all(y_true=y_train, y_pred=y_pred_training, y_score=y_score_training)

            # Store test prediction
            x_test, y_test = inner_data.testing_data
            y_true_test.append(y_test[0])
            y_pred_test.append(self.model.predict(x_test)[0])
            if hasattr(self.model, "predict_proba"):
                y_score_test.append(self.model.predict_proba(x_test)[:, -1][0])
        y_score_test = y_score_test if len(y_score_test) > 0 else None
        self.testing_performance.calculate_all(y_true=y_true_test, y_pred=y_pred_test, y_score=y_score_test)

    def holdout_evaluation(self, outer_data: TrainTestSplit):
        self.holdout_performance = Performance()
        self.model.fit(outer_data.training_data[0], outer_data.training_data[1])
        y_pred = self.model.predict(outer_data.testing_data[0])
        y_true = outer_data.testing_data[1]
        y_score = None
        if hasattr(self.model, "predict_proba"):
            y_score = self.model.predict_proba(outer_data.testing_data[0])
        self.holdout_performance.calculate_all(y_true=y_true, y_pred=y_pred, y_score=y_score)

    def feature_importance(self):
        pass

    def calibration(self):
        pass


class ModelPipeline:
    def __init__(self, data: TrainTestSplit, features: List[str], resampling_method: Resample):
        self._models = {}
        self._locked: bool = False
        self.data: TrainTestSplit = data
        self.features: List[str] = features
        self.resampling_method: Resample = resampling_method

    def _resample(self):
        pass

    @property
    def models(self) -> Dict:
        return self._models

    def edit_models(self, new_models: Dict):
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


def confusion_matrix_plot():
    pass
