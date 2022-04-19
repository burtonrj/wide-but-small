import logging
from abc import ABC
from abc import abstractmethod
from typing import Callable
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
import pandas as pd
import pingouin as pg
from genetic_selection import GeneticSelectionCV
from mrmr import mrmr_classif
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.feature_selection import SelectFdr
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from skrebate import MultiSURF

from src.widebutsmall.data import TrainTestSplit
from src.widebutsmall.modelling import Model
from src.widebutsmall.modelling import Performance
from src.widebutsmall.resample import Bootstrap
from src.widebutsmall.utils import progress_bar

logger = logging.getLogger(__name__)


class Features:
    def __init__(self, creator: str, ranked_names: List[str], weights: List[float]):
        self.creator = creator
        self.ranked_names = ranked_names
        self.weights = weights
        self.performance = None

    def top(self, n: int, as_set: bool = False):
        if as_set:
            return set(self.ranked_names[:n])
        return self.ranked_names[:n]

    def as_set(self):
        return set(self.ranked_names)


def stability(feature_sets: List[Features], n_selected: int, original_n: int) -> Tuple[float, np.ndarray]:
    pairwise_ki = []
    for fs_i in feature_sets:
        fs_i = fs_i.top(n=n_selected, as_set=True)
        for fs_j in feature_sets:
            fs_j = fs_j.top(n=n_selected, as_set=True)
            r = len(fs_i.intersection(fs_j))
            ki = (r - (n_selected**2 / original_n)) / (n_selected - (n_selected**2 / original_n))
            pairwise_ki.append(ki)

    return (2 * sum(pairwise_ki)) / (len(feature_sets) * (len(feature_sets) - 1)), np.array(pairwise_ki)


def aggregate_features(feature_sets: List[Features]) -> List[str]:
    pass


def aggregate_features_with_weights(feature_sets: List[Features]) -> List[str]:
    pass


class FeatureSelectionWrapper(ABC):
    @abstractmethod
    def fit_predict(self, data: TrainTestSplit) -> Features:
        pass


class ReliefWrapper(FeatureSelectionWrapper):
    name = "MultiSURF"

    def __init__(self, n_features: Optional[int] = None, discrete_limit: int = 5, n_jobs: int = -1):
        self.selector = MultiSURF(n_features_to_select=n_features, discrete_threshold=discrete_limit, n_jobs=n_jobs)
        self.feature_importances_ = None

    def fit_predict(self, data: TrainTestSplit) -> Features:
        self.selector.fit(*data.training_data)
        features_weights = [(name, weight) for name, weight in zip(data.features, self.selector.feature_importances_)]
        features_weights = sorted(features_weights, key=lambda x: x[1])[::-1]
        self.feature_importances_ = self.selector.feature_importances_
        return Features(
            creator=self.name, ranked_names=[x[0] for x in features_weights], weights=[x[1] for x in features_weights]
        )


class MRMRWrapper(FeatureSelectionWrapper):
    name = "MRMR"

    def __init__(
        self,
        relevance: Union[str, Callable] = "f",
        redundancy: Union[str, Callable] = "c",
        categorical_features: Optional[List[str]] = None,
        n_jobs: int = -1,
    ):
        self.relevance = relevance
        self.redundancy = redundancy
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs

    def fit_predict(self, data: TrainTestSplit) -> Features:
        feature_ranking, relevance, redundancy = mrmr_classif(
            data.training_dataframe[data.features],
            data.training_dataframe[data.target],
            K=len(data.features),
            return_scores=True,
            n_jobs=self.n_jobs,
            relevance=self.relevance,
            redundancy=self.redundancy,
            cat_features=self.categorical_features,
        )
        weights = np.linspace(0, 1, len(feature_ranking))[::-1]
        return Features(ranked_names=feature_ranking, weights=list(weights), creator=self.name)


class GAWrapper(FeatureSelectionWrapper):
    name = "Genetic Algorithm"

    def __init__(
        self,
        estimator=None,
        cv: int = 5,
        scoring: str = "roc_auc",
        n_jobs: int = -1,
        n_population: int = 300,
        n_generations: int = 40,
        caching: bool = True,
        **kwargs
    ):
        estimator = estimator if estimator is not None else LogisticRegression()
        self.selector = GeneticSelectionCV(
            estimator=estimator,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            n_population=n_population,
            n_generations=n_generations,
            caching=caching,
            **kwargs
        )

    def fit_predict(self, data: TrainTestSplit) -> Features:
        self.selector.fit(*data.training_data)
        features = list(np.array(data.features)[self.selector.support_])
        weights = [1 for _ in range(len(features))] + [0 for _ in range(len(data.features) - len(features))]
        return Features(
            creator=self.name, ranked_names=features + [x for x in data.features if x not in features], weights=weights
        )


class RFEWrapper(FeatureSelectionWrapper):
    name = "RFE"

    def __init__(self, estimator=None, n_features_to_select: float = 5):
        estimator = estimator if estimator is not None else LinearSVC(random_state=42)
        self.selector = RFE(estimator=estimator, n_features_to_select=n_features_to_select)

    def fit_predict(self, data: TrainTestSplit) -> Features:
        self.selector.fit(*data.training_data)
        feature_weight = [(name, weight) for name, weight in zip(data.features, self.selector.ranking_)]
        feature_weight = sorted(feature_weight, key=lambda x: x[1])
        features = [x[0] for x in feature_weight]
        weights = [len(data.features) - x[1] for x in feature_weight]
        return Features(creator=self.name, ranked_names=features, weights=weights)


def ttest(X, y):
    scores = []
    p_vals = []
    classes = np.unique(y)
    for i in range(X.shape[1]):
        x1 = X[np.where(y == classes[0]), i][0]
        x2 = X[np.where(y == classes[1]), i][0]
        stats = pg.ttest(x1, x2, paired=False, correction=True)
        scores.append(stats.iloc[0]["T"])
        p_vals.append(stats.iloc[0]["p-val"])
    return np.array(scores), np.array(p_vals)


class TtestWrapper(FeatureSelectionWrapper):
    name = "FDR T-test"

    def __init__(self, score_method: str = "t"):
        self.selector = SelectFdr(score_func=ttest, alpha=0.05)
        self.score_method = score_method

    def fit_predict(self, data: TrainTestSplit) -> Features:
        self.selector.fit(*data.training_data)
        if self.score_method == "t":
            feature_weight = [(name, weight) for name, weight in zip(data.features, self.selector.scores_)]
        else:
            weights = 1 - self.selector.pvalues_
            feature_weight = [(name, weight) for name, weight in zip(data.features, weights)]
        feature_weight = sorted(feature_weight, key=lambda x: x[1])[::-1]

        return Features(
            creator=self.name, ranked_names=[x[0] for x in feature_weight], weights=[x[1] for x in feature_weight]
        )


class MutualInfoWrapper(FeatureSelectionWrapper):
    name = "Mutual Information"

    def __init__(self, discrete_features: Optional[np.ndarray] = None, n_neighbors: int = 5, random_state: int = 42):
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit_predict(self, data: TrainTestSplit) -> Features:
        mi = mutual_info_classif(
            *data.training_data,
            discrete_features=self.discrete_features,
            random_state=self.random_state,
            n_neighbors=self.n_neighbors
        )
        feature_weight = [(name, weight) for name, weight in zip(data.features, mi)]
        feature_weight = sorted(feature_weight, key=lambda x: x[1])[::-1]

        return Features(
            creator=self.name, ranked_names=[x[0] for x in feature_weight], weights=[x[1] for x in feature_weight]
        )


class DAWrapper(FeatureSelectionWrapper):
    def fit_predict(self, data: TrainTestSplit) -> Features:
        pass


class PermutationWrapper(FeatureSelectionWrapper):
    def fit_predict(self, data: TrainTestSplit) -> Features:
        pass


class FeatureSelectionPipeline:
    def __init__(
        self,
        n_features: int,
        outer_data: TrainTestSplit,
        sample_frac: float = 0.9,
        n_rounds: int = 50,
        random_state: int = 42,
        oob_evaluator: Optional = None,
    ):
        self.n_features = n_features
        self.outer_data = outer_data
        self.sampler = Bootstrap(sample_frac=sample_frac, n_rounds=n_rounds, random_state=random_state)
        self.feature_sets = []
        self.oob_evaluator = oob_evaluator
        self.n_rounds = n_rounds

    def fit(self, methods: List[FeatureSelectionWrapper], verbose: bool = True):
        for i, sample in progress_bar(
            enumerate(self.sampler.resample(data=self.outer_data)), verbose=verbose, total=self.n_rounds
        ):
            for fs_method in methods:
                optimal_features = fs_method.fit_predict(data=sample)
                if self.oob_evaluator is not None:
                    self.oob_evaluator.fit(
                        sample.training_dataframe[optimal_features.ranked_names].values, sample.training_data[1]
                    )
                    x_test, y_test = sample.testing_data[1]
                    y_pred = self.oob_evaluator.predict(x_test)
                    y_score = None
                    if hasattr(self.oob_evaluator, "predict_proba"):
                        y_score = self.oob_evaluator.predict_proba(x_test)
                    optimal_features.performance = Performance().calculate_all(
                        y_true=y_test, y_pred=y_pred, y_score=y_score
                    )
                self.feature_sets.append(optimal_features)

    def stability(self, n_selected: int) -> Tuple[float, np.ndarray]:
        return stability(
            feature_sets=self.feature_sets, n_selected=n_selected, original_n=len(self.outer_data.features)
        )


def plot_feature_ranking():
    pass
