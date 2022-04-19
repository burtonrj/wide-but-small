from abc import ABC
from abc import abstractmethod
from typing import Generator

from sklearn.model_selection import BaseCrossValidator

from .data import TrainTestSplit


class Resample(ABC):
    @abstractmethod
    def resample(self, data: TrainTestSplit) -> Generator[TrainTestSplit, None, None]:
        ...


class Bootstrap(Resample):
    def __init__(self, sample_frac: float, random_state: int = 42, n_rounds: int = 50):
        self.sample_frac = sample_frac
        self.random_state = random_state
        self.n_rounds = n_rounds

    def resample(self, data: TrainTestSplit) -> Generator[TrainTestSplit, None, None]:
        training_data = data.training_dataframe.reset_index(drop=True).copy()
        for _ in range(self.n_rounds):
            sample = training_data.sample(frac=self.sample_frac)
            train_idx = sample.index.values
            test_idx = training_data[~training_data.index.isin(train_idx)].index.values
            yield TrainTestSplit(
                data=training_data,
                features=data.features,
                target=data.target,
                train_index=train_idx,
                test_index=test_idx,
            )


class CrossValidation(Resample):
    def __init__(self, cross_validator: BaseCrossValidator):
        self._cv = cross_validator

    def resample(self, data: TrainTestSplit) -> Generator[TrainTestSplit, None, None]:
        training_data = data.training_dataframe.reset_index(drop=True).copy()
        for train_index, test_index in self._cv.split(X=data.training_data[0], y=data.training_data[1]):
            yield TrainTestSplit(
                data=training_data,
                features=data.features,
                target=data.target,
                train_index=train_index,
                test_index=test_index,
            )
