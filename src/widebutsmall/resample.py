from typing import Union, List
from abc import ABC, abstractmethod
from .data import TrainTestSplit


class Resample(ABC):

    def resample(self, data: TrainTestSplit) -> Union[TrainTestSplit, List[TrainTestSplit]]:
        ...


class SMOTEENN(Resample):

    def resample(self, data: TrainTestSplit) -> TrainTestSplit:
        pass


class Bootstrap(Resample):

    def resample(self, data: TrainTestSplit) -> List[TrainTestSplit]:
        pass


class KFoldCV(Resample):

    def resample(self, data: TrainTestSplit) -> List[TrainTestSplit]:
        pass


class LOOCV(Resample):

    def resample(self, data: TrainTestSplit) -> List[TrainTestSplit]:
        pass
