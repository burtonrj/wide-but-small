import logging

import numpy as np
import pytest
from sklearn.preprocessing import MinMaxScaler

from src.widebutsmall.data import TrainTestSplit
from src.widebutsmall.select import Features
from src.widebutsmall.select import GAWrapper
from src.widebutsmall.select import MRMRWrapper
from src.widebutsmall.select import ReliefWrapper
from src.widebutsmall.select import RFEWrapper
from src.widebutsmall.select import TtestWrapper

logging.basicConfig(format="%(asctime)s %(levelname)-8s %(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)


def test_features():
    features = Features(creator="test", ranked_names=["A", "B", "C"], weights=[1, 2, 3])
    assert features.performance is None
    assert features.top(n=2) == ["A", "B"]
    assert features.top(n=2, as_set=True) == {"A", "B"}
    assert features.as_set() == {"A", "B", "C"}


@pytest.mark.parametrize(
    "method,kwargs",
    [
        (ReliefWrapper, dict(n_features=200, n_jobs=-1, discrete_limit=5)),
        (MRMRWrapper, {}),
        (GAWrapper, {}),
        (RFEWrapper, {}),
        (TtestWrapper, {}),
    ],
)
def test_feature_selection_wrapper(dummy_data, method, kwargs):
    dummy_data[list(range(200))] = MinMaxScaler().fit_transform(dummy_data[list(range(200))])
    data = TrainTestSplit.from_dataframe(data=dummy_data, features=list(range(200)), target="Target")
    selector = method(**kwargs)
    features = selector.fit_predict(data=data)
    perc_noise = len([x for x in features.top(n=50) if x in list(range(100, 200))])
    logger.info(f"[{selector.name}] - Percentage noise in top 50: {(perc_noise / 50) * 100}%")
    assert isinstance(features, Features)
    assert features.weights[0] == max(features.weights)
    assert features.weights[199] == min(features.weights)
