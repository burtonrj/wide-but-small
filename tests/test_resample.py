import numpy as np
import pytest
from sklearn.model_selection import KFold
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from src.widebutsmall.data import TrainTestSplit
from src.widebutsmall.resample import Bootstrap
from src.widebutsmall.resample import CrossValidation


def test_bootstrap(dummy_data):
    outer_data = TrainTestSplit.from_dataframe(
        data=dummy_data, features=list(range(200)), target="Target", test_size=0.1
    )
    bootstrap_sampler = Bootstrap(sample_frac=0.9, random_state=42, n_rounds=50)
    samples = list(bootstrap_sampler.resample(data=outer_data))
    assert len(samples) == 50
    assert all([isinstance(x, TrainTestSplit) for x in samples])
    assert all([x.data.shape[0] == 90 for x in samples])
    assert all([x.training_dataframe.shape[0] == 81 for x in samples])
    assert all([x.testing_dataframe.shape[0] == 9 for x in samples])
    assert not all([np.array_equal(x.train_index, x.test_index) for x in samples])


@pytest.mark.parametrize(
    "cv",
    [
        KFold(n_splits=5),
        LeaveOneOut(),
        ShuffleSplit(n_splits=5, test_size=0.25, random_state=42),
        StratifiedKFold(n_splits=5),
        StratifiedShuffleSplit(n_splits=5, random_state=42, test_size=0.25),
    ],
)
def test_crossvalidation(dummy_data, cv):
    cross_val = CrossValidation(cross_validator=cv)
    outer_data = TrainTestSplit.from_dataframe(
        data=dummy_data, features=list(range(200)), target="Target", test_size=0.1
    )
    samples = list(cross_val.resample(data=outer_data))

    if isinstance(cv, LeaveOneOut):
        assert len(samples) == outer_data.training_dataframe.shape[0]
    else:
        assert len(samples) == 5

    assert all([isinstance(x, TrainTestSplit) for x in samples])
    assert all([x.data.shape[0] == 90 for x in samples])
    assert all([x.testing_dataframe.shape[0] < x.training_dataframe.shape[0] for x in samples])
    assert not all([np.array_equal(x.train_index, x.test_index) for x in samples])
