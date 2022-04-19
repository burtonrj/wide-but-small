from typing import List
from typing import Tuple

import pandas as pd
import pytest

from src.widebutsmall.data import CompleteCase
from src.widebutsmall.data import TrainTestSplit


@pytest.fixture(scope="function")
def missing_data() -> Tuple[pd.DataFrame, List[Tuple[str, List[int]]]]:
    return (
        pd.DataFrame(
            {
                "A": [1, 2, None, None],
                "B": [1, 2, None, None],
                "C": [1, 2, None, 4],
                "D": [1, None, 3, 4],
                "Target": [1, 2, 3, 4],
            }
        ),
        [("A,B,C", [0, 1]), ("A,B,C,D", [0]), ("D", [0, 2, 3]), ("C,D", [0, 3]), ("C", [0, 1, 3])],
    )


def test_complete_case_init(missing_data):
    complete_case = CompleteCase(data=missing_data[0], features=["A", "B", "C", "D"], target="Target")
    assert complete_case.features == ["A", "B", "C", "D"]
    assert complete_case.data.shape[0] == 4


def test_complete_case_search(missing_data):
    complete_case = CompleteCase(data=missing_data[0], features=["A", "B", "C", "D"], target="Target")
    complete_case.search()
    assert len(complete_case.datasets) == 5

    for features, members in missing_data[1]:
        dataset = [x for _, x in complete_case.datasets.items() if x["Features"] == set(features.split(","))]
        assert len(dataset) == 1
        assert dataset[0]["Members"] == set(members)


@pytest.mark.parametrize("min_members,expected_n", [(None, 5), (3, 2)])
def test_complete_case_summarise_datasets(missing_data, min_members, expected_n):
    complete_case = CompleteCase(data=missing_data[0], features=["A", "B", "C", "D"], target="Target")
    complete_case.search()
    summary = complete_case.summarise_datasets(min_members=min_members)
    assert summary.columns.tolist() == ["Dataset ID", "Property", "Count"]
    assert summary["Dataset ID"].nunique() == expected_n


def test_complete_case_get_dataset(missing_data):
    complete_case = CompleteCase(data=missing_data[0], features=["A", "B", "C", "D"], target="Target")
    complete_case.search()
    data = complete_case.get_dataset(2)
    assert data.shape == (2, 4)
    assert data.index.tolist() == [0, 1]
    assert set(data.columns.tolist()) == {"A", "B", "C", "Target"}


def test_traintestsplit(dummy_data):
    data = TrainTestSplit.from_dataframe(data=dummy_data, features=list(range(500)), target="Target", test_size=0.1)
    assert data.training_dataframe.shape == (90, 501)
    assert data.testing_dataframe.shape == (10, 501)
    assert data.training_data[0].shape == (90, 500)
    assert data.training_data[1].shape == (90,)
    assert data.testing_data[0].shape == (10, 500)
    assert data.testing_data[1].shape == (10,)
