from typing import Tuple, List

import pandas as pd

from src.widebutsmall.data import CompleteCase
import pytest


@pytest.fixture(scope="function")
def missing_data() -> Tuple[pd.DataFrame, List[Tuple[str, List[int]]]]:
    return (
        pd.DataFrame(
            {"A":      [1,     2,      None,     None],
             "B":      [1,     2,      None,     None],
             "C":      [1,     2,      None,     4],
             "D":      [1,     None,   3,        4],
             "Target": [1,     2,      3,        4]}
        ),
        [
            ("A,B,C", [0, 1]),
            ("A,B,C,D", [0]),
            ("D", [0, 2, 3]),
            ("C,D", [0, 3]),
            ("C", [0, 1, 3])
        ]
    )


def test_complete_case_init(missing_data):
    complete_case = CompleteCase(
        data=missing_data[0], features=["A", "B", "C", "D"], target="Target"
    )
    assert complete_case.features == ["A", "B", "C", "D"]
    assert complete_case.data.shape[0] == 4


def test_complete_case_search(missing_data):
    complete_case = CompleteCase(
        data=missing_data[0], features=["A", "B", "C", "D"], target="Target"
    )
    complete_case.search()
    assert len(complete_case.datasets) == 5

    for features, members in missing_data[1]:
        dataset = [x for _, x in complete_case.datasets.items() if x["Features"] == set(features.split(","))]
        assert len(dataset) == 1
        assert dataset[0]["Members"] == set(members)

