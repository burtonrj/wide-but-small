from sklearn.datasets import make_classification
import pandas as pd
import pytest


@pytest.fixture(scope="function")
def dummy_data() -> pd.DataFrame:
    x, y = make_classification(
        n_samples=100,
        n_features=500,
        n_informative=20,
        n_redundant=100,
        n_repeated=10,
        n_classes=2,
        weights=(0.2,),
        random_state=42
    )
    x = pd.DataFrame(x)
    x["Target"] = y
    return x
