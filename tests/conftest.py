import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture(scope="function")
def dummy_data() -> pd.DataFrame:
    x, y = make_classification(
        n_samples=100,
        n_features=200,
        n_informative=50,
        n_redundant=50,
        n_classes=2,
        weights=(0.3,),
        random_state=42,
        class_sep=3.0,
        shuffle=False,
    )
    x = pd.DataFrame(x)
    x["Target"] = y
    return x
