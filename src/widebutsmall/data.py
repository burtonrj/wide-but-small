from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit


class Imputed:
    def __init__(self):
        pass


class CompleteCase:
    def __init__(self, data: pd.DataFrame, target: str, features: List[str]):
        self.data = data[~data[target].isnull()].copy()
        self.features = features
        self.datasets: Optional[Dict[int, Dict[str, Set]]] = None
        self.target = target

    def search(self):
        missing = self.data[self.features].isnull()
        feature_sets = []
        datasets = {}

        # Find all possible sets
        for i, row_i in missing.iterrows():
            for j, row_j in missing.iterrows():
                features_i = row_i[~row_i].index.tolist()
                features_j = row_j[~row_j].index.tolist()
                matched_features = set(features_i).intersection(set(features_j))
                if len(matched_features) == 0:
                    continue
                feature_sets.append(matched_features)

        # Find all members of each set
        for features in feature_sets:
            datasets[",".join(features)] = set(self.data[~self.data[features].isnull().any(axis=1)].index.values)

        self.datasets: Dict[int, Dict[str, Set]] = {
            i + 1: {"Members": set(members), "Features": set(features.split(","))}
            for i, (features, members) in enumerate(datasets.items())
        }
        return self

    def summarise_datasets(self, min_members: Optional[int] = None) -> pd.DataFrame:
        summary_of_datasets = []
        for dataset_id, dataset_info in self.datasets.items():
            summary_of_datasets.append(
                {
                    "Dataset ID": dataset_id,
                    "Number of features": len(dataset_info.get("Features")),
                    "Number of samples": len(dataset_info.get("Members")),
                }
            )
        summary_of_datasets = pd.DataFrame(summary_of_datasets)

        if min_members:
            summary_of_datasets = summary_of_datasets[summary_of_datasets["Number of samples"] >= min_members]

        return summary_of_datasets.melt(id_vars="Dataset ID", var_name="Property", value_name="Count").sort_values(
            "Count"
        )

    def plot_dataset_summary(
        self, min_members: Optional[int] = None, datasets: Optional[List[int]] = None, **kwargs
    ) -> plt.Axes:
        data = self.summarise_datasets(min_members=min_members)
        if datasets:
            data = data[data["Dataset ID"].isin(datasets)]
        ax = sns.barplot(data=data, x="Dataset ID", y="Count", hue="Property", **kwargs)
        return ax

    def get_dataset(self, dataset_id: int) -> pd.DataFrame:
        return self.data[list(self.datasets[dataset_id]["Features"]) + [self.target]].loc[
            self.datasets[dataset_id]["Members"]
        ]


class TrainTestSplit:
    def __init__(
        self, data: pd.DataFrame, features: List[str], target: str, random_state: int = 42, test_size: float = 0.2
    ):
        self.data = data.copy()
        self.features = features
        self.target = target
        spliter = StratifiedShuffleSplit(n_splits=2, test_size=test_size, random_state=random_state)
        self.train_index, self.test_index = next(spliter.split(data[features].values, data[target].values))

    def _dataframe(self, idx: np.ndarray):
        return self.data.iloc[idx][self.features + [self.target]]

    def _arrays(self, idx: np.ndarray):
        return self.data[self.features].values[idx], self.data[self.target].values[idx]

    @property
    def training_dataframe(self) -> pd.DataFrame:
        return self._dataframe(idx=self.train_index)

    @property
    def testing_dataframe(self) -> pd.DataFrame:
        return self._dataframe(idx=self.test_index)

    @property
    def training_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._arrays(idx=self.train_index)

    @property
    def testing_data(self) -> Tuple[np.ndarray, np.ndarray]:
        return self._arrays(idx=self.test_index)
