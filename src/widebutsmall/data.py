from collections import defaultdict
from typing import List, Optional, Dict, Set

import pandas as pd


class Imputed:
    def __init__(self):
        pass


class CompleteCase:
    def __init__(self, data: pd.DataFrame, target: str, features: List[str]):
        self.data = data[~data[target].isnull()].copy()
        self.features = features
        self.datasets: Optional[Dict[int, Dict[str, Set]]] = None

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
            i + 1: {
                "Members": set(members),
                "Features": set(features.split(","))
            }
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
                    "Number of samples": len(dataset_info.get("Members"))
                }
            )
        summary_of_datasets = pd.DataFrame(summary_of_datasets)

        if min_members:
            summary_of_datasets = summary_of_datasets[summary_of_datasets["Number of samples"] >= min_members]

        return summary_of_datasets.melt(
            id_vars="Dataset ID", var_name="Property", value_name="Count"
        ).sort_values("Count")



class TrainTestSplit:
    def __init__(self):
        pass
