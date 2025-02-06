import json
from typing import Dict

import joblib
import kserve
import numpy as np


class XGBoostPreprocessor(kserve.KFModel):
    def __init__(self, name: str, storage_uri: str):
        super().__init__(name)
        self.name = name
        self.storage_uri = storage_uri
        self.ready = False
        self.imputer = None
        self.scaler = None
        self.feature_names = None

    def load(self):
        """Load preprocessors and feature information."""
        # Load preprocessing objects
        preprocess_objects = joblib.load(f"{self.storage_uri}/feature_info.json")
        self.imputer = preprocess_objects["imputer"]
        self.scaler = preprocess_objects["scaler"]

        # Load feature information
        with open(f"{self.storage_uri}/feature_info.json", "r") as f:
            feature_info = json.load(f)
            self.feature_names = feature_info["feature_names"]

        self.ready = True

    def preprocess(self, inputs: Dict) -> Dict:
        """Preprocess the input data."""
        instances = inputs.get("instances", [])

        # Convert to numpy array
        input_array = np.array(instances)

        # Apply preprocessing steps
        if self.imputer:
            input_array = self.imputer.transform(input_array)
        if self.scaler:
            input_array = self.scaler.transform(input_array)

        # Create feature interactions (same as in training)
        processed_data = self._create_interactions(input_array)

        return {"instances": processed_data.tolist()}

    def _create_interactions(self, X: np.ndarray) -> np.ndarray:
        """Create interaction features between selected features."""
        # Convert to feature names if available
        if self.feature_names:
            base_features = self.feature_names[:5]
            interactions = []

            for i in range(len(base_features)):
                for j in range(i + 1, len(base_features)):
                    idx1 = self.feature_names.index(base_features[i])
                    idx2 = self.feature_names.index(base_features[j])
                    interaction = X[:, idx1] * X[:, idx2]
                    interactions.append(interaction)

            if interactions:
                interaction_matrix = np.column_stack(interactions)
                X = np.column_stack([X, interaction_matrix])

        return X


if __name__ == "__main__":
    model = XGBoostPreprocessor("xgboost-preprocessor", "/mnt/models")
    kserve.KFServer(workers=1).start([model])
