from typing import Any

import numpy as np
import pandas as pd
from dvc.api import open
from joblib import load

from app.database import SessionLocal
from mlds23_authorship_identification.utils import label2name


def load_model(model_name: str) -> Any:
    """
    Function loads DVC-tracked ML-model from remote S3.

    :param model_name: model name
    :return: model
    """
    with open(
        path=f"models/{model_name}",
        repo="https://github.com/kirill-rubashevskiy/mlds23-authorship-identification/",
        mode="rb",
        remote="yandexcloudhttp",
    ) as f:
        model = load(f)
    return model


class Model:
    """
    ML-model with loading from s3 and postprocessing.
    """

    def __init__(self, model_name, **kwargs):
        self.model = load_model(model_name, **kwargs)

    def predict(
        self,
        X: pd.Series,
        threshold: float | bool = 0.5,
        return_labels: bool = True,
        return_names: bool = False,
    ) -> np.ndarray:
        """
        Function predicts authors of texts and returns predicted labels and/or author's names.
        If threshold is set in none of the classes probabilities exceed it, model returns -1 (indicator that it cannot
        confidently predict the author).

        :param X: pd Series with single column with texts
        :param threshold: threshold
        :param return_labels: whether to return predicted labels
        :param return_names:  whether to return predicted author's names
        :return: predicted labels and/or author's names
        """
        if threshold:
            labels = self.model.predict(X)
        else:
            proba = self.model.predict_proba(X)
            labels = np.where(
                np.max(proba, axis=1) > threshold, np.argmax(proba, axis=1), -1
            )
        if return_names:
            names = np.vectorize(label2name.get)(labels)
        if return_labels and return_names:
            return np.hstack((labels, names), dtype=object)
        elif return_names:
            return names
        else:
            return labels


def get_db():
    """
    Function creates a new SQLAlchemy SessionLocal that will be used in a single request, and then close it once
    the request is finished.

    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
