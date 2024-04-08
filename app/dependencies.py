import numpy as np
import pandas as pd
from fastapi_cache import FastAPICache

from app.database import SessionLocal
from mlds23_authorship_identification.utils import label2name, load_model


class Model:
    """
    ML-model with loading from s3 and postprocessing.
    """

    def __init__(self, model_name):
        self.model = load_model(model_name)

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
            proba = self.model.predict_proba(X)
            labels = np.where(
                np.max(proba, axis=1) > threshold, np.argmax(proba, axis=1), -1
            )
        else:
            labels = self.model.predict(X)
        if return_names:
            names = np.vectorize(label2name.get)(labels)
        if return_labels and return_names:
            return np.hstack((labels.reshape(-1, 1), names.reshape(-1, 1)), dtype=object)
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


def my_key_builder(func, *args, **kwargs):
    prefix = FastAPICache.get_prefix()
    cache_key = f"{prefix}:{func.__module__}:{func.__name__}"
    return cache_key
