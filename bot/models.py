import numpy as np
import pandas as pd

from data_preproc.src.utils import preprocess_text5
from mlds23_authorship_identification.utils import (
    confident_predict,
    label2name,
    load_model,
)


class Estimator:
    def __init__(
        self,
        service_name: str,
        endpoint_url: str,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        region_name: str,
        bucket: str,
        models_dir: str,
        model_name: str,
    ):
        self.model = load_model(
            service_name,
            endpoint_url,
            aws_access_key_id,
            aws_secret_access_key,
            region_name,
            bucket,
            models_dir,
            model_name,
        )
        self.preprocessing = preprocess_text5
        self.postprocessing = confident_predict
        self.label2name = label2name

    def predict_item(self, item: str) -> str:
        preprocessed_item = self.preprocessing(item)
        preprocessed_item = np.array([preprocessed_item])
        proba = self.model.predict_proba(preprocessed_item)
        predicted_label = self.postprocessing(proba)
        if predicted_label != -1:
            prediction = (
                f"Кажется, этот фрагмент написал {self.label2name[predicted_label]}"
            )
        else:
            prediction = "Я не могу уверенно определить автора данного фрагмента"

        return prediction

    def predict_items(self, items: pd.DataFrame) -> pd.Series:
        preprocessed_items = items["text"].apply(self.preprocessing)
        proba = self.model.predict_proba(preprocessed_items)
        predicted_labels = np.apply_along_axis(self.postprocessing, 1, proba)
        predictions = pd.Series(predicted_labels, name="predictions").apply(
            lambda x: self.label2name[x]
        )

        return predictions
