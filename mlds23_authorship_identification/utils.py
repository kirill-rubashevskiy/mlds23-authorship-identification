from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from dvc.api import open
from joblib import load

import wandb


def load_model(model_name: str) -> Any:
    """
    Loads DVC-tracked ML-model from remote S3.
    Args:
        model_name: model name

    Returns: model
    """

    with open(
        path=f"models/{model_name}",
        repo="https://github.com/kirill-rubashevskiy/mlds23-authorship-identification/",
        mode="rb",
        remote="yandexcloudhttp",
    ) as f:
        model = load(f)
    return model


def load_data(data_name: str) -> Any:
    """
    Loads DVC-tracked data from remote S3.

    Args:
        data_name: data name
    Returns: data
    """

    with open(
        path=f"data/{data_name}",
        repo="https://github.com/kirill-rubashevskiy/mlds23-authorship-identification/",
        mode="rb",
        remote="yandexcloudhttp",
    ) as f:
        data = pd.read_csv(f, index_col=0)
    return data


def save_model(model_name: str, model):
    # if models/tmp dir does not exist — create it
    models_tmp_dir = Path("models/tmp")
    if not models_tmp_dir.exists():
        models_tmp_dir.mkdir()

    # save model
    filename = "_".join(model_name.lower().split()) + ".joblib"
    joblib.dump(model, f"models/tmp/{filename}")


def wb_log(**data_to_log):
    if "config" in data_to_log:
        for key, value in data_to_log["config"].items():
            wandb.config[key] = value

    if "params" in data_to_log:
        params_to_log = dict()
        for key, value in data_to_log["params"].items():
            *_, step, param = key.split("__")
            if step not in params_to_log:
                params_to_log[step] = dict()
            params_to_log[step][param] = value
        for step, params in params_to_log.items():
            wandb.config[step] = params

    if "scores" in data_to_log:
        wandb.log(data_to_log["scores"])


label2name = {
    -1: "не могу определить автора",
    0: "А. Пушкин",
    1: "Д. Мамин-Сибиряк",
    2: "И. Тургенев",
    3: "А. Чехов",
    4: "Н. Гоголь",
    5: "И. Бунин",
    6: "А. Куприн",
    7: "А. Платонов",
    8: "В. Гаршин",
    9: "Ф. Достоевский",
}
