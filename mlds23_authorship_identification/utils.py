from typing import Any

from dvc.api import open
from joblib import load


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
