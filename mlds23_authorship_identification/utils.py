import pickle
from io import BytesIO

import boto3
import numpy as np


def load_model(
    service_name: str,
    endpoint_url: str,
    aws_access_key_id: str,
    aws_secret_access_key: str,
    region_name: str,
    bucket: str,
    models_dir: str,
    model_name: str,
):
    """
    Load model from s3 storage.

    :param service_name:
    :param endpoint_url:
    :param aws_access_key_id:
    :param aws_secret_access_key:
    :param region_name:
    :param bucket:
    :param models_dir:
    :param model_name:
    :return:
    """
    session = boto3.session.Session()
    s3 = session.client(
        service_name=service_name,
        endpoint_url=endpoint_url,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        region_name=region_name,
    )
    with BytesIO() as data:
        s3.download_fileobj(Bucket=bucket, Key=f"{models_dir}{model_name}", Fileobj=data)
        data.seek(0)
        model = pickle.load(data)
    return model


label2name = {
    -1: "не знаю",
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


def confident_predict(proba: np.ndarray, threshold: float = 0.4) -> int:
    if np.max(proba) >= threshold:
        return np.argmax(proba)
    return -1
