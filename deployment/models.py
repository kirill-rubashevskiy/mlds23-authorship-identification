# работа с S3
import os
import boto3
import pickle
from io import BytesIO


# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_ACCESS_KEY_ID = 'REDACTED_KEY_ID_1'
AWS_SECRET_ACCESS_KEY = 'REDACTED_SECRET_KEY_1'

BUCKET_NAME = "mlds23-authorship-identification"
MODEL_NAME = 'kr-26-11-23-exp-1_pipeline.pkl'
MODELS_DIR = 'models/'


def load_model():

    session = boto3.session.Session()

    s3 = session.client(
        service_name='s3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name='ru-cental1'
    )

    with BytesIO() as data:
        s3.download_fileobj(Bucket=BUCKET_NAME, Key=MODELS_DIR + MODEL_NAME, Fileobj=data)
        data.seek(0)
        model = pickle.load(data)

    return model
