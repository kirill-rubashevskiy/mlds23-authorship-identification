# работа с S3
import boto3
import pickle
from io import BytesIO


BUCKET_NAME = "mlds23-authorship-identification"
MODELS_DIR = 'models/'
MODEL_NAME = 'kr-26-11-23-exp-1_pipeline.pkl'

session = boto3.session.Session()

s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id='REDACTED_KEY_ID_1',
    aws_secret_access_key='REDACTED_SECRET_KEY_1',
    region_name='ru-cental1'
)

with BytesIO() as data:
    s3.download_fileobj(Bucket=BUCKET_NAME, Key=MODELS_DIR + MODEL_NAME, Fileobj=data)
    data.seek(0)
    baseline_model = pickle.load(data)