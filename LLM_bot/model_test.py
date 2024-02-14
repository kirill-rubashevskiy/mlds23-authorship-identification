from llama_cpp import Llama
import boto3
from io import BytesIO
BUCKET_NAME = "mlds23-authorship-identification"
MODELS_DIR = 'models/'

# Загрузка модели в s3
# model_file_path = f'/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf'
#
# session = boto3.session.Session()
#
# s3 = session.client(
#     service_name='s3',
#     endpoint_url='https://storage.yandexcloud.net',
#     aws_access_key_id='YCAJErlaldUmioGbHQSqJ70MR',
#     aws_secret_access_key='YCPSba_JgloNYSNWcnKO2CYCEB8PFR1Iwgr2jIUy',
#     region_name='ru-cental1'
# )
#
# print('Инициализация клиента S3')

# model_file_name = model_file_path.split('/')[-1]
# s3_key = f'{MODELS_DIR}{model_file_name}'
#
#
# with open(model_file_path, 'rb') as model_file:
#     s3.upload_fileobj(model_file, BUCKET_NAME, s3_key)
#
# print(f'Модель успешно загружена в S3. Путь: s3://{BUCKET_NAME}/{s3_key}')

# Выгрузка модели из s3
model_file_name = 'openchat_3.5.Q4_K_M.gguf'

local_model_path = f'/Users/dariamishina/Downloads/{model_file_name}'

session = boto3.session.Session()
s3 = session.client(
    service_name='s3',
    endpoint_url='https://storage.yandexcloud.net',
    aws_access_key_id='YCAJErlaldUmioGbHQSqJ70MR',
    aws_secret_access_key='YCPSba_JgloNYSNWcnKO2CYCEB8PFR1Iwgr2jIUy',
    region_name='ru-central1'
)
print('Инициализация клиента S3')

s3_key = f'{MODELS_DIR}{model_file_name}'


with BytesIO() as model_buffer:
    s3.download_fileobj(BUCKET_NAME, s3_key, model_buffer)
    model_buffer.seek(0)

    # Сохранение модели локально - пока не придумала как без этого
    with open(local_model_path, 'wb') as local_model_file:
        local_model_file.write(model_buffer.read())
print(f'Модель успешно загружена из S3')

# собственно сама модель
llm = Llama(
    model_path=local_model_path,
    n_ctx=8192,
    n_threads=8,
    n_gpu_layers=0
)
print(f'Модель инициализирована')


# Set gpu_layers to the number of layers to offload to GPU. Set to 0 if no GPU acceleration is available on your system.
# llm = Llama(
#     model_path="/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf",
#     n_ctx=8192,  # The max sequence length to use - note that longer sequence lengths require much more resources
#     n_threads=8,  # The number of CPU threads to use, tailor to your system and the resulting performance
#     n_gpu_layers=0,  # The number of layers to offload to GPU, if you have GPU acceleration available
# )


output = llm(
    #   "GPT4 Correct User: какая ты модель<|end_of_turn|>GPT4 Correct Assistant:", # Prompt для теста
    "GPT4 Correct User: кто написал эти строки: Мороз и солнце, день чудесный! <|end_of_turn|>GPT4 Correct Assistant:",
    max_tokens=512,  # Generate up to 512 tokens
    stop=[
        "</s>"
    ],  # Example stop token - not necessarily correct for this specific model! Please check before using.
    echo=True,  # Whether to echo the prompt
)
print(output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[1])
