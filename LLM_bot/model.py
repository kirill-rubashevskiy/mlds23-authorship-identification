import logging

from llama_cpp import Llama


# from io import BytesIO
# import boto3


BUCKET_NAME = "mlds23-authorship-identification"
MODELS_DIR = "models/"
logging.basicConfig(level=logging.INFO)


class LLMWrapper:
    def __init__(self, model_path, n_ctx=8192, n_threads=8, n_gpu_layers=0):
        self.llm = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
        )

    def predict(self, command_text):
        input_text = command_text
        prompt = f"GPT4 Correct User: кто из русских классиков написал эти строки: {input_text}GPT4 Correct Assistant:"
        output = self.llm(prompt, max_tokens=512, stop=["</s>"], echo=True)
        logging.info("ответ от llm готов")
        generated_text = output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[1]
        return generated_text


# Инициализация модели от LlamaCpp - если загружаем с локалки
# model_path = "/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf"
# llm_wrapper = LLMWrapper(model_path=model_path)
# command_text = "В начале было слово."
# generated_text = llm_wrapper.predict(command_text)
# print(generated_text)

# # Выгрузка модели из s3
# model_file_name = "openchat_3.5.Q4_K_M.gguf"
#
# local_model_path = f"/Users/dariamishina/Downloads/{model_file_name}"
#
# session = boto3.session.Session()
# s3 = session.client(
#     service_name="s3",
#     endpoint_url="https://storage.yandexcloud.net",
#     aws_access_key_id=AWS_ACCESS_KEY_ID,
#     aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#     region_name="ru-central1",
# )
#
# s3_key = f"{MODELS_DIR}{model_file_name}"
#
#
# with BytesIO() as model_buffer:
#     s3.download_fileobj(BUCKET_NAME, s3_key, model_buffer)
#     model_buffer.seek(0)
#
#     # Сохранение модели локально - пока не придумала как без этого
#     with open(local_model_path, "wb") as local_model_file:
#         local_model_file.write(model_buffer.read())
#
# # собственно сама модель
# llm_wrapper = LLMWrapper(model_path=local_model_path)
# command_text = "В начале было слово."
# generated_text = llm_wrapper.predict(command_text)
# print(generated_text)
