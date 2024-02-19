import asyncio
import logging

# from llama_cpp import Llama
import os

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters.command import Command
from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from dotenv import load_dotenv

# import boto3
# from io import BytesIO
from model import LLMWrapper


BUCKET_NAME = "mlds23-authorship-identification"
MODELS_DIR = "models/"

load_dotenv()
TOKEN = os.getenv("TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# Инициализация модели от LlamaCpp - если загружаем с локалки
model_path = "/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf"
llm_wrapper = LLMWrapper(model_path=model_path)

bot = Bot(token=TOKEN)
dp = Dispatcher(
    llm_wrapper=llm_wrapper,
    ratings=[],
    usage_stats={"total_requests": 0, "average_rating": 0},
)

logging.basicConfig(level=logging.INFO)

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
# logging.info("Инициализация клиента S3")
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
# logging.info(f"Модель успешно загружена из S3")
#
# # собственно сама модель
# llm_wrapper = LLMWrapper(model_path=local_model_path)
# logging.info(f"Модель инициализирована")


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. Введите свой вопрос после команды /predict."
    )


@dp.message(F.text, Command("predict"))
async def cmd_predict(message: types.Message, llm_wrapper, usage_stats: dict):
    usage_stats["total_requests"] += 1
    command_text = message.text.removeprefix("/predict")
    generated_text = llm_wrapper.predict(command_text)
    await message.reply(generated_text)
    logging.info("ответ отправлен")


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "Это бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. У него есть следующие команды:\n"
        "- /start - начать работу;\n "
        "- /predict - отправить текст отрывка для определения авторства текста;\n "
        "- /predict_from_file - отправить файл в формате txt c отрывком для определения авторства текста;\n "
        "- /rate - оценить работу бота;\n "
        "- /stats — получить статистику работы бота\n "
        "- /help — получить список команд бота\n "
        "TL;DR: просто отправьте текстовый фрагмент после команды /predict, и я постараюсь ответить."
    )
    await message.answer(help_text)


@dp.message(Command("rate"))
async def cmd_rate(message: types.Message):
    buttons = [KeyboardButton(text=str(i)) for i in range(1, 6)]
    keyboard = ReplyKeyboardMarkup(keyboard=[buttons], resize_keyboard=True)
    await message.answer("Пожалуйста, оцените качество ответа:", reply_markup=keyboard)


@dp.message(lambda message: message.text.isdigit() and 1 <= int(message.text) <= 5)
async def process_rating(message: types.Message, ratings: list, usage_stats: dict):
    rating = int(message.text)
    ratings.append(rating)

    logging.info("пересчет среднего рейтинга")
    total_ratings = sum(ratings)
    average_rating = total_ratings / len(ratings)
    usage_stats["average_rating"] = round(average_rating, 2)

    await message.answer(
        f"Спасибо за вашу оценку! Средний рейтинг: {usage_stats['average_rating']}"
    )


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message, usage_stats: dict):
    stats_text = (
        f"Статистика использования сервиса:\n"
        f"Общее количество запросов: {usage_stats['total_requests']}\n"
        f"Средний рейтинг: {usage_stats['average_rating']}"
    )
    await message.answer(stats_text)


async def main():
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

# TODO
# - добавить еще пару моделей на выбор (скорей всего возьму сайгу и какой-нибудь микстраль)
# - ускорить инференс обязательно, тут проблема именно в gguf формате, такие модели отвечают дольше, чем модели без квантизации (gguf формат выбран для первой версии из-за экономии места)
# - пока размещено у меня на локалке, надо переезжать на сервак с гпу и там уже см п 2
# - целевая картина - дообученная LLM на русской классике, пока бот - почти заглушка и дз по прикладному питону
