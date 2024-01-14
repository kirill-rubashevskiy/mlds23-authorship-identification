import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters.command import Command
from llama_cpp import Llama
import os
from dotenv import load_dotenv
import boto3
from io import BytesIO

BUCKET_NAME = "mlds23-authorship-identification"
MODELS_DIR = "models/"

load_dotenv()
TOKEN = os.getenv("TOKEN")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")


bot = Bot(token=TOKEN)
dp = Dispatcher()

logging.basicConfig(level=logging.INFO)

# Инициализация модели от LlamaCpp - если загружаем с локалки
# model_path = "/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf"
# llm = Llama(model_path=model_path, n_ctx=8192, n_threads=8, n_gpu_layers=0)

# Выгрузка модели из s3
model_file_name = "openchat_3.5.Q4_K_M.gguf"

local_model_path = f"/Users/dariamishina/Downloads/{model_file_name}"

session = boto3.session.Session()
s3 = session.client(
    service_name="s3",
    endpoint_url="https://storage.yandexcloud.net",
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name="ru-central1",
)
logging.info("Инициализация клиента S3")

s3_key = f"{MODELS_DIR}{model_file_name}"


with BytesIO() as model_buffer:
    s3.download_fileobj(BUCKET_NAME, s3_key, model_buffer)
    model_buffer.seek(0)

    # Сохранение модели локально - пока не придумала как без этого
    with open(local_model_path, "wb") as local_model_file:
        local_model_file.write(model_buffer.read())
logging.info(f"Модель успешно загружена из S3")

# собственно сама модель
llm = Llama(model_path=local_model_path, n_ctx=8192, n_threads=8, n_gpu_layers=0)
logging.info(f"Модель инициализирована")

# Переменные для статистики
ratings = []
usage_stats = {"total_requests": 0, "average_rating": 0}


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. Введите свой вопрос после команды /test."
    )


@dp.message(F.text, Command("test"))
async def generate_response(message: types.Message):
    global usage_stats
    usage_stats["total_requests"] += 1

    command_text = message.text[len("/test") :].strip()
    if command_text:
        logging.info(f"Текст от пользователя получен")
        # generated_text = command_text + ' NEW!!!' #заглушка для теста бота
        input_text = command_text
        prompt = f"GPT4 Correct User: кто из русских классиков написал эти строки: {input_text}<|end_of_turn|>GPT4 Correct Assistant:"
        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=True)
        logging.info(f"ответ от llm готов")
        generated_text = output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[
            1
        ]
        await message.reply(generated_text)
        logging.info(f"ответ отправлен")
        await message.answer(
            "С помощью команды /rate вы можете оценить качество ответа"
        )
    else:
        await message.reply("Напишите ваш запрос после команды /test")


@dp.message(Command("file"))
async def process_file(message: types.Message):
    if message.document:
        logging.info(f"Файл от пользователя получен")
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path
        file_data = await bot.download_file(file_path)
        file_content = file_data.read()
        logging.info(f"Файл прочитан")
        text_from_file = file_content.decode("utf-8").strip()
        logging.info(f"Преобразованы байтовые данные в строку")
        prompt = f"GPT4 Correct User: кто из русских классиков написал эти строки: {text_from_file}GPT4 Correct Assistant:"
        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=True)
        logging.info(f"ответ от llm готов")
        generated_text = output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[
            1
        ]
        await message.reply(generated_text)
        logging.info(f"ответ отправлен")
        await message.answer(
            "С помощью команды /rate вы можете оценить качество ответа"
        )
    else:
        await message.reply(
            "Пожалуйста, прикрепите файл формата txt после команды /file"
        )


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = """
    Это бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. У него есть следующие команды: 
    - /start - начать работу; 
    - /test - отправить текст отрывка для определения авторства текста; 
    - /file - отправить файл в формате txt c отрывком для определения авторства текста; 
    - /rate - оценить работу бота;
    - /stats — получить статистику работы бота
    - /help — получить список команд бота
    
    TL;DR: просто отправьте текстовый фрагмент после команды /test, и я постараюсь ответить.
    """
    await message.answer(help_text)


@dp.message(Command("rate"))
async def cmd_rate(message: types.Message):
    buttons = [KeyboardButton(text=str(i)) for i in range(1, 6)]
    keyboard = ReplyKeyboardMarkup(keyboard=[buttons], resize_keyboard=True)
    await message.answer("Пожалуйста, оцените качество ответа:", reply_markup=keyboard)


@dp.message(lambda message: message.text.isdigit() and 1 <= int(message.text) <= 5)
async def process_rating(message: types.Message):
    global ratings, usage_stats
    rating = int(message.text)
    ratings.append(rating)

    logging.info(f"пересчет срднего рейтинга")
    total_ratings = sum(ratings)
    average_rating = total_ratings / len(ratings)
    usage_stats["average_rating"] = round(average_rating, 2)

    await message.answer(
        f"Спасибо за вашу оценку! Средний рейтинг: {usage_stats['average_rating']}"
    )


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message):
    global usage_stats
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
