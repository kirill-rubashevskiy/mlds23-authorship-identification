import logging
import asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton
from aiogram.filters.command import Command
from llama_cpp import Llama
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN")


bot = Bot(token=TOKEN)
dp = Dispatcher()

# Инициализация модели от LlamaCpp
model_path = "/Users/dariamishina/Downloads/openchat_3.5.Q4_K_M.gguf"
llm = Llama(model_path=model_path, n_ctx=8192, n_threads=8, n_gpu_layers=0)

# Переменные для статистики
ratings = []
usage_stats = {"total_requests": 0, "average_rating": 0}


# Обработчик команды /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Привет! Я бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. Введите свой вопрос после команды /test."
    )


# Обработчик текстовых сообщений
@dp.message(F.text, Command("test"))
async def generate_response(message: types.Message):
    global usage_stats
    usage_stats["total_requests"] += 1

    # Получение текста после команды /test
    command_text = message.text[len("/test") :].strip()

    # Проверка, что текст после команды существует
    if command_text:
        # generated_text = command_text + ' NEW!!!' #заглушка для теста бота
        # Генерация ответа с использованием модели
        input_text = command_text
        prompt = f"GPT4 Correct User: кто из русских классиков написал эти строки: {input_text}<|end_of_turn|>GPT4 Correct Assistant:"
        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=True)
        generated_text = output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[
            1
        ]
        # Отправка ответа
        await message.reply(generated_text)
        await message.answer(
            "С помощью команды /rate вы можете оценить качество ответа"
        )
    else:
        # В случае отсутствия текста после команды /test
        await message.reply("Напишите ваш запрос после команды /test")


# обработчик для команды /file
@dp.message(Command("file"))
async def process_file(message: types.Message):
    # Проверяем, есть ли прикрепленный файл
    if message.document:
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        file_path = file.file_path

        # Читаем содержимое файла и используем его текст для генерации ответа
        file_data = await bot.download_file(file_path)
        file_content = file_data.read()

        # Преобразуем байтовые данные в строку
        text_from_file = file_content.decode("utf-8").strip()

        # Генерация ответа
        prompt = f"GPT4 Correct User: кто из русских классиков написал эти строки: {text_from_file}GPT4 Correct Assistant:"
        output = llm(prompt, max_tokens=512, stop=["</s>"], echo=True)
        generated_text = output["choices"][0]["text"].split("GPT4 Correct Assistant: ")[
            1
        ]

        # Отправка ответа
        await message.reply(generated_text)
        await message.answer(
            "С помощью команды /rate вы можете оценить качество ответа"
        )
    else:
        await message.reply(
            "Пожалуйста, прикрепите файл формата txt после команды /file"
        )


# Обработчик команды /help
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


# Обработчик команды /rate
@dp.message(Command("rate"))
async def cmd_rate(message: types.Message):
    buttons = [KeyboardButton(text=str(i)) for i in range(1, 6)]
    keyboard = ReplyKeyboardMarkup(keyboard=[buttons], resize_keyboard=True)
    await message.answer("Пожалуйста, оцените качество ответа:", reply_markup=keyboard)


# Обработчик оценки от пользователя
@dp.message(lambda message: message.text.isdigit() and 1 <= int(message.text) <= 5)
async def process_rating(message: types.Message):
    global ratings, usage_stats
    rating = int(message.text)
    ratings.append(rating)

    # Пересчет среднего рейтинга
    total_ratings = sum(ratings)
    average_rating = total_ratings / len(ratings)
    usage_stats["average_rating"] = round(average_rating, 2)

    await message.answer(
        f"Спасибо за вашу оценку! Средний рейтинг: {usage_stats['average_rating']}"
    )


# Обработчик команды /stats
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
    logging.basicConfig(level=logging.INFO)
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
