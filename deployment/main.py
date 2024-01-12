import asyncio
import logging
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters.command import Command
import numpy as np
import time
from sklearn.pipeline import Pipeline
from typing import Callable
import pandas as pd
import io
from aiogram.types import BufferedInputFile
from aiohttp import web
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
import os
from aiogram.utils.keyboard import ReplyKeyboardBuilder

from models import load_model
from utils import confident_predict, label2name, preprocess_text5

TOKEN = os.getenv('TOKEN')
WEBHOOK_HOST = "https://mlds23-authorship-identification.onrender.com"
WEBHOOK_PATH = f'/webhook/{TOKEN}'
WEBHOOK_URL = f'{WEBHOOK_HOST}{WEBHOOK_PATH}'
WEB_SERVER_HOST = '0.0.0.0'
WEB_SERVER_PORT = 10000

# Объект бота
bot = Bot(token=TOKEN)
# Диспетчер
dp = Dispatcher(bot=bot)
dp['preprocessing'] = preprocess_text5
dp['model'] = load_model()
dp['postprocessing'] = label2name
dp['stats'] = {
    'users': set(),
    'requests': [0],
    'rate_sum': [0],
    'rate_num': [0]
}


# Хэндлер на команду /start
@dp.message(Command("start"))
async def cmd_start(message: types.Message, stats: dict):
    stats['users'].add(message.from_user.id)
    content = """
    Привет! Это бот для определения авторства текстов

    Я могу определить, кто из русских классиков написал один или несколько фрагментов текста

    Для получения подсказок по моей работе можно отправить /help
    """

    await message.reply(content)


@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    content = """
    Бот поддерживает следующие режимы/команды:
    - /help — получить подсказки по работе бота
    - /predict_item — отправить фрагмент текста и получить предсказание об авторе фрагмента
    - /predict_items — отправить csv-файл, в котором есть столбец "text" c фрагментами текстов, и получить csv-файл с предсказаниями по каждому фрагменту
    - /rate — оценить работу бота
    - /start — запустить бота
    - /stats — получить статистику работы бота
    
    - отправить csv-файл, в котором есть столбец "text" c фрагментами текстов — получить csv-файл с предсказаниями по каждому фрагменту
    """

    await message.reply(content)


@dp.message(Command("rate"))
async def cmd_rate(message: types.Message):
    builder = ReplyKeyboardBuilder()
    for i in range(1, 6):
        builder.add(
            types.KeyboardButton(
                text=str(i),
                callback_data=f'num_{str(i)}'
            )
        )

    await message.answer(
        "Оцените работу бота:",
        reply_markup=builder.as_markup(resize_keyboard=True),
    )

@dp.message(F.text.in_({str(i) for i in range(1, 6)}))
async def rate_callback(message: types.Message, stats: dict):
    stats['rate_sum'][0] += int(message.text)
    stats['rate_num'][0] += 1
    await message.reply("Cпасибо, что оценили работу бота!")


@dp.message(Command("stats"))
async def cmd_stats(message: types.Message, stats: dict):
    if stats['rate_num'][0] > 0:
        mean_rate = np.round(stats["rate_sum"][0] / stats["rate_num"][0], 2)
        content = (
            f'За время работы бота {len(stats["users"])} пользователь(а/ей) использовали его {stats["requests"][0]} раз(а)\n'
            f'Средняя оценка работы бота — {mean_rate}'
        )
    else:
        content = f'За время работы бота {len(stats["users"])} юзер(а/ов) использовали его {stats["requests"][0]} раз(а)'

    await message.reply(content)


@dp.message(Command("predict_item"))
async def predict_item(message: types.Message,
                       preprocessing: Callable,
                       model: Pipeline,
                       postprocessing: dict,
                       stats: dict):
    user_request = message.text[len('/predict_item'):]
    logging.info(user_request)
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Message: {message}')

    user_request_preprocessed = preprocessing(user_request)
    logging.info('Данные предобработаны')

    proba = model.predict_proba(np.array([user_request_preprocessed]))
    logging.info('Модель сделала предсказание')

    label = confident_predict(proba)
    if label != -1:
        reply = f'Кажется, этот фрагмент написал {postprocessing[label]}'
    else:
        reply = f'Я не могу уверенно определить автора данного фрагмента'
    logging.info('Предсказания прошли постпроцессинг')

    stats["requests"][0] += 1

    await message.reply(reply)


@dp.message(Command("predict_items"))
async def predict_items(message: types.Message,
                        preprocessing: Callable,
                        model: Pipeline,
                        postprocessing: dict,
                        stats: dict):

    document = message.document
    buffer = await bot.download(document)
    df = pd.read_csv(buffer)
    logging.info('От пользователя получен csv-файл')

    df_preprocessed = df['text'].apply(preprocessing)
    logging.info('Данные предобработаны')

    proba = model.predict_proba(df_preprocessed)
    logging.info('Модель вернула предсказания')

    labels = np.apply_along_axis(confident_predict, 1, proba)
    predictions_postprocessed = pd.Series(labels, name='predictions').apply(lambda x: postprocessing[x])
    logging.info('Предсказания прошли постпроцессинг')

    buffer = io.BytesIO()
    buffer.write(predictions_postprocessed.to_csv(index=False).encode())
    buffer.seek(0)

    stats["requests"][0] += 1

    await message.reply_document(
        BufferedInputFile(
            buffer.read(),
            filename="predictions.csv"
        )
    )


async def on_startup(bot):
    await bot.set_webhook(
        url=WEBHOOK_URL
    )


def main():
    dp.startup.register(on_startup)

    app = web.Application()
    webhook_requests_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot
    )

    webhook_requests_handler.register(app, path=WEBHOOK_PATH)

    setup_application(app, dp, bot=bot)

    web.run_app(app, host=WEB_SERVER_HOST, port=WEB_SERVER_PORT)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()