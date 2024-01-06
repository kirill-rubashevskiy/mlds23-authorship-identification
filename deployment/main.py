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
from aiogram.utils.keyboard import InlineKeyboardBuilder

from models import load_model
from utils import label2name, preprocess_text5

TOKEN = os.getenv('TOKEN')
WEBHOOK_HOST = "https://applied-python-aiogram-bot.onrender.com"
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
    - /rate — оценить работу бота
    - /start — запустить бота
    - /stats — получить статистику работы бота
    - отправить фрагмент текста — получить предсказание об авторе фрагмента
    - отправить csv-файл, в котором есть столбец "text" c фрагментами текстов — получить csv-файл с предсказаниями по каждому фрагменту
    """

    await message.reply(content)


@dp.message(Command("rate"))
async def cmd_rate(message: types.Message):
    builder = InlineKeyboardBuilder()
    for i in range(1, 6):
        builder.add(
            types.InlineKeyboardButton(
                text=str(i),
                callback_data=f'num_{str(i)}')
        )

    await message.answer(
        "Оцените работу бота:",
        reply_markup=builder.as_markup(),
    )


@dp.callback_query(F.data.startswith('num_'))
async def callbacks_num(callback: types.CallbackQuery, stats: dict):
    action = int(callback.data.split("_")[1])
    stats['rate_sum'][0] += action
    stats['rate_num'][0] += 1

    await callback.answer(
        text="Спасибо, что оценили работу бота!",
    )


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


@dp.message(F.text)
async def predict_item(message: types.Message,
                       preprocessing: Callable,
                       model: Pipeline,
                       postprocessing: dict,
                       stats: dict):
    user_request = message.text
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Message: {message}')

    user_request_preprocessed = preprocessing(user_request)
    logging.info('Данные предобработаны')

    pred = model.predict(np.array([user_request_preprocessed]))[0]
    logging.info('Модель сделала предсказание')

    stats["requests"][0] += 1

    await message.reply(
        f'Кажется, этот фрагмент написал {postprocessing[pred]}'
    )


@dp.message(F.document)
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

    predictions = model.predict(df_preprocessed)
    logging.info('Модель вернула предсказания')

    predictions_postprocessed = pd.Series(predictions, name='predictions').apply(lambda x: postprocessing[x])
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