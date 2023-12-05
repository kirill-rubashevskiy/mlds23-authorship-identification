# импорт библиотек
from fastapi import FastAPI
import time
import logging
import os
from aiogram import Bot, Dispatcher, types
import numpy as np

from models import load_model
from utils import label2name, preprocess_text5

TOKEN = os.getenv('TOKEN')
WEBHOOK_PATH = f"/bot/{TOKEN}"
RENDER_WEB_SERVICE_NAME = "mlds23-authorship-identification"
WEBHOOK_URL = "https://" + RENDER_WEB_SERVICE_NAME + ".onrender.com" + WEBHOOK_PATH

logging.basicConfig(filemode='a', level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot=bot)

app = FastAPI()

items = {}


@app.on_event("startup")
async def on_startup():
    webhook_info = await bot.get_webhook_info()
    if webhook_info.url != WEBHOOK_URL:
        await bot.set_webhook(
            url=WEBHOOK_URL
        )

    items['model'] = load_model()
    logging.info('Модель загружена')


@dp.message_handler(commands=['start'])
async def start_handler(message: types.Message):
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Start: {user_id} {user_full_name} {time.asctime()}. Message: {message}')
    await message.reply("Привет! Это бот для определения авторства текстов. Пришли фрагмент текста, а я попробую "
                        "определить, кто из классиков его написал.")


@dp.message_handler()
async def main_handler(message: types.Message):
    user_request = message.text
    user_id = message.from_user.id
    user_full_name = message.from_user.full_name
    logging.info(f'Main: {user_id} {user_full_name} {time.asctime()}. Message: {message}')

    user_request_preprocessed = preprocess_text5(user_request)
    logging.info('Данные предобработаны')

    pred = items['model'].predict(np.array([user_request_preprocessed]))[0]
    logging.info('Модель что-то предсказала')

    reply = f'Кажется, этот фрагмент написал {label2name[pred]}'

    await message.reply(reply)


@app.post(WEBHOOK_PATH)
async def bot_webhook(update: dict):
    telegram_update = types.Update(**update)
    Dispatcher.set_current(dp)
    Bot.set_current(bot)
    await dp.process_update(telegram_update)


@app.on_event("shutdown")
async def on_shutdown():
    items.clear()

    session = await bot.get_session()
    await session.close()


@app.get("/")
def main_web_handler():
    return "Everything ok!"
