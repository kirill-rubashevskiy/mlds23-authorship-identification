import io

import numpy as np
import pandas as pd
from aiogram import Bot, F, Router, types
from aiogram.filters.command import Command
from aiogram.types import BufferedInputFile

from bot.keyboards import get_rate_kb


router = Router()


@router.message(Command("start"))
async def cmd_start(message: types.Message, users: set):
    users.add(message.from_user.id)
    text = (
        "Привет! Это бот для определения авторства текстов.\nЯ могу определить, кто из русских классиков "
        "написал один или несколько фрагментов.\nДля получения подсказок по моей работе можно отправить /help"
    )

    await message.reply(text)


@router.message(Command("help"))
async def cmd_help(message: types.Message):
    text = (
        "Бот поддерживает следующие режимы/команды:\n- /help — получить подсказки по работе бота\n- "
        "/predict_item — отправить фрагмент текста и получить предсказание об авторе фрагмента\n- "
        '/predict_items — отправить csv-файл, в котором есть столбец "text" c фрагментами текстов, и получить '
        "csv-файл с предсказаниями по каждому фрагменту\n- /rate — оценить работу бота\n- /start — запустить "
        "бота\n- /stats — получить статистику работы бота"
    )
    await message.reply(text)


@router.message(Command("rate"))
async def cmd_rate(message: types.Message):
    await message.answer(
        "Оцените работу бота:",
        reply_markup=get_rate_kb(),
    )


@router.message(F.text.in_({str(i) for i in range(1, 6)}))
async def rate_callback(message: types.Message, ratings: list):
    # update ratings sum and count
    ratings[0] += int(message.text)
    ratings[1] += 1

    await message.reply("Спасибо, что оценили работу бота!")


@router.message(Command("stats"))
async def cmd_stats(message: types.Message, users: set, requests: list, ratings: list):
    # if bot has not received ratings
    if ratings[1] == 0:
        text = f"За время работы бота {len(users)} юзер(а/ов) использовали его {requests[0]} раз(а)"
    # if bot has received ratings
    else:
        mean_rating = np.round(ratings[0] / ratings[1], 1)
        text = (
            f"За время работы бота {len(users)} юзер(а/ов) использовали его {requests[0]} раз(а)\nСредняя оценка "
            f"работы бота — {mean_rating}"
        )

    # send bot stats to user
    await message.reply(text)


@router.message(Command("predict_item"))
async def cmd_predict_item(message: types.Message, estimator, requests: list):
    # update requests number
    requests[0] += 1

    user_input = message.text.removeprefix("/predict_item")

    # get prediction
    text = estimator.predict_item(user_input)

    # send prediction to user
    await message.reply(text)


@router.message(Command("predict_items"))
async def cmd_predict_items(message: types.Message, estimator, requests: list, bot: Bot):
    # update requests number
    requests[0] += 1

    # download user document
    document = message.document
    buffer = await bot.download(document)
    user_input = pd.read_csv(buffer)

    # get predictions
    predictions = estimator.predict_items(user_input)

    # convert predictions to .csv file
    buffer = io.BytesIO()
    buffer.write(predictions.to_csv(index=False).encode())
    buffer.seek(0)

    # send predictions.csv to user
    reply_document = BufferedInputFile(buffer.read(), filename="predictions.csv")
    await message.reply_document(reply_document)
