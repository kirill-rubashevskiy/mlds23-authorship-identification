from io import StringIO
from unittest.mock import MagicMock, create_autospec, patch

import pandas as pd
import pytest
from aiogram import Bot, Dispatcher
from aiogram.filters.command import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE, MESSAGE_WITH_DOCUMENT

from bot.handlers import (
    cmd_help,
    cmd_predict_item,
    cmd_predict_items,
    cmd_rate,
    cmd_start,
    cmd_stats,
    rate_callback,
)


# create mock estimator instead of loading ML fitted model from s3
mock_estimator = MagicMock(
    predict_item=MagicMock(return_value="Кажется, этот фрагмент написал А. Пушкин"),
    predict_items=MagicMock(
        return_value=pd.Series(data=["А. Куприн"] * 4, name="predictions.csv")
    ),
)


# create mock Bot.download method that returns mock test .csv file to emulate bot downloading test file sent by user
# from Telegram servers
mock_download = create_autospec(
    Bot.download,
    side_effect=[StringIO("text\ntext 1\ntext 2\ntext 3\ntext 4") for _ in range(2)],
)


@pytest.mark.asyncio
async def test_cmd_start_handler():
    requester = MockedBot(
        MessageHandler(cmd_start, Command("start"), dp=Dispatcher(users=set()))
    )
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    reply_message = calls.send_message.fetchone().text
    assert dict(MESSAGE)["from"]["id"] in requester._handler.dp["users"]
    assert reply_message == (
        "Привет! Это бот для определения авторства текстов.\nЯ могу определить, кто из русских "
        "классиков написал один или несколько фрагментов.\nДля получения подсказок по моей "
        "работе можно отправить /help"
    )


@pytest.mark.asyncio
async def test_cmd_help_handler():
    requester = MockedBot(MessageHandler(cmd_help, Command("help")))
    calls = await requester.query(MESSAGE.as_object(text="/help"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == (
        "Бот поддерживает следующие режимы/команды:\n- /help — получить подсказки по работе "
        "бота\n- /predict_item — отправить фрагмент текста и получить предсказание об авторе "
        'фрагмента\n- /predict_items — отправить csv-файл, в котором есть столбец "text" c '
        "фрагментами текстов, и получить csv-файл с предсказаниями по каждому фрагменту\n- /rate "
        "— оценить работу бота\n- /start — запустить бота\n- /stats — получить статистику работы "
        "бота"
    )


@pytest.mark.asyncio
async def test_cmd_rate_handler():
    request_handler = MessageHandler(cmd_rate, Command("rate"))
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="/rate"))
    answer_message = calls.send_message.fetchone()
    assert answer_message.text == "Оцените работу бота:"
    assert "keyboard" in answer_message.reply_markup


@pytest.mark.asyncio
async def test_rate_callback_handler():
    request_handler = MessageHandler(rate_callback, dp=Dispatcher(ratings=[0, 0]))
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="5"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == "Спасибо, что оценили работу бота!"
    assert requester._handler.dp["ratings"] == [5, 1]


@pytest.mark.asyncio
async def test_cmd_predict_item_handler():
    request_handler = MessageHandler(
        cmd_predict_item,
        Command("predict_item"),
        dp=Dispatcher(estimator=mock_estimator, requests=[0]),
    )
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="/predict_item"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == "Кажется, этот фрагмент написал А. Пушкин"
    assert requester._handler.dp["requests"][0] == 1


@patch.object(Bot, "download", mock_download, create=True)
@pytest.mark.asyncio
async def test_cmd_predict_items_handler():
    request_handler = MessageHandler(
        cmd_predict_items,
        Command("predict_items"),
        dp=Dispatcher(estimator=mock_estimator, requests=[0]),
    )
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE_WITH_DOCUMENT.as_object(text="/predict_items"))
    reply_document = calls.send_document.fetchone()
    assert reply_document.document.filename == "predictions.csv"
    assert requester._handler.dp["requests"][0] == 1


@pytest.mark.parametrize(
    "users,requests,ratings,expected",
    [
        (
            set(),
            [0],
            [0, 0],
            "За время работы бота 0 юзер(а/ов) использовали его 0 раз(а)",
        ),
        (
            {12345, 67890},
            [4],
            [9, 2],
            "За время работы бота 2 юзер(а/ов) использовали его 4 раз(а)\nСредняя оценка "
            "работы бота — 4.5",
        ),
    ],
)
@pytest.mark.asyncio
async def test_cmd_stats_handler(users, requests, ratings, expected):
    request_handler = MessageHandler(
        cmd_stats,
        Command("stats"),
        dp=Dispatcher(users=users, requests=requests, ratings=ratings),
    )
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="/stats"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == expected
