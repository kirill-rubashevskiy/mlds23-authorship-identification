from io import StringIO
from unittest.mock import create_autospec, patch

import pytest
import requests
import responses
from aiogram import Bot, Dispatcher, F
from aiogram.filters.command import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import CallbackQueryHandler, MessageHandler
from aiogram_tests.types.dataset import (
    CALLBACK_QUERY,
    MESSAGE,
    MESSAGE_WITH_DOCUMENT,
    USER,
)

from bot.routers.items import enter_text, predict_text, predict_texts, upload_file
from bot.routers.users import cmd_start, get_help, get_stats, rate, rated
from bot.states import PredictText, PredictTexts


mock_app_url = "http://app_url/"


class TestUsers:

    @pytest.fixture
    def mock_stats(self):
        return {"total_users": 3, "total_requests": 19, "avg_rating": 5.0}

    @pytest.mark.parametrize("status", [200, 400])
    @pytest.mark.asyncio
    @responses.activate
    async def test_cmd_start_success(self, status):
        responses.get(url=f"{mock_app_url}users/", status=200)
        responses.post(url=f"{mock_app_url}users/", status=status)
        requester = MockedBot(
            MessageHandler(
                cmd_start, Command("start"), dp=Dispatcher(app_url=mock_app_url)
            )
        )
        calls = await requester.query(MESSAGE.as_object(text="/start"))
        reply_message = calls.send_message.fetchone()
        assert reply_message.text == (
            "Привет! Я — бот для определения авторства текстов.\n"
            "Я могу определить, кто из русских классиков написал тот или иной текст"
        )
        assert "inline_keyboard" in reply_message.reply_markup

    @pytest.mark.asyncio
    @responses.activate
    async def test_cmd_start_app_down(self):
        responses.get(url=f"{mock_app_url}users/", body=requests.ConnectionError())
        requester = MockedBot(
            MessageHandler(
                cmd_start, Command("start"), dp=Dispatcher(app_url=mock_app_url)
            )
        )
        calls = await requester.query(MESSAGE.as_object(text="/start"))
        reply_message = calls.send_message.fetchone()
        assert (
            reply_message.text
            == "К сожалению, сервис сейчас недоступен, попробуйте попозже"
        )

    @pytest.mark.asyncio
    @responses.activate
    async def test_get_stats(self, mock_stats):
        responses.get(url=f"{mock_app_url}users/stats", status=200, json=mock_stats)
        request_handler = CallbackQueryHandler(
            get_stats, F.data == "stats", dp=Dispatcher(app_url=mock_app_url)
        )
        requester = MockedBot(request_handler)
        callback_query = CALLBACK_QUERY.as_object(
            data="stats", message=MESSAGE.as_object()
        )

        calls = await requester.query(callback_query)
        replies = calls.send_message.fetchall()
        assert replies[0].text == (
            f"За время работы бота {mock_stats['total_users']} пользователя(ей) воспользовались им "
            f"{mock_stats['total_requests']} раз\n"
            f"Средняя оценка бота пользователями: {mock_stats['avg_rating']}"
        )
        assert replies[1].text == "Я могу еще чем-то помочь?"

    @pytest.mark.asyncio
    async def test_get_help(self):
        request_handler = CallbackQueryHandler(get_help, F.data == "help")
        requester = MockedBot(request_handler)
        callback_query = CALLBACK_QUERY.as_object(
            data="help", message=MESSAGE.as_object()
        )

        calls = await requester.query(callback_query)
        replies = calls.send_message.fetchall()
        assert replies[0].text == (
            "Для определения автора текста достаточно отправить сам текст\n"
            "Для определения авторов нескольких текстов необходимо отправить csv-файл с одной колонкой-текстами"
        )
        assert replies[1].text == "Я могу еще чем-то помочь?"

    @pytest.mark.asyncio
    async def test_rate(self):
        request_handler = CallbackQueryHandler(
            rate,
            F.data == "rate",
        )
        requester = MockedBot(request_handler)
        callback_query = CALLBACK_QUERY.as_object(
            data="rate", message=MESSAGE.as_object()
        )

        calls = await requester.query(callback_query)
        reply = calls.send_message.fetchone()
        assert reply.text == "Оцени работу бота"
        assert "inline_keyboard" in reply.reply_markup

    @pytest.mark.asyncio
    @responses.activate
    async def test_rated(self):
        responses.patch(
            url=f"{mock_app_url}users/{USER.get('id')}/5",
            status=200,
        )
        request_handler = CallbackQueryHandler(
            rated, F.data.startswith("rated_"), dp=Dispatcher(app_url=mock_app_url)
        )
        requester = MockedBot(request_handler)
        callback_query = CALLBACK_QUERY.as_object(
            data="rated_5", message=MESSAGE.as_object()
        )

        calls = await requester.query(callback_query)
        replies = calls.send_message.fetchall()
        assert replies[0].text == "Спасибо, что оценил работу бота"
        assert replies[1].text == "Я могу еще чем-то помочь?"


class TestItems:

    mock_download = create_autospec(
        Bot.download, return_value=StringIO("text\ntext 1\ntext 2")
    )

    @pytest.fixture
    def mock_predictions(self):
        return "text,author\ntext 1,А. Пушкин\ntext 2,И. Тургенев"

    @pytest.mark.asyncio
    async def test_enter_text(self):
        request_handler = CallbackQueryHandler(
            enter_text,
            F.data == "predict_text",
        )
        requester = MockedBot(request_handler)
        callback_query = CALLBACK_QUERY.as_object(
            data="predict_text", message=MESSAGE.as_object()
        )

        calls = await requester.query(callback_query)
        reply = calls.send_message.fetchone()
        assert reply.text == "Введи текст, автора которого нужно определить"

    @pytest.mark.parametrize(
        "label,name,expected",
        [
            (-1, "не могу определить автора", "Я не могу определить автора этого текста"),
            (0, "А. Пушкин", "Возможно, этот текст написал А. Пушкин"),
        ],
    )
    @pytest.mark.asyncio
    @responses.activate
    async def test_predict_text(self, label, name, expected):
        responses.patch(
            url=f"{mock_app_url}users/{USER.get('id')}/requests",
            status=200,
        )
        responses.post(
            url=f"{mock_app_url}items/predict_text",
            status=200,
            json={"label": label, "name": name},
        )
        request_handler = MessageHandler(
            predict_text,
            state=PredictText.entering_text,
            dp=Dispatcher(app_url=mock_app_url),
        )

        requester = MockedBot(request_handler)
        calls = await requester.query(MESSAGE.as_object())
        replies = calls.send_message.fetchall()
        assert replies[0].text == expected
        assert replies[1].text == "Я могу еще чем-то помочь?"

    @pytest.mark.asyncio
    async def test_upload_file(self):
        request_handler = CallbackQueryHandler(
            upload_file,
            F.data == "predict_texts",
        )
        requester = MockedBot(request_handler)
        callback_query = CALLBACK_QUERY.as_object(
            data="predict_texts", message=MESSAGE.as_object()
        )

        calls = await requester.query(callback_query)
        reply = calls.send_message.fetchone()
        assert reply.text == "Загрузи csv-файл c текстами"

    @patch.object(Bot, "download", mock_download, create=True)
    @pytest.mark.asyncio
    @responses.activate
    async def test_predict_texts(self, mock_predictions):
        responses.patch(
            url=f"{mock_app_url}users/{USER.get('id')}/requests",
            status=200,
        )
        responses.post(
            url=f"{mock_app_url}items/predict_texts",
            status=200,
            body=mock_predictions,
            content_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=predictions.csv"},
        )
        request_handler = MessageHandler(
            predict_texts,
            F.document,
            state=PredictTexts.uploading_file,
            dp=Dispatcher(app_url=mock_app_url),
        )
        requester = MockedBot(request_handler)
        calls = await requester.query(MESSAGE_WITH_DOCUMENT.as_object())
        reply = calls.send_document.fetchone()
        assert reply.document.filename == "predictions.csv"
