from unittest.mock import MagicMock

import pytest
from aiogram import Dispatcher
from aiogram.filters.command import Command
from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE

# импортируем ручки для тестирования
from bot import cmd_help, cmd_predict, cmd_rate, cmd_start, cmd_stats, process_rating


# воспользуемся MagicMock из unittest вместо загрузки из s3
mock_estimator = MagicMock(
    predict=MagicMock(
        return_value="Эти строки были написаны Алексеем Николаевичем Толстым - русским классическим писателем"
    )
)


@pytest.mark.asyncio
async def test_cmd_predict():
    request_handler = MessageHandler(
        cmd_predict,
        Command("predict"),
        dp=Dispatcher(
            llm_wrapper=mock_estimator,
            usage_stats={"total_requests": 0, "average_rating": 0},
        ),
    )
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="/predict"))
    reply_message = calls.send_message.fetchone().text
    assert (
        reply_message
        == "Эти строки были написаны Алексеем Николаевичем Толстым - русским классическим писателем"
    )
    assert requester._handler.dp["usage_stats"]["total_requests"] == 1


@pytest.mark.asyncio
async def test_cmd_start_text():
    requester = MockedBot(MessageHandler(cmd_start, Command("start")))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == (
        "Привет! Я бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. Введите свой вопрос после команды /predict."
    )


@pytest.mark.asyncio
async def test_cmd_help_text():
    requester = MockedBot(MessageHandler(cmd_help, Command("help")))
    calls = await requester.query(MESSAGE.as_object(text="/help"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == (
        "Это бот для генерации ответов на вопросы об авторстве отрывков из русской литературы. У него есть следующие команды:\n"
        "- /start - начать работу;\n "
        "- /predict - отправить текст отрывка для определения авторства текста;\n "
        "- /predict_from_file - отправить файл в формате txt c отрывком для определения авторства текста;\n "
        "- /rate - оценить работу бота;\n "
        "- /stats — получить статистику работы бота\n "
        "- /help — получить список команд бота\n "
        "TL;DR: просто отправьте текстовый фрагмент после команды /predict, и я постараюсь ответить."
    )


@pytest.mark.asyncio
async def test_cmd_rate_text_button():
    request_handler = MessageHandler(cmd_rate, Command("rate"))
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="/rate"))
    answer_message = calls.send_message.fetchone()
    assert answer_message.text == "Пожалуйста, оцените качество ответа:"
    assert "keyboard" in answer_message.reply_markup


@pytest.mark.asyncio
async def test_process_rating():
    request_handler = MessageHandler(
        process_rating,
        dp=Dispatcher(ratings=[], usage_stats={"total_requests": 0, "average_rating": 0}),
    )
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="5"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == "Спасибо за вашу оценку! Средний рейтинг: 5.0"
    assert requester._handler.dp["ratings"] == [5]


@pytest.mark.parametrize(
    "usage_stats,example",
    [
        (
            {"total_requests": 3, "average_rating": 5},
            "Статистика использования сервиса:\n"
            "Общее количество запросов: 3\n"
            "Средний рейтинг: 5",
        ),
    ],
)
@pytest.mark.asyncio
async def test_cmd_stats(usage_stats, example):
    request_handler = MessageHandler(
        cmd_stats,
        Command("stats"),
        dp=Dispatcher(usage_stats=usage_stats),
    )
    requester = MockedBot(request_handler)
    calls = await requester.query(MESSAGE.as_object(text="/stats"))
    reply_message = calls.send_message.fetchone().text
    assert reply_message == example
