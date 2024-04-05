import logging

import requests
from aiogram import F, Router
from aiogram.filters.command import Command
from aiogram.fsm.context import FSMContext
from aiogram.types import CallbackQuery, Message

from bot.keyboards import get_rate_kb, get_start_kb
from bot.utils import loop


router = Router()


@router.message(Command("start"))
async def cmd_start(message: Message, app_url: str):
    try:
        # check if FastAPI app is up
        requests.get(f"{app_url}users/")
        user_id = message.from_user.id
        # if user is new — register them in db
        response = requests.post(f"{app_url}users/", json={"id": user_id})
        if response.status_code == 200:
            logging.info("Новый пользователь начал диалог с ботом")
        elif response.status_code == 400:
            logging.info("Существующий пользователь начал диалог с ботом")

        await message.answer(
            text=(
                "Привет! Я — бот для определения авторства текстов.\n"
                "Я могу определить, кто из русских классиков написал тот или иной текст"
            ),
            reply_markup=get_start_kb(),
        )
    # if FastAPI app is down
    except requests.exceptions.ConnectionError:
        await message.answer(
            text="К сожалению, сервис сейчас недоступен, попробуйте попозже"
        )


@router.callback_query(F.data == "rate")
async def rate(callback: CallbackQuery, state: FSMContext):
    logging.info("Пользователь начал диалог по оценке работы бота")
    await callback.message.reply(text="Оцени работу бота", reply_markup=get_rate_kb())


@router.callback_query(F.data.startswith("rated_"))
@loop
async def rated(callback: CallbackQuery, state: FSMContext, app_url: str, **kwargs):
    rating = int(callback.data.removeprefix("rated_"))
    user_id = callback.from_user.id
    requests.patch(f"{app_url}users/{user_id}/{rating}")
    await callback.message.reply(text="Спасибо, что оценил работу бота")


@router.callback_query(F.data == "stats")
@loop
async def get_stats(callback: CallbackQuery, state: FSMContext, app_url: str, **kwargs):
    logging.info("Пользователь запросил статистику бота")
    response = requests.get(f"{app_url}users/stats")
    if response.status_code == 200:
        data = response.json()
        await callback.message.reply(
            text=(
                f"За время работы бота {data['total_users']} пользователя(ей) воспользовались им "
                f"{data['total_requests']} раз\n"
                f"Средняя оценка бота пользователями: {data['avg_rating']}"
            )
        )


@router.callback_query(F.data == "help")
@loop
async def get_help(callback: CallbackQuery, state: FSMContext, **kwargs):
    logging.info("Пользователь запросил подсказку по работе бота")
    await callback.message.reply(
        text=(
            "Для определения автора текста достаточно отправить сам текст\n"
            "Для определения авторов нескольких текстов необходимо отправить csv-файл с одной колонкой-текстами"
        )
    )
