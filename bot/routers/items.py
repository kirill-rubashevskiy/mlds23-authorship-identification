import logging
from io import BytesIO

import requests
from aiogram import Bot, F, Router
from aiogram.fsm.context import FSMContext
from aiogram.types import BufferedInputFile, CallbackQuery, Message

from bot.states import PredictText, PredictTexts
from bot.utils import loop


router = Router()


@router.callback_query(F.data == "predict_text")
async def entering_text(callback: CallbackQuery, state: FSMContext):
    logging.info("Пользователь начал диалог по определению автора текста")
    await callback.message.reply(text="Введи текст, автора которого нужно определить")
    await state.set_state(PredictText.entering_text)


@router.message(PredictText.entering_text)
@loop
async def predict_text(message: Message, state: FSMContext, app_url: str, **kwargs):
    text = message.text
    user_id = message.from_user.id
    if text:
        logging.info("Пользователь ввел текст")
        requests.patch(f"{app_url}users/{user_id}/requests")
        response = requests.post(f"{app_url}items/predict_text", json={"text": text})
        if response.status_code == 200:
            data = response.json()
            if data["label"] != -1:
                await message.reply(f"Возможно, этот текст написал {data['name']}")
            else:
                await message.reply("Я не могу определить автора этого текста")


@router.callback_query(F.data == "predict_texts")
async def uploading_file(callback: CallbackQuery, state: FSMContext):
    logging.info("Пользователь начал диалог по определению авторов текстов")
    await callback.message.reply(text="Загрузи csv-файл c текстами")
    await state.set_state(PredictTexts.uploading_file)


@router.message(PredictTexts.uploading_file, F.document)
@loop
async def predict_texts(
    message: Message, state: FSMContext, app_url: str, bot: Bot, **kwargs
):
    document_name = message.document.file_name
    document = message.document
    user_id = message.from_user.id
    logging.info("Пользователь загрузил файн")
    requests.patch(f"{app_url}users/{user_id}/requests")
    buffer = await bot.download(document)
    files = {"file": (document_name, buffer, "text/csv")}
    response = requests.post(f"{app_url}items/predict_texts", files=files)
    buffer = BytesIO(response.content)
    document = BufferedInputFile(buffer.read(), filename="predictions.csv")
    await message.reply_document(document=document)
