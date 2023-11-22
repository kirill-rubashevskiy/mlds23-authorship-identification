import os
import sys

# библиотеки для бота
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import logging
import asyncio

# импортируем модель и препроцессинг
from models import dummy_clf
from preproc import dummy_preprocessing

# забираем токен бота из enviroment variables
BOT_TOKEN = os.getenv('BOT_TOKEN')

dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: types.Message):
    await message.answer("Привет! Это бот для определения авторства текстов. Пришли фрагмент текста, а я попробую "
                         "определить, кто из классиков его написал.")

@dp.message()
async def echo_handler(message: types.Message):

    user_input = message.text
    logging.info('Запрос получен')

    user_input_preprocessed = dummy_preprocessing(user_input)
    logging.info('Данные предобработаны')

    pred = dummy_clf.predict(user_input_preprocessed)[0]
    logging.info('Модель что-то предсказала')

    await message.answer(pred)
    logging.info('Бот вернул предсказание')

async def main() -> None:
    # Initialize Bot instance
    bot = Bot(BOT_TOKEN)
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())