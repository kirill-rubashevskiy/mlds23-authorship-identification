# импорт библиотек
import os
import sys
import numpy as np

# библиотеки для бота
from aiogram import Bot, Dispatcher, types
from aiogram.filters import CommandStart
import logging
import asyncio

# локальный импорт препроцессинга и модели
from data_preproc.src.utils import preprocess_text5
from models import baseline_model
from utils import label2name

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

    user_input_preprocessed = preprocess_text5(user_input)
    logging.info('Данные предобработаны')

    pred = baseline_model.predict(np.array([user_input_preprocessed]))[0]
    logging.info('Модель что-то предсказала')

    ans = f'Кажется, этот фрагмент написал {label2name[pred]}'

    await message.answer(ans)
    logging.info('Бот вернул предсказание')


async def main() -> None:
    # Initialize Bot instance
    bot = Bot(BOT_TOKEN)
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
