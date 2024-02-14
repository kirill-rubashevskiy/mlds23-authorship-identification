from aiogram.types import KeyboardButton, ReplyKeyboardMarkup
from aiogram.utils.keyboard import ReplyKeyboardBuilder


def get_rate_kb() -> ReplyKeyboardMarkup:
    """
    Function generates keyboard to rate bot.

    :return: rate keyboard
    """
    kb = ReplyKeyboardBuilder()
    for i in range(1, 6):
        kb.add(KeyboardButton(text=str(i), callback_data=f"num_{str(i)}"))

    return kb.as_markup(resize_keyboard=True)
