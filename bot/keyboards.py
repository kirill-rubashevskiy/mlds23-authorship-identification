from aiogram.types import InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder


start_kb_texts = [
    "определить автора текста",
    "определить авторов нескольких текстов",
    "получить подсказку по работе бота",
    "оценить работу бота",
    "получить статистику по работе бота",
]

start_kb_callback_data = ["predict_text", "predict_texts", "help", "rate", "stats"]


def get_start_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for text, callback_data in zip(start_kb_texts, start_kb_callback_data, strict=True):
        builder.button(text=text, callback_data=callback_data)
    builder.adjust(1)
    return builder.as_markup(resize_keyboard=True)


def get_rate_kb() -> InlineKeyboardMarkup:
    builder = InlineKeyboardBuilder()
    for i in range(1, 6):
        builder.button(text=f"{i}", callback_data=f"rated_{i}")
    builder.adjust(1)
    return builder.as_markup(resize_keyboard=True)
