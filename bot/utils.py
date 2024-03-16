import asyncio
from typing import Callable

from aiogram.types import Message

from bot.keyboards import get_start_kb


loop_answer = dict(text="Я могу еще чем-то помочь?", reply_markup=get_start_kb())


def loop(func: Callable) -> Callable:
    """
    Decorator clears user state and loops dialog with starting menu.

    :param func: function to decorate
    :return: decoarted function
    """

    async def wrapper(*args, **kwargs):
        await func(*args, **kwargs)
        await asyncio.sleep(1)
        message = locals().get("args")[0]
        state = locals().get("kwargs")["state"]
        await state.clear()
        if isinstance(message, Message):
            await message.answer(**loop_answer)
        else:
            await message.message.answer(**loop_answer)

    return wrapper
