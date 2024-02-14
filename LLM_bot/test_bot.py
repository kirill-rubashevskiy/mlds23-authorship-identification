import pytest

from aiogram_tests import MockedBot
from aiogram_tests.handler import MessageHandler
from aiogram_tests.types.dataset import MESSAGE

from aiogram.filters.command import Command
#импортиурем ручки для тестирования
from bot import cmd_start

@pytest.mark.asyncio
async def test_cmd_start():
    requester = MockedBot(MessageHandler(cmd_start, Command("start")))
    calls = await requester.query(MESSAGE.as_object(text="/start"))
    assert 1 == 1