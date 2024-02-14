import logging
import os

import hydra
from aiogram import Bot, Dispatcher
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from dotenv import load_dotenv
from hydra.utils import instantiate
from omegaconf import DictConfig

from bot.handlers import router


load_dotenv()
TOKEN = os.getenv("TOKEN")


async def on_startup(bot, url, token):
    await bot.set_webhook(url=url, secret_token=token)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize Dispatcher instance
    dp = Dispatcher(
        estimator=instantiate(cfg.bot.estimator),
        users=set(),
        requests=[0],
        ratings=[0, 0],
    )

    # include handlers router
    dp.include_router(router)

    # register startup hook to initialize webhook
    dp.startup.register(on_startup, f"{cfg.webhook.base_url}{cfg.webhook.path}", TOKEN)

    # initialize Bot instance
    bot = Bot(token=TOKEN)

    # create web.Application instance
    app = web.Application()

    # create SimpleRequestHandler instance
    webhook_requests_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)

    # register webhook handler on application
    webhook_requests_handler.register(app, path=cfg.webhook.path)

    # mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # start webserver
    web.run_app(app, host=cfg.bot.web_server.host, port=cfg.bot.web_server.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
