import logging
from functools import partial

import hydra
from aiogram import Bot, Dispatcher
from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
from aiohttp import web
from omegaconf import DictConfig

from bot.routers import items, users


async def on_startup(bot, url):
    await bot.set_webhook(url=url)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize Dispatcher instance
    dp = Dispatcher(app_url=f"http://{cfg.app.host.name}:{cfg.app.port}/")

    # include routers
    dp.include_routers(items.router, users.router)

    # register startup hook to initialize webhook
    dp.startup.register(
        partial(
            on_startup,
            url=f"https://{cfg.bot.base_webhook_url}/webhook/{cfg.bot.token}",
        )
    )

    # initialize Bot instance
    bot = Bot(token=cfg.bot.token)

    # create web.Application instance
    app = web.Application()

    # create SimpleRequestHandler instance
    webhook_requests_handler = SimpleRequestHandler(dispatcher=dp, bot=bot)

    # register webhook handler on application
    webhook_requests_handler.register(app, path=f"/webhook/{cfg.bot.token}")

    # mount dispatcher startup and shutdown hooks to aiohttp application
    setup_application(app, dp, bot=bot)

    # start webserver
    web.run_app(app, host=cfg.bot.host, port=cfg.bot.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
