import logging
from contextlib import asynccontextmanager
from functools import partial

import hydra
import uvicorn
from fastapi import FastAPI
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from omegaconf import DictConfig
from redis import asyncio as aioredis

from app import models
from app.database import engine
from app.dependencies import Model
from app.routers import items, users


@asynccontextmanager
async def lifespan(app: FastAPI, cfg: DictConfig):
    # load DVC-tracked model from s3
    app.state.model = Model(model_name=cfg.app.model_name)
    logging.info("Model loaded")

    # initialize Redis cache
    redis = aioredis.from_url(f"redis://{cfg.redis.host}", decode_responses=True)
    FastAPICache.init(RedisBackend(redis), prefix="fastapi-cache")
    logging.info("Redis initialized")
    yield


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    models.Base.metadata.create_all(bind=engine)
    app = FastAPI(lifespan=partial(lifespan, cfg=cfg))
    app.include_router(items.router)
    app.include_router(users.router)

    @app.get("/")
    def root():
        return {"message": "Welcome to the Authorship Identification Service"}

    uvicorn.run(app, host=cfg.app.host, port=cfg.app.port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()