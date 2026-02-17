import asyncio

from arq import create_pool
from arq.connections import RedisSettings

from ads.matcher import find_matching_ads_background

REDIS_HOST = "redis"
REDIS_PORT = 6379


async def startup(ctx):
    """
    Binds the redis connection pool to the context.
    """
    ctx["redis"] = await create_pool(RedisSettings(host=REDIS_HOST, port=REDIS_PORT))


async def shutdown(ctx):
    """
    Closes the redis connection pool.
    """
    await ctx["redis"].close()


class WorkerSettings:
    """
    arq worker settings.
    """

    functions = [find_matching_ads_background]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
