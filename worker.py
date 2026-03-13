from arq import create_pool
from arq.connections import RedisSettings

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


async def placeholder_task(ctx):
    """
    Placeholder background task.
    Add actual background tasks here as needed.
    """
    pass


class WorkerSettings:
    """
    arq worker settings.
    """

    functions = [placeholder_task]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings(host=REDIS_HOST, port=REDIS_PORT)
