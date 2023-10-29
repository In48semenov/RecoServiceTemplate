import asyncio
from concurrent.futures.thread import ThreadPoolExecutor
from typing import Any, Dict

import uvloop
from fastapi import FastAPI

from ..log import app_logger, setup_logging
from ..settings import ServiceConfig
from .exception_handlers import add_exception_handlers
from .middlewares import add_middlewares
from .views import add_views

import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

__all__ = ("create_app",)


def setup_asyncio(thread_name_prefix: str) -> None:
    uvloop.install()

    loop = asyncio.get_event_loop()

    executor = ThreadPoolExecutor(thread_name_prefix=thread_name_prefix)
    loop.set_default_executor(executor)

    def handler(_, context: Dict[str, Any]) -> None:
        message = "Caught asyncio exception: {message}".format_map(context)
        app_logger.warning(message)

    loop.set_exception_handler(handler)


def create_app(config: ServiceConfig) -> FastAPI:
    setup_logging(config)
    setup_asyncio(thread_name_prefix=config.service_name)

    sentry_sdk.init(
        dsn="https://91d373387f481fd146be7df7c7f9b0a7@o4506128446259200.ingest.sentry.io/4506128464281600",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
        integrations=[
        FastApiIntegration(
            transaction_style="endpoint"
        ),
    ],
        enable_tracing=True,
    )
    
    app = FastAPI(debug=False)
    app.state.k_recs = config.k_recs

    add_views(app)
    add_middlewares(app)
    add_exception_handlers(app)

    return app
