"""
mordomo-brain — entrypoint.
"""
import asyncio
import logging
import signal

import nats

from src.config import NATS_URL, SUBJECT_GENERATE
from src.handlers import handle_generate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mordomo-brain")

_shutdown = asyncio.Event()


def _handle_signal(sig):
    logger.info("Signal %s received — shutting down", sig.name)
    _shutdown.set()


async def run() -> None:
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, lambda s=sig: _handle_signal(s))

    nc = None
    while not _shutdown.is_set():
        try:
            nc = await nats.connect(
                NATS_URL,
                name="mordomo-brain",
                reconnect_time_wait=2,
                max_reconnect_attempts=-1,
            )
            logger.info("Connected to NATS at %s", NATS_URL)

            async def _wrapper(msg):
                msg._client = nc
                await handle_generate(msg)

            await nc.subscribe(SUBJECT_GENERATE, cb=_wrapper)
            logger.info("Brain ready — subscribed to %s", SUBJECT_GENERATE)

            await _shutdown.wait()

        except Exception as exc:
            logger.error("Connection error: %s — retrying in 5s", exc)
            await asyncio.sleep(5)
        finally:
            if nc and not nc.is_closed:
                await nc.drain()

    logger.info("Brain stopped")


if __name__ == "__main__":
    asyncio.run(run())
