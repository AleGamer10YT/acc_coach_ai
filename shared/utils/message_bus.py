from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Optional

from redis import asyncio as aioredis

from .config import get_settings
from .logging import configure_logging

logger = configure_logging("message_bus")


_INMEMORY_QUEUES: dict[str, "asyncio.Queue[str]"] = {}


class TelemetryPublisher:
    def __init__(self, channel: str = "telemetry.raw") -> None:
        self.settings = get_settings()
        self.channel = channel
        self._redis: Optional[aioredis.Redis] = None
        self._queue: "asyncio.Queue[str]" = _INMEMORY_QUEUES.setdefault(channel, asyncio.Queue())

    async def connect(self) -> None:
        if self.settings.redis_url:
            self._redis = await aioredis.from_url(self.settings.redis_url)
            logger.info("Connected to Redis at %s", self.settings.redis_url)
        else:
            logger.info("Using in-memory queue for telemetry publisher")

    async def publish(self, payload: dict) -> None:
        data = json.dumps(payload)
        if self._redis:
            await self._redis.publish(self.channel, data)
        else:
            await self._queue.put(data)

    @asynccontextmanager
    async def subscribe(self) -> AsyncIterator["TelemetrySubscriber"]:
        subscriber = TelemetrySubscriber(self.channel, self._queue, self._redis)
        await subscriber.connect()
        try:
            yield subscriber
        finally:
            await subscriber.disconnect()


class TelemetrySubscriber:
    def __init__(
        self,
        channel: str,
        queue: "asyncio.Queue[str]",
        redis: Optional[aioredis.Redis],
    ) -> None:
        self.channel = channel
        self._queue = queue
        self._redis = redis
        self._pubsub: Optional[aioredis.client.PubSub] = None

    async def connect(self) -> None:
        if self._redis:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(self.channel)
            logger.info("Subscribed to Redis channel %s", self.channel)
        else:
            logger.info("Subscribed to in-memory queue")

    async def disconnect(self) -> None:
        if self._pubsub:
            await self._pubsub.unsubscribe(self.channel)
            await self._pubsub.close()

    async def listen(self, callback: Callable[[dict], None]) -> None:
        if self._pubsub:
            async for message in self._pubsub.listen():
                if message["type"] != "message":
                    continue
                data = json.loads(message["data"])
                result = callback(data)
                if asyncio.iscoroutine(result):
                    await result
        else:
            while True:
                data = await self._queue.get()
                result = callback(json.loads(data))
                if asyncio.iscoroutine(result):
                    await result
