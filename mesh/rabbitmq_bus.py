from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable

from .messages import GossipMessage, MessageKind

MessageHandler = Callable[[GossipMessage], Awaitable[None]]

logger = logging.getLogger(__name__)


class RabbitMQMeshBus:
    """RabbitMQ adapter for the Phase 4 mesh topics.

    The real node mesh will use libp2p. This adapter keeps the neighborhood-server
    contracts stable while local development uses the RabbitMQ service already in Compose.
    """

    def __init__(
        self,
        amqp_url: str,
        exchange_name: str = "vayugrid.mesh",
        replay_window_seconds: int = 30,
    ) -> None:
        self.amqp_url = amqp_url
        self.exchange_name = exchange_name
        self.replay_window_seconds = replay_window_seconds
        self._seen: dict[str, float] = {}
        self._handlers: dict[MessageKind, list[MessageHandler]] = {}

    def subscribe(self, kind: MessageKind, handler: MessageHandler) -> None:
        """Register a handler for a specific message kind."""
        self._handlers.setdefault(kind, []).append(handler)

    def subscribe_all(self, handler: MessageHandler) -> None:
        """Register a catch-all handler for every message kind."""
        for kind in MessageKind:
            self.subscribe(kind, handler)

    @staticmethod
    def routing_key(kind: MessageKind) -> str:
        return f"mesh.{kind.value}"

    @staticmethod
    def encode(message: GossipMessage) -> bytes:
        return json.dumps(message.to_dict(), sort_keys=True, default=str).encode("utf-8")

    def validate_and_record(self, message: GossipMessage) -> bool:
        """Return False (reject) if the message is a replay or duplicate; True if valid.

        Side-effect: records the message_id for future dedup checks and prunes
        stale entries older than replay_window_seconds.
        """
        import time

        # Prune stale entries
        cutoff = time.monotonic() - self.replay_window_seconds
        self._seen = {k: v for k, v in self._seen.items() if v > cutoff}

        if message.is_replay(self.replay_window_seconds):
            return False
        if message.message_id in self._seen:
            return False

        self._seen[message.message_id] = time.monotonic()
        return True

    async def publish(self, message: GossipMessage) -> None:
        import aio_pika

        connection = await aio_pika.connect_robust(self.amqp_url)
        async with connection:
            channel = await connection.channel()
            exchange = await channel.declare_exchange(
                self.exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
            )
            await exchange.publish(
                aio_pika.Message(
                    body=self.encode(message),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                ),
                routing_key=self.routing_key(message.kind),
            )

    async def listen(self) -> None:
        """Start consuming messages from the exchange and dispatching to registered handlers.

        Runs forever — connect to RabbitMQ, bind an exclusive queue to ``mesh.#``,
        validate each message, and forward it to matching handlers.
        """
        import aio_pika

        connection = await aio_pika.connect_robust(self.amqp_url)
        channel = await connection.channel()
        exchange = await channel.declare_exchange(
            self.exchange_name, aio_pika.ExchangeType.TOPIC, durable=True
        )
        queue = await channel.declare_queue("", exclusive=True)

        for kind in MessageKind:
            await queue.bind(exchange, routing_key=self.routing_key(kind))

        logger.info(
            "Mesh consumer listening on %s (%d handlers registered)",
            self.exchange_name,
            sum(len(h) for h in self._handlers.values()),
        )

        async with queue.iterator() as queue_iter:
            async for aio_message in queue_iter:
                async with aio_message.process():
                    try:
                        data = json.loads(aio_message.body)
                        message = GossipMessage.from_dict(data)
                    except Exception:
                        logger.warning("Failed to decode mesh message", exc_info=True)
                        continue

                    if not self.validate_and_record(message):
                        logger.debug("Rejected duplicate/replay message %s", message.message_id)
                        continue

                    handlers = self._handlers.get(message.kind, [])
                    for handler in handlers:
                        try:
                            await handler(message)
                        except Exception:
                            logger.error(
                                "Handler error for %s message %s",
                                message.kind.value,
                                message.message_id,
                                exc_info=True,
                            )
