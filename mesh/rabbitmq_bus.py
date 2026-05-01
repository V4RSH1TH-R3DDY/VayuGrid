from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from .messages import GossipMessage, MessageKind

MessageHandler = Callable[[dict[str, Any]], Awaitable[None]]


class RabbitMQMeshBus:
    """RabbitMQ adapter for the Phase 4 mesh topics.

    The real node mesh will use libp2p. This adapter keeps the neighborhood-server
    contracts stable while local development uses the RabbitMQ service already in Compose.
    """

    def __init__(
        self,
        amqp_url: str,
        exchange_name: str = "vayugrid.mesh",
    ) -> None:
        self.amqp_url = amqp_url
        self.exchange_name = exchange_name

    @staticmethod
    def routing_key(kind: MessageKind) -> str:
        return f"mesh.{kind.value}"

    @staticmethod
    def encode(message: GossipMessage) -> bytes:
        return json.dumps(message.to_dict(), sort_keys=True, default=str).encode("utf-8")

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
