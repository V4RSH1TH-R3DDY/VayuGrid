from __future__ import annotations

from datetime import datetime, timedelta, timezone

from mesh.messages import GossipMessage, MessageKind
from mesh.rabbitmq_bus import RabbitMQMeshBus


def _make_msg(age_seconds: float = 0.0, msg_id: str = "msg-1") -> GossipMessage:
    created_at = datetime.now(timezone.utc) - timedelta(seconds=age_seconds)
    return GossipMessage(
        kind=MessageKind.HEARTBEAT,
        source_node_id=1,
        payload={},
        created_at=created_at,
        message_id=msg_id,
    )


def test_is_replay_returns_false_for_fresh_message() -> None:
    msg = _make_msg(age_seconds=5.0)
    assert msg.is_replay(window_seconds=30) is False


def test_is_replay_returns_true_for_old_message() -> None:
    msg = _make_msg(age_seconds=60.0)
    assert msg.is_replay(window_seconds=30) is True


def test_is_replay_returns_true_for_future_message() -> None:
    # Message 10 seconds in the future
    msg = _make_msg(age_seconds=-10.0)
    assert msg.is_replay(window_seconds=30) is True


def test_rabbitmq_bus_accepts_fresh_message() -> None:
    bus = RabbitMQMeshBus(amqp_url="amqp://localhost", replay_window_seconds=30)
    msg = _make_msg(age_seconds=1.0, msg_id="fresh-1")
    assert bus.validate_and_record(msg) is True


def test_rabbitmq_bus_rejects_replay_message() -> None:
    bus = RabbitMQMeshBus(amqp_url="amqp://localhost", replay_window_seconds=30)
    msg = _make_msg(age_seconds=60.0, msg_id="old-1")
    assert bus.validate_and_record(msg) is False


def test_rabbitmq_bus_rejects_duplicate_message() -> None:
    bus = RabbitMQMeshBus(amqp_url="amqp://localhost", replay_window_seconds=30)
    msg = _make_msg(age_seconds=1.0, msg_id="dup-1")
    assert bus.validate_and_record(msg) is True  # first time: accepted
    assert bus.validate_and_record(msg) is False  # second time: duplicate
