from __future__ import annotations

from datetime import datetime, timezone

from trading.ledger import AyaLedger
from trading.models import MatchResult
from trading.signatures import generate_keypair, sign_payload, verify_payload


def _match(trade_id: str = "trade-1") -> MatchResult:
    return MatchResult(
        trade_id=trade_id,
        buyer_order_id="buy-1",
        seller_order_id="sell-1",
        buyer_node_id=2,
        seller_node_id=1,
        quantity_kwh=2.0,
        cleared_price_inr_per_kwh=8.0,
        matched_at=datetime(2026, 5, 1, 10, 0, tzinfo=timezone.utc),
    )


def test_ledger_hash_chain_detects_tampering() -> None:
    ledger = AyaLedger()
    entry = ledger.append(_match())

    assert ledger.verify()

    entry.quantity_kwh = 3.0

    assert not ledger.verify()


def test_ledger_links_entries_to_previous_hash() -> None:
    ledger = AyaLedger()
    first = ledger.append(_match("trade-1"))
    second = ledger.append(_match("trade-2"))

    assert second.previous_hash == first.entry_hash
    assert ledger.verify()


def test_ed25519_signature_round_trip() -> None:
    private_key, public_key = generate_keypair()
    payload = {"trade_id": "trade-1", "quantity_kwh": 2.0, "price": 8.0}

    signature = sign_payload(private_key, payload)

    assert verify_payload(public_key, payload, signature)
    assert not verify_payload(public_key, {**payload, "price": 9.0}, signature)
