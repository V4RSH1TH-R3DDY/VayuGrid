from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from typing import Iterable

from .models import LedgerEntry, MatchResult

GENESIS_HASH = "0" * 64


class AyaLedger:
    def __init__(self, entries: Iterable[LedgerEntry] | None = None) -> None:
        self.entries = sorted(list(entries or []), key=lambda entry: entry.sequence)

    @property
    def latest_hash(self) -> str:
        return self.entries[-1].entry_hash if self.entries else GENESIS_HASH

    def append(
        self,
        match: MatchResult,
        buyer_signature: str | None = None,
        seller_signature: str | None = None,
        metadata: dict | None = None,
    ) -> LedgerEntry:
        sequence = len(self.entries) + 1
        entry = LedgerEntry(
            sequence=sequence,
            trade_id=match.trade_id,
            ts=datetime.now(timezone.utc),
            buyer_node_id=match.buyer_node_id,
            seller_node_id=match.seller_node_id,
            quantity_kwh=match.quantity_kwh,
            cleared_price_inr_per_kwh=match.cleared_price_inr_per_kwh,
            previous_hash=self.latest_hash,
            entry_hash="",
            buyer_signature=buyer_signature,
            seller_signature=seller_signature,
            metadata=metadata or {},
        )
        entry.entry_hash = self.hash_entry(entry)
        self.entries.append(entry)
        return entry

    @staticmethod
    def hash_entry(entry: LedgerEntry) -> str:
        payload = entry.to_dict()
        payload["entry_hash"] = ""
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def verify(self) -> bool:
        previous = GENESIS_HASH
        for expected_sequence, entry in enumerate(self.entries, start=1):
            if entry.sequence != expected_sequence:
                return False
            if entry.previous_hash != previous:
                return False
            if self.hash_entry(entry) != entry.entry_hash:
                return False
            previous = entry.entry_hash
        return True
