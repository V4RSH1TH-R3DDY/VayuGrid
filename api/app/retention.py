from __future__ import annotations

import logging
from datetime import datetime, timezone

from .db import execute, fetch_all

logger = logging.getLogger(__name__)


def process_deletion_requests():
    """
    Processes scheduled data deletion requests that have passed their 72-hour window.
    """
    now = datetime.now(timezone.utc)
    requests = fetch_all(
        """
        SELECT request_id, node_id
        FROM data_deletion_requests
        WHERE scheduled_for <= %s AND status = 'scheduled'
        """,
        (now,),
    )

    for req in requests:
        node_id = req["node_id"]
        request_id = req["request_id"]
        logger.info(f"Processing deletion request {request_id} for node {node_id}")

        try:
            # 1. Delete telemetry
            execute("DELETE FROM node_telemetry WHERE node_id = %s", (node_id,))
            # 2. Delete trades
            execute(
                "DELETE FROM trade_records WHERE buyer_node_id = %s OR seller_node_id = %s",
                (node_id, node_id),
            )
            # 3. Delete consents
            execute("DELETE FROM household_consents WHERE node_id = %s", (node_id,))
            execute("DELETE FROM critical_load_flags WHERE node_id = %s", (node_id,))
            # 4. Update request status
            execute(
                """
                UPDATE data_deletion_requests
                SET status = 'completed',
                    metadata = metadata || jsonb_build_object('completed_at', %s)
                WHERE request_id = %s
                """,
                (now.isoformat(), request_id),
            )
            logger.info(f"Successfully deleted all data for node {node_id}")
        except Exception as e:
            logger.error(f"Failed to process deletion for node {node_id}: {e}")


if __name__ == "__main__":
    process_deletion_requests()


def purge_stale_seen_messages(window_seconds: int = 60) -> int:
    """Delete seen_message_ids entries older than window_seconds.

    Called by the security scheduler every 5 minutes to keep the replay-prevention
    table from growing unboundedly.

    Returns the number of rows deleted (approximate; psycopg3 rowcount may be -1
    for some server configurations).
    """
    from .db import pool

    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM seen_message_ids"
                " WHERE received_at < now() - (%s * INTERVAL '1 second')",
                (window_seconds,),
            )
            return cur.rowcount if cur.rowcount >= 0 else 0
