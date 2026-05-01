"""
VayuGrid Security Scheduler
============================
Runs recurring background security-maintenance tasks:

  • Every  5 min  — purge stale seen_message_ids (replay-prevention table)
  • Every 15 min  — dispatch breach notifications for unnotified security events
  • Every  1 hour — process pending data-deletion requests (DPDP 72-hour window)
  • Every  6 hours — refresh the Isolation Forest anomaly-detection model
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler

from api.app.db import pool

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job functions
# ---------------------------------------------------------------------------


def purge_seen_messages() -> None:
    try:
        from api.app.retention import purge_stale_seen_messages

        deleted = purge_stale_seen_messages(window_seconds=60)
        logger.info("Purged %d stale seen_message_ids entries.", deleted)
    except Exception as exc:
        logger.error("purge_seen_messages failed: %s", exc)


def send_breach_notifications() -> None:
    try:
        from api.app.breach import notifier

        count = notifier.notify_pending()
        if count:
            logger.info("Dispatched %d breach notification(s).", count)
    except Exception as exc:
        logger.error("send_breach_notifications failed: %s", exc)


def run_deletion_processor() -> None:
    try:
        from api.app.retention import process_deletion_requests

        logger.info("Running data-deletion processor\u2026")
        process_deletion_requests()
    except Exception as exc:
        logger.error("run_deletion_processor failed: %s", exc)


def refresh_anomaly_model() -> None:
    try:
        from api.app.anomaly import detector

        logger.info("Refreshing anomaly-detection model\u2026")
        detector.fit_from_db()
        logger.info("Anomaly model refreshed (trained=%s).", detector.is_trained)
    except Exception as exc:
        logger.error("refresh_anomaly_model failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    pool.open(wait=True, timeout=30)

    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(purge_seen_messages, "interval", minutes=5, id="purge_seen_messages")
    scheduler.add_job(send_breach_notifications, "interval", minutes=15, id="breach_notifications")
    scheduler.add_job(run_deletion_processor, "interval", hours=1, id="deletion_processor")
    scheduler.add_job(refresh_anomaly_model, "interval", hours=6, id="anomaly_refresh")

    logger.info("VayuGrid security scheduler started.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Security scheduler shutting down.")
    finally:
        pool.close()


if __name__ == "__main__":
    main()
