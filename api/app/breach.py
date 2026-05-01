from __future__ import annotations

import logging
import smtplib
import uuid
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Any

from psycopg.types.json import Json

from .db import execute, fetch_all

logger = logging.getLogger(__name__)


class BreachNotifier:
    """Records security events and dispatches email notifications for high-severity incidents."""

    def record_event(
        self,
        event_type: str,
        severity: float,
        description: str,
        node_id: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Insert a security event and return its event_id."""
        event_id = str(uuid.uuid4())
        try:
            execute(
                """
                INSERT INTO security_events
                    (event_id, ts, event_type, severity, node_id, description, metadata)
                VALUES (%s, now(), %s, %s, %s, %s, %s)
                """,
                (
                    event_id,
                    event_type,
                    float(severity),
                    node_id,
                    description,
                    Json(metadata or {}),
                ),
            )
            logger.warning(
                "[SECURITY] %s (severity=%.2f, node=%s): %s",
                event_type,
                severity,
                node_id,
                description,
            )
        except Exception as exc:
            logger.error("Failed to record security event: %s", exc)
        return event_id

    def notify_pending(self) -> int:
        """Send email for all unnotified events with severity >= 0.7. Returns count notified."""
        from .config import settings

        pending = fetch_all(
            """
            SELECT event_id, ts, event_type, severity, node_id, description
            FROM security_events
            WHERE notified_at IS NULL AND severity >= 0.7
            ORDER BY ts
            """
        )
        if not pending:
            return 0

        notified = 0
        for event in pending:
            if self._send_notification(event, settings):
                execute(
                    "UPDATE security_events SET notified_at = %s WHERE event_id = %s",
                    (datetime.now(timezone.utc), event["event_id"]),
                )
                notified += 1
        return notified

    def get_recent_events(self, limit: int = 50) -> list[dict]:
        """Return the most recent security events."""
        try:
            return fetch_all(
                "SELECT event_id, ts, event_type, severity, node_id, description, notified_at "
                "FROM security_events ORDER BY ts DESC LIMIT %s",
                (limit,),
            )
        except Exception:
            return []

    def _send_notification(self, event: dict, settings: Any) -> bool:
        """Send a single breach notification. Returns True on success."""
        if not settings.smtp_host or not settings.breach_notify_email:
            # No SMTP configured — log at ERROR level as the fallback notification
            logger.error(
                "[BREACH ALERT] %s severity=%.2f node=%s — %s",
                event["event_type"],
                event["severity"],
                event["node_id"],
                event["description"],
            )
            return True  # treated as "notified" via logging

        try:
            body = (
                f"VayuGrid Security Alert\n\n"
                f"Event type : {event['event_type']}\n"
                f"Severity   : {event['severity']:.2f}\n"
                f"Node ID    : {event.get('node_id', 'N/A')}\n"
                f"Time       : {event['ts']}\n"
                f"Description: {event['description']}\n"
            )
            msg = MIMEText(body)
            msg["Subject"] = (
                f"[VayuGrid] Security Event: {event['event_type']}"
                f" (severity {event['severity']:.2f})"
            )
            msg["From"] = settings.smtp_from
            msg["To"] = settings.breach_notify_email

            with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=10) as server:
                if settings.smtp_user:
                    server.starttls()
                    server.login(settings.smtp_user, settings.smtp_password)
                server.sendmail(settings.smtp_from, [settings.breach_notify_email], msg.as_string())
            return True
        except Exception as exc:
            logger.error("Failed to send breach notification email: %s", exc)
            return False


# Module-level singleton
notifier = BreachNotifier()
