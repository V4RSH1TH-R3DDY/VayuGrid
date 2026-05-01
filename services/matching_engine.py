from __future__ import annotations

import os
import time

from api.app.db import pool
from trading.db_matching import settle_open_orders


def main() -> None:
    interval_seconds = float(os.getenv("MATCHING_INTERVAL_SECONDS", "10"))
    pool.open(wait=True, timeout=30)
    try:
        while True:
            settle_open_orders()
            time.sleep(interval_seconds)
    finally:
        pool.close()


if __name__ == "__main__":
    main()
