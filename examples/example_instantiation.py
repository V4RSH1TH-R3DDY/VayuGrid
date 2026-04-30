from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def main() -> None:
    from simulator.simulator import run_simulation_from_config

    config_path = Path("scenarios/phase1_default.json")
    result = run_simulation_from_config(config_path)

    output_dir = Path("outputs/phase1_example")
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(str(output_dir))

    print("Simulation complete")
    print(f"Node rows: {len(result.node_timeseries):,}")
    print(f"Transformer rows: {len(result.transformer_timeseries):,}")
    print(f"Event rows: {len(result.event_log):,}")
    print(f"Saved to: {output_dir}")


if __name__ == "__main__":
    main()
