from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingConfig:
    total_timesteps: int = 1_000_000
    steps_per_epoch: int = 2048
    n_envs: int = 4
    device: str = "cpu"
    seed: int = 42

    curriculum_stages: list[dict[str, Any]] = field(default_factory=lambda: [
        {"name": "single_home", "num_homes": 1,
         "p2p_enabled": False, "faults": False, "epochs": 500},
        {"name": "p2p_market", "num_homes": 10,
         "p2p_enabled": True, "faults": False, "epochs": 1000},
        {"name": "multi_node_fault", "num_homes": 30,
         "p2p_enabled": True, "faults": True, "epochs": 1500},
        {"name": "real_data", "num_homes": 30,
         "p2p_enabled": True, "faults": True, "epochs": 2000},
    ])

    scenario_path: str = "scenarios/phase1_default.json"
    checkpoint_dir: str = "outputs/checkpoints"
    log_dir: str = "outputs/logs"
