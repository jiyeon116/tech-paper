"""Explicit, versioned pilot configuration without implicit environment overrides."""
from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path


@dataclass(frozen=True)
class DynamicConfig:
    version: int = 1
    pool_users: int = 150
    total_steps: int = 100
    max_delay: int = 10
    load_levels: tuple[int, ...] = (30, 90, 150)
    load_mode: str = "persistent"
    dwell_min: int = 5
    dwell_max: int = 20
    train_episodes: int = 30
    calibration_episodes: int = 6
    eval_episodes: int = 12
    selector_dwell: int = 5
    calibration_min_tasks: int = 20
    batch_size: int = 32
    memory_size: int = 500
    history_steps: int = 10
    learning_rate: float = 0.01
    gamma: float = 0.9
    target_update: int = 200
    learn_every: int = 10
    exploration_start: float = 1.0
    exploration_end: float = 0.05
    diagnostic: bool = False

    def __post_init__(self):
        integer_fields = [f.name for f in fields(self) if f.type is int]
        for name in integer_fields:
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.version != 1 or self.total_steps <= self.max_delay:
            raise ValueError("Unsupported version or no arrival slots")
        if self.dwell_max < self.dwell_min or self.memory_size < self.batch_size:
            raise ValueError("Invalid dwell or replay bounds")
        if self.load_mode not in ("persistent", "iid"):
            raise ValueError("load_mode must be persistent or iid")
        if (not self.load_levels or len(set(self.load_levels)) != len(self.load_levels)
                or any(type(n) is not int or not 0 <= n <= self.pool_users for n in self.load_levels)):
            raise ValueError("load_levels must be unique integer counts in [0,pool_users]")
        if not 0 <= self.gamma <= 1 or not 0 < self.learning_rate < 1:
            raise ValueError("Invalid learning rate or discount")
        if not 0 <= self.exploration_end <= self.exploration_start <= 1:
            raise ValueError("Invalid exploration probabilities")
        if type(self.diagnostic) is not bool:
            raise ValueError("diagnostic must be boolean")

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        unknown = set(payload) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        if "load_levels" in payload:
            payload["load_levels"] = tuple(payload["load_levels"])
        return cls(**payload)

    def payload(self):
        return asdict(self)

    def digest(self):
        return hashlib.sha256(json.dumps(self.payload(), sort_keys=True).encode()).hexdigest()
