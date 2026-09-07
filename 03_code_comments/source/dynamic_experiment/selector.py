"""Predeclared context bins and calibration-only specialist selection."""
import numpy as np

POLICIES = ("fixed", "adaptive")


def context_bin(context):
    values = np.asarray(context, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError("Expected three finite context fields")
    score = float(np.clip(np.dot([0.6, 0.25, 0.15], values), 0, 1))
    return min(int(score * 3), 2)


def calibrate(records, minimum_tasks=20):
    """Records are calibration-only origin-bin rewards, not held-out outcomes."""
    mapping, evidence = [], []
    for bin_id in range(3):
        samples = {p: [float(r["qoe"]) for r in records
                       if r["policy"] == p and r["context_bin"] == bin_id] for p in POLICIES}
        counts = {p: len(v) for p, v in samples.items()}
        if any(not np.isfinite(v).all() for v in samples.values()):
            raise ValueError("Non-finite calibration observations")
        means = {p: float(np.mean(v)) if v else None for p, v in samples.items()}
        covered = all(n >= minimum_tasks for n in counts.values())
        selected = "fixed"
        if covered and means["adaptive"] > means["fixed"] + 1e-12:
            selected = "adaptive"
        mapping.append(selected)
        evidence.append({"bin": bin_id, "counts": counts, "means": means,
                         "selected": selected, "coverage": "sufficient" if covered else "insufficient_fixed_default"})
    return tuple(mapping), evidence


class DwellSelector:
    def __init__(self, mapping, minimum_dwell=5):
        if len(mapping) != 3 or any(p not in POLICIES for p in mapping) or minimum_dwell < 1:
            raise ValueError("Invalid selector mapping/dwell")
        self.mapping = tuple(mapping)
        self.minimum_dwell = minimum_dwell
        self.current = None
        self.last_switch = 0
        self.last_step = -1
        self.switch_count = 0

    def select(self, context, step):
        if step <= self.last_step:
            raise ValueError("Selector steps must increase")
        self.last_step = step
        desired = self.mapping[context_bin(context)]
        if self.current is None:
            self.current, self.last_switch = desired, step
        elif desired != self.current and step - self.last_switch >= self.minimum_dwell:
            self.current, self.last_switch = desired, step
            self.switch_count += 1
        return self.current
