"""Independent D3QN learners for the dynamic-population experiment.

This module deliberately does not use ``DuelingDoubleDeepQNetwork.learn``.
The legacy learner reconstructs recurrent batches from adjacent completion
records, whereas this experiment stores the chronological recurrent windows
that existed when each action was taken.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
from typing import Any, Deque, Dict, Mapping

import numpy as np


POLICIES = ("fixed", "adaptive", "contextual")


@dataclass(frozen=True)
class Transition:
    """An action-time record whose reward is attached when work completes."""

    origin_expert: str
    origin_energy_weight: float
    state: np.ndarray
    history_window: np.ndarray
    action: int
    next_state: np.ndarray
    next_history_window: np.ndarray
    origin_done: bool
    context_bin: int


@dataclass(frozen=True)
class _ReplayItem:
    transition: Transition
    reward: float


@dataclass
class _AgentSlot:
    network: Any
    replay: Deque[_ReplayItem]
    action_rng: np.random.Generator
    replay_rng: np.random.Generator
    update_count: int = 0


def _load_runtime_class():
    # TensorFlow is intentionally imported only when real agents are built so
    # pure unit tests can exercise replay semantics without that dependency.
    from qeco_runtime.D3QN import PerAgentD3QNSessions

    return PerAgentD3QNSessions


class SpecialistBank:
    """Own three independent UE-specific D3QN policy banks.

    ``fixed`` and ``adaptive`` are the two gate specialists. ``contextual`` is
    the independently trained no-gate comparison policy.  Every policy has a
    separate network, optimizer state, replay buffer, update counter, and RNG
    for each UE.
    """

    policies = POLICIES

    def __init__(
        self,
        config: Any,
        seed: int,
        n_actions: int,
        n_features: int,
        n_lstm_features: int,
    ) -> None:
        self.config = config
        self.seed = int(seed)
        self.n_actions = int(n_actions)
        self.n_features = int(n_features)
        self.n_lstm_features = int(n_lstm_features)
        self.pool_users = int(config.pool_users)
        self.history_steps = int(config.history_steps)
        self.batch_size = int(config.batch_size)
        self.memory_size = int(config.memory_size)
        self.gamma = float(config.gamma)
        self.target_update = int(config.target_update)
        self._closed = False

        self._validate_configuration()
        runtime_class = _load_runtime_class()
        self._runtime = runtime_class(seed=self.seed)
        self._slots: Dict[str, list[_AgentSlot]] = {
            policy: [] for policy in self.policies
        }

        seed_sequence = np.random.SeedSequence(self.seed)
        child_seeds = iter(seed_sequence.spawn(len(self.policies) * self.pool_users * 2))
        try:
            for policy in self.policies:
                for ue in range(self.pool_users):
                    network = self._runtime.build_agent(
                        f"{policy}_ue_{ue}",
                        n_actions=self.n_actions,
                        n_features=self.n_features,
                        n_lstm_features=self.n_lstm_features,
                        n_time=int(config.total_steps),
                        learning_rate=float(config.learning_rate),
                        reward_decay=self.gamma,
                        replace_target_iter=self.target_update,
                        memory_size=self.memory_size,
                        batch_size=self.batch_size,
                        n_lstm_step=self.history_steps,
                        dueling=True,
                        double_q=True,
                    )
                    self._slots[policy].append(
                        _AgentSlot(
                            network=network,
                            replay=deque(maxlen=self.memory_size),
                            action_rng=np.random.default_rng(next(child_seeds)),
                            replay_rng=np.random.default_rng(next(child_seeds)),
                        )
                    )
            self._runtime.initialize()
        except BaseException:
            self.close()
            raise

    def _validate_configuration(self) -> None:
        positive_ints = {
            "pool_users": self.pool_users,
            "total_steps": int(self.config.total_steps),
            "batch_size": self.batch_size,
            "memory_size": self.memory_size,
            "history_steps": self.history_steps,
            "target_update": self.target_update,
            "n_actions": self.n_actions,
            "n_features": self.n_features,
            "n_lstm_features": self.n_lstm_features,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        if invalid:
            raise ValueError(f"Expected positive values for: {', '.join(invalid)}")
        if self.batch_size > self.memory_size:
            raise ValueError("batch_size cannot exceed memory_size")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between 0 and 1")
        if float(self.config.learning_rate) <= 0.0:
            raise ValueError("learning_rate must be positive")

    def _slot(self, policy: str, ue: int) -> _AgentSlot:
        if policy not in self._slots:
            raise ValueError(f"Unknown policy {policy!r}; expected one of {self.policies}")
        if not 0 <= int(ue) < self.pool_users:
            raise IndexError(f"UE index {ue} is outside [0, {self.pool_users})")
        return self._slots[policy][int(ue)]

    def _state(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        if array.shape != (self.n_features,):
            raise ValueError(f"{name} must have shape ({self.n_features},), got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value")
        return np.array(array, copy=True)

    def _window(self, value: Any, name: str) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32)
        expected = (self.history_steps, self.n_lstm_features)
        if array.shape != expected:
            raise ValueError(f"{name} must have shape {expected}, got {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains a non-finite value")
        return np.array(array, copy=True)

    @staticmethod
    def _require_finite(name: str, value: Any) -> None:
        if not np.isfinite(np.asarray(value)).all():
            raise FloatingPointError(f"{name} contains NaN or infinity")

    @staticmethod
    def _predict_eval(network: Any, states: np.ndarray, windows: np.ndarray) -> np.ndarray:
        if hasattr(network, "predict_eval"):
            return np.asarray(network.predict_eval(states, windows), dtype=np.float32)
        return np.asarray(
            network.sess.run(
                network.q_eval,
                feed_dict={network.s: states, network.lstm_s: windows},
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _predict_next(network: Any, states: np.ndarray, windows: np.ndarray) -> np.ndarray:
        if hasattr(network, "predict_next"):
            return np.asarray(network.predict_next(states, windows), dtype=np.float32)
        return np.asarray(
            network.sess.run(
                network.q_next,
                feed_dict={network.s_: states, network.lstm_s_: windows},
            ),
            dtype=np.float32,
        )

    @staticmethod
    def _sync_target(network: Any) -> None:
        if hasattr(network, "sync_target"):
            network.sync_target()
            return
        network.sess.run(network.replace_target_op)

    @staticmethod
    def _optimize(
        network: Any,
        states: np.ndarray,
        windows: np.ndarray,
        targets: np.ndarray,
    ) -> float:
        if hasattr(network, "train_batch"):
            return float(network.train_batch(states, windows, targets))
        _, loss = network.sess.run(
            [network._train_op, network.loss],
            feed_dict={
                network.s: states,
                network.lstm_s: windows,
                network.q_target: targets,
            },
        )
        return float(loss)

    def choose(
        self,
        policy: str,
        ue: int,
        state: Any,
        window: Any,
        exploration: float = 0,
    ) -> int:
        """Choose from all actions; zero exploration is mutation-free argmax."""

        exploration = float(exploration)
        if not 0.0 <= exploration <= 1.0:
            raise ValueError("exploration must be between 0 and 1")
        slot = self._slot(policy, ue)
        state_array = self._state(state, "state")
        window_array = self._window(window, "window")
        if exploration and slot.action_rng.random() < exploration:
            return int(slot.action_rng.integers(0, self.n_actions))
        q_values = self._predict_eval(
            slot.network, state_array[np.newaxis, :], window_array[np.newaxis, :, :]
        )
        if q_values.shape != (1, self.n_actions):
            raise RuntimeError(
                f"Policy {policy!r} returned Q shape {q_values.shape}, "
                f"expected (1, {self.n_actions})"
            )
        self._require_finite(f"Policy {policy!r} Q values", q_values)
        return int(np.argmax(q_values[0]))

    def store(
        self,
        policy: str,
        ue: int,
        transition: Transition,
        reward: float,
    ) -> None:
        """Attach a delayed reward to the policy that originated the action."""

        slot = self._slot(policy, ue)
        if transition.origin_expert != policy:
            raise ValueError(
                f"Transition originated from {transition.origin_expert!r}, not {policy!r}"
            )
        if not 0 <= int(transition.action) < self.n_actions:
            raise ValueError(f"action {transition.action} is outside [0, {self.n_actions})")
        if not np.isfinite(float(transition.origin_energy_weight)):
            raise ValueError("origin_energy_weight must be finite")
        if int(transition.context_bin) not in (0, 1, 2):
            raise ValueError("context_bin must be 0, 1, or 2")
        if not np.isfinite(float(reward)):
            raise ValueError("reward must be finite")

        normalized = Transition(
            origin_expert=transition.origin_expert,
            origin_energy_weight=float(transition.origin_energy_weight),
            state=self._state(transition.state, "transition.state"),
            history_window=self._window(
                transition.history_window, "transition.history_window"
            ),
            action=int(transition.action),
            next_state=self._state(transition.next_state, "transition.next_state"),
            next_history_window=self._window(
                transition.next_history_window, "transition.next_history_window"
            ),
            origin_done=bool(transition.origin_done),
            context_bin=int(transition.context_bin),
        )
        slot.replay.append(_ReplayItem(normalized, float(reward)))

    def _learn_slot(self, slot: _AgentSlot) -> float | None:
        if len(slot.replay) < self.batch_size:
            return None
        if slot.update_count % self.target_update == 0:
            self._sync_target(slot.network)

        indices = slot.replay_rng.choice(
            len(slot.replay), size=self.batch_size, replace=False
        )
        items = [slot.replay[int(index)] for index in indices]
        states = np.stack([item.transition.state for item in items])
        windows = np.stack([item.transition.history_window for item in items])
        next_states = np.stack([item.transition.next_state for item in items])
        next_windows = np.stack(
            [item.transition.next_history_window for item in items]
        )

        q_eval = self._predict_eval(slot.network, states, windows)
        q_eval_next = self._predict_eval(slot.network, next_states, next_windows)
        q_target_next = self._predict_next(slot.network, next_states, next_windows)
        expected = (self.batch_size, self.n_actions)
        if (
            q_eval.shape != expected
            or q_eval_next.shape != expected
            or q_target_next.shape != expected
        ):
            raise RuntimeError("D3QN prediction shape does not match configured batch/actions")
        self._require_finite("q_eval", q_eval)
        self._require_finite("q_eval_next", q_eval_next)
        self._require_finite("q_target_next", q_target_next)

        targets = np.array(q_eval, copy=True)
        batch_index = np.arange(self.batch_size)
        actions = np.asarray([item.transition.action for item in items], dtype=np.int64)
        rewards = np.asarray([item.reward for item in items], dtype=np.float32)
        done = np.asarray(
            [item.transition.origin_done for item in items], dtype=np.float32
        )
        greedy_next = np.argmax(q_eval_next, axis=1)
        # A zero task-size observation has only the idle/local action. Edge
        # actions are unavailable, regardless of their untrained Q estimates.
        greedy_next[next_states[:, 0] == 0.0] = 0
        bootstrap = q_target_next[batch_index, greedy_next]
        self._require_finite("selected target bootstrap", bootstrap)
        with np.errstate(over="ignore", invalid="ignore"):
            targets[batch_index, actions] = (
                rewards + self.gamma * (1.0 - done) * bootstrap
            )
        self._require_finite("training targets", targets)

        loss = self._optimize(slot.network, states, windows, targets)
        self._require_finite("training loss", loss)
        slot.update_count += 1
        if hasattr(slot.network, "learn_step_counter"):
            slot.network.learn_step_counter = slot.update_count
        return loss

    def learn(self, policy: str) -> Mapping[str, Any]:
        """Train every ready UE in one policy and return bounded diagnostics."""

        if policy not in self._slots:
            raise ValueError(f"Unknown policy {policy!r}; expected one of {self.policies}")
        losses = [
            loss
            for slot in self._slots[policy]
            for loss in [self._learn_slot(slot)]
            if loss is not None
        ]
        return {
            "policy": policy,
            "losses": losses,
            "agents_updated": len(losses),
            "update_count": sum(slot.update_count for slot in self._slots[policy]),
        }

    @staticmethod
    def _variable_values(
        network: Any,
    ) -> tuple[list[tuple[str, np.ndarray]], list[tuple[str, np.ndarray]], int]:
        if hasattr(network, "snapshot_variables"):
            variables = network.snapshot_variables()
            pairs = sorted(
                (str(name), np.asarray(value)) for name, value in variables.items()
            )
            trainable_names = set(getattr(network, "trainable_variable_names", variables))
            trainable_pairs = [pair for pair in pairs if pair[0] in trainable_names]
            count = sum(value.size for _, value in trainable_pairs)
            return pairs, trainable_pairs, int(count)

        graph = network.q_eval.graph
        variables = graph.get_collection("variables", scope=network.scope)
        trainable = graph.get_collection("trainable_variables", scope=network.scope)
        values = network.sess.run(variables)
        pairs = sorted(
            ((variable.name, np.asarray(value)) for variable, value in zip(variables, values)),
            key=lambda item: item[0],
        )
        trainable_values = network.sess.run(trainable)
        trainable_pairs = sorted(
            (
                (variable.name, np.asarray(value))
                for variable, value in zip(trainable, trainable_values)
            ),
            key=lambda item: item[0],
        )
        count = sum(np.prod(variable.shape.as_list()) for variable in trainable)
        return pairs, trainable_pairs, int(count)

    @staticmethod
    def _digest_pairs(pairs: list[tuple[str, np.ndarray]]) -> str:
        digest = hashlib.sha256()
        for name, value in pairs:
            contiguous = np.ascontiguousarray(value)
            SpecialistBank._require_finite(f"snapshot variable {name!r}", contiguous)
            digest.update(name.encode("utf-8"))
            digest.update(str(contiguous.dtype).encode("ascii"))
            digest.update(repr(contiguous.shape).encode("ascii"))
            digest.update(contiguous.tobytes())
        return digest.hexdigest()

    @classmethod
    def _network_digest(cls, network: Any) -> tuple[str, str, int]:
        pairs, trainable_pairs, parameter_count = cls._variable_values(network)
        return (
            cls._digest_pairs(pairs),
            cls._digest_pairs(trainable_pairs),
            parameter_count,
        )

    def snapshot(self) -> Mapping[str, Any]:
        """Return compact mutation evidence without serializing model state."""

        policy_metadata: Dict[str, Any] = {}
        for policy, slots in self._slots.items():
            digests = []
            trainable_digests = []
            parameter_counts = []
            for slot in slots:
                digest, trainable_digest, count = self._network_digest(slot.network)
                digests.append(digest)
                trainable_digests.append(trainable_digest)
                parameter_counts.append(count)
            combined = hashlib.sha256("".join(digests).encode("ascii")).hexdigest()
            combined_trainable = hashlib.sha256(
                "".join(trainable_digests).encode("ascii")
            ).hexdigest()
            policy_metadata[policy] = {
                "weight_digest": combined,
                "agent_weight_digests": digests,
                "trainable_weight_digest": combined_trainable,
                "agent_trainable_weight_digests": trainable_digests,
                "network_count": len(slots),
                "parameter_count": int(sum(parameter_counts)),
                "agent_parameter_counts": parameter_counts,
                "update_count": int(sum(slot.update_count for slot in slots)),
                "replay_count": int(sum(len(slot.replay) for slot in slots)),
            }
        return {
            "schema_version": 1,
            "seed": self.seed,
            "pool_users": self.pool_users,
            "n_actions": self.n_actions,
            "n_features": self.n_features,
            "n_lstm_features": self.n_lstm_features,
            "history_steps": self.history_steps,
            "runtime": self._runtime.metadata(),
            "policies": policy_metadata,
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        runtime = getattr(self, "_runtime", None)
        if runtime is not None:
            runtime.close()

    def __enter__(self) -> "SpecialistBank":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
