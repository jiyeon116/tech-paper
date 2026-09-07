from __future__ import annotations

import os
from dataclasses import dataclass

from experiment_support import (
    DENSITY_REGIMES,
    SharedExperiment,
    adapt2_regime_policy_map,
    classify_density_regime,
    qeco_adapt_energy_weight,
    qeco_adapt_gating_strength,
    qeco_adapt_reward_function,
    qoe_function,
    users_per_edge,
)


def normalize_algorithm_name(value: str) -> str:
    return value.replace("_", "-").lower()


@dataclass(frozen=True)
class AlgorithmInfo:
    name: str
    label: str
    paper_family: str
    source_type: str
    implementation_note: str


class ChannelAlgorithmAdapter:
    """Algorithm boundary used by the common evaluator.

    The evaluator owns only the environment loop. Algorithm-specific network
    shape, action post-processing, reward definition, and metadata stay behind
    this registry boundary.
    """

    info = AlgorithmInfo(
        name="base",
        label="Base",
        paper_family="abstract",
        source_type="adapter",
        implementation_note="base adapter",
    )
    agent_kwargs: dict[str, bool] = {}

    def __init__(self, env, config):
        self.env = env
        self.config = config
        self._agent_runtime = None
        # Active-user count of the current episode. Equals config.N_UE in fixed
        # density mode; the evaluator updates it per episode in random mode.
        self.active_users = int(getattr(config, "N_UE", SharedExperiment.NUM_USERS))
        self.episode_index = 0

    @property
    def name(self) -> str:
        return self.info.name

    def on_episode_start(self, episode: int, active_users: int) -> None:
        """Evaluator hook: called before each episode with the live active-user count."""
        self.episode_index = int(episode)
        self.active_users = int(active_users)

    def episode_selection(self) -> dict | None:
        """Per-episode selection record for adapters that switch behaviour by density."""
        return None

    def build_agents(self):
        from qeco_runtime.D3QN import PerAgentD3QNSessions, SharedD3QNSession

        common_kwargs = dict(
            learning_rate=self.config.LEARNING_RATE,
            reward_decay=self.config.REWARD_DECAY,
            e_greedy=self.config.E_GREEDY,
            replace_target_iter=self.config.N_NETWORK_UPDATE,
            memory_size=self.config.MEMORY_SIZE,
            **self.agent_kwargs,
        )
        session_mode = os.environ.get("QECO_AGENT_SESSION_MODE", "shared").strip().lower()
        if session_mode == "shared":
            self._agent_runtime = SharedD3QNSession(seed=SharedExperiment.RANDOM_SEED)
        elif session_mode == "per-agent":
            self._agent_runtime = PerAgentD3QNSessions(seed=SharedExperiment.RANDOM_SEED)
        else:
            raise ValueError(
                "QECO_AGENT_SESSION_MODE must be 'shared' or 'per-agent', "
                f"got {session_mode!r}."
            )
        agents = [
            self._agent_runtime.build_agent(
                f"ue_{ue_index}",
                self.env.n_actions,
                self.env.n_features,
                self.env.n_lstm_state,
                self.env.n_time,
                **common_kwargs,
            )
            for ue_index in range(self.config.N_UE)
        ]
        self._agent_runtime.initialize()
        return agents

    def close_agents(self):
        if self._agent_runtime is not None:
            self._agent_runtime.close()
            self._agent_runtime = None

    def select_action(self, agent, observation) -> int:
        return int(agent.choose_action(observation))

    def reward(self, env, ue_index: int, time_index: int) -> float:
        return qoe_function(
            env.process_delay[time_index, ue_index],
            env.max_delay,
            env.unfinish_task[time_index, ue_index],
            env.ue_energy_state[ue_index],
            env.ue_comp_energy[time_index, ue_index],
            env.ue_tran_energy[time_index, ue_index],
            env.edge_comp_energy[time_index, ue_index],
            env.ue_idle_energy[time_index, ue_index],
        )

    def metadata(self) -> dict:
        session_mode = os.environ.get("QECO_AGENT_SESSION_MODE", "shared").strip().lower()
        if session_mode == "per-agent":
            agent_execution = {
                "agent_session_model": "one_tf_graph_and_session_per_ue",
                "agent_variable_scope": "one_scope_per_ue_graph",
                "gpu_allow_growth": False,
                "configured_agent_count": self.config.N_UE,
            }
        else:
            agent_execution = {
                "agent_session_model": "one_tf_graph_and_session_per_algorithm_process",
                "agent_variable_scope": "one_scope_per_ue",
                "gpu_allow_growth": True,
                "configured_agent_count": self.config.N_UE,
            }
        payload = {
            "name": self.info.name,
            "label": self.info.label,
            "paper_family": self.info.paper_family,
            "source_type": self.info.source_type,
            "implementation_module": f"{self.__class__.__module__}.{self.__class__.__name__}",
            "implementation_note": self.info.implementation_note,
            "agent_execution": agent_execution,
        }
        if self._agent_runtime is not None:
            payload["agent_execution"] = self._agent_runtime.metadata()
        return payload


class TangWongAdapter(ChannelAlgorithmAdapter):
    info = AlgorithmInfo(
        name="tang-wong",
        label="Tang&Wong DQN",
        paper_family="Deep Reinforcement Learning for Task Offloading in MEC Systems",
        source_type="local-paper-adapter",
        implementation_note="DQN/LSTM-style baseline adapter using the common AP-cluster environment",
    )
    agent_kwargs = {
        "dueling": False,
        "double_q": False,
    }


class QecoAdapter(ChannelAlgorithmAdapter):
    info = AlgorithmInfo(
        name="qeco",
        label="QECO",
        paper_family="QECO QoE-oriented computation offloading",
        source_type="local-paper-adapter",
        implementation_note="D3QN/LSTM QECO adapter using the common AP-cluster environment",
    )
    agent_kwargs = {
        "dueling": True,
        "double_q": True,
    }


class QecoAdaptAblationBase(QecoAdapter):
    energy_weight_mode = "qeco"
    gate_mode = "none"

    def select_action(self, agent, observation) -> int:
        proposed_action = super().select_action(agent, observation)
        if self.gate_mode == "none":
            return proposed_action
        return self._apply_access_gate(observation, proposed_action)

    def _gate_strength(self) -> float:
        if self.gate_mode == "fixed":
            return 1.0
        if self.gate_mode == "adaptive":
            return qeco_adapt_gating_strength(self.active_users)
        return 0.0

    def _apply_access_gate(self, observation, proposed_action: int) -> int:
        if proposed_action == 0:
            return 0
        edge_index = min(max(int(proposed_action) - 1, 0), self.env.n_edge - 1)
        gating_strength = self._gate_strength()
        task_size = float(observation[0])
        estimated_local_time = float(observation[1])
        estimated_tran_time = float(observation[2])
        edge_backlog = float(observation[3 + edge_index])
        energy_state = float(observation[self.env.base_n_features - 1])
        access_score = float(observation[self.env.base_n_features + edge_index])

        min_energy_state = SharedExperiment.QECO_ADAPT_MIN_ENERGY_STATE * gating_strength
        edge_backlog_threshold = SharedExperiment.QECO_ADAPT_EDGE_BACKLOG_THRESHOLD / max(gating_strength, 1e-6)
        size_threshold = SharedExperiment.QECO_ADAPT_SIZE_THRESHOLD * gating_strength
        access_score_threshold = 0.15 * gating_strength

        if energy_state <= min_energy_state:
            return 0
        if edge_backlog >= edge_backlog_threshold:
            return 0
        if task_size <= size_threshold and estimated_tran_time >= 0:
            return 0
        if estimated_local_time >= 0 and estimated_tran_time >= 0 and estimated_local_time <= estimated_tran_time:
            return 0
        if access_score <= access_score_threshold:
            return 0
        return int(proposed_action)

    def _energy_weight(self) -> float:
        if self.energy_weight_mode == "qeco":
            return 1.0
        if self.energy_weight_mode == "fixed":
            return SharedExperiment.QECO_ADAPT_BASE_ENERGY_WEIGHT
        if self.energy_weight_mode == "adaptive":
            return qeco_adapt_energy_weight(self.active_users)
        raise ValueError(f"Unknown energy_weight_mode: {self.energy_weight_mode}")

    def reward(self, env, ue_index: int, time_index: int) -> float:
        if self.energy_weight_mode == "qeco":
            return super().reward(env, ue_index, time_index)
        return qeco_adapt_reward_function(
            env.process_delay[time_index, ue_index],
            env.max_delay,
            env.unfinish_task[time_index, ue_index],
            env.ue_energy_state[ue_index],
            env.ue_comp_energy[time_index, ue_index],
            env.ue_tran_energy[time_index, ue_index],
            env.edge_comp_energy[time_index, ue_index],
            env.ue_idle_energy[time_index, ue_index],
            self._energy_weight(),
        )

    def metadata(self) -> dict:
        payload = super().metadata()
        payload["qeco_adapt_ablation"] = {
            "energy_weight_mode": self.energy_weight_mode,
            "energy_weight": self._energy_weight(),
            "gate_mode": self.gate_mode,
            "gating_strength": self._gate_strength(),
            "access_gate": "none" if self.gate_mode == "none" else "reject offloading when energy/backlog/size/local-time/access-score checks fail",
            "load_source": "episode_active_users",
            "active_users_at_metadata": self.active_users,
        }
        return payload


class QecoRewardWeightOnlyAdapter(QecoAdaptAblationBase):
    info = AlgorithmInfo(
        name="qeco-reward-weight-only",
        label="QECO + reward weight",
        paper_family="QECO-ADAPT ablation",
        source_type="local-ablation-adapter",
        implementation_note="QECO D3QN/LSTM with adaptive energy-weight reward only; no action gate",
    )
    energy_weight_mode = "adaptive"
    gate_mode = "none"


class QecoGateOnlyAdapter(QecoAdaptAblationBase):
    info = AlgorithmInfo(
        name="qeco-gate-only",
        label="QECO + gate only",
        paper_family="QECO-ADAPT ablation",
        source_type="local-ablation-adapter",
        implementation_note="QECO D3QN/LSTM with original QECO reward and adaptive action gate",
    )
    energy_weight_mode = "qeco"
    gate_mode = "adaptive"


class QecoFixedWeightFixedGateAdapter(QecoAdaptAblationBase):
    info = AlgorithmInfo(
        name="qeco-fixed-weight-fixed-gate",
        label="QECO + fixed weight/gate",
        paper_family="QECO-ADAPT ablation",
        source_type="local-ablation-adapter",
        implementation_note="QECO D3QN/LSTM with fixed base energy weight and fixed gate strength",
    )
    energy_weight_mode = "fixed"
    gate_mode = "fixed"


class QecoAdaptiveWeightFixedGateAdapter(QecoAdaptAblationBase):
    info = AlgorithmInfo(
        name="qeco-adaptive-weight-fixed-gate",
        label="QECO + adaptive weight/fixed gate",
        paper_family="QECO-ADAPT ablation",
        source_type="local-ablation-adapter",
        implementation_note="QECO D3QN/LSTM with adaptive energy weight and fixed gate strength",
    )
    energy_weight_mode = "adaptive"
    gate_mode = "fixed"


class QecoFixedWeightAdaptiveGateAdapter(QecoAdaptAblationBase):
    info = AlgorithmInfo(
        name="qeco-fixed-weight-adaptive-gate",
        label="QECO + fixed weight/adaptive gate",
        paper_family="QECO-ADAPT ablation",
        source_type="local-ablation-adapter",
        implementation_note="QECO D3QN/LSTM with fixed base energy weight and adaptive action gate",
    )
    energy_weight_mode = "fixed"
    gate_mode = "adaptive"


class QecoAdaptAdapter(QecoAdaptAblationBase):
    info = AlgorithmInfo(
        name="qeco-adapt",
        label="QECO-ADAPT",
        paper_family="QECO-ADAPT load-adaptive extension",
        source_type="local-paper-adapter",
        implementation_note="QECO adapter plus adaptive reward and access-aware offloading gate",
    )
    energy_weight_mode = "adaptive"
    gate_mode = "adaptive"


class QecoAdaptFullAdapter(QecoAdaptAdapter):
    info = AlgorithmInfo(
        name="qeco-adapt-full",
        label="QECO-ADAPT full",
        paper_family="QECO-ADAPT ablation",
        source_type="local-ablation-adapter",
        implementation_note="Alias of QECO-ADAPT full adaptive reward and adaptive action gate for ablation tables",
    )


# ---------------------------------------------------------------------------
# qeco-adapt2 family
# ---------------------------------------------------------------------------


class QecoAdaptGeneralAdapter(QecoAdaptiveWeightFixedGateAdapter):
    """General algorithm of the adapt2 family: adaptive energy weight + fixed gate (sigma=1).

    Behaviourally identical to ``qeco-adaptive-weight-fixed-gate``; the alias
    exists so adapt2 experiments and result folders name it as the general
    policy rather than as an ablation cell.
    """

    info = AlgorithmInfo(
        name="qeco-adapt-general",
        label="QECO-ADAPT general",
        paper_family="QECO-ADAPT2 family",
        source_type="local-paper-adapter",
        implementation_note="Alias of adaptive energy weight + fixed gate (sigma=1); adapt2 general-regime policy",
    )


class QecoAdaptiveGateAdapter(QecoAdaptAdapter):
    """Adaptive-gate algorithm of the adapt2 family: adaptive energy weight + load-conditioned gate.

    Behaviourally identical to ``qeco-adapt`` / ``qeco-adapt-full``. In random
    density mode the gate scale sigma=g(N_active) and the energy weight follow
    the episode's active-user count.
    """

    info = AlgorithmInfo(
        name="qeco-adaptive-gate",
        label="QECO adaptive gate",
        paper_family="QECO-ADAPT2 family",
        source_type="local-paper-adapter",
        implementation_note="Alias of adaptive energy weight + adaptive (load-conditioned) gate; adapt2 dense/sparse-regime policy",
    )


class QecoAdapt2Adapter(QecoAdaptAblationBase):
    """QECO-ADAPT2: density-regime policy selection over the QECO-family backbone.

    At the start of every episode the adapter classifies the active-user
    density (users per edge) into sparse / general / dense and switches its
    (energy_weight_mode, gate_mode) configuration to the sub-policy registered
    for that regime. All sub-policies share the same D3QN/LSTM network and
    replay memory, so the switch changes only the pre-execution gate and the
    reward weighting, never the network or the action space.
    """

    info = AlgorithmInfo(
        name="qeco-adapt2",
        label="QECO-ADAPT2",
        paper_family="QECO-ADAPT2 density-adaptive selection",
        source_type="local-paper-adapter",
        implementation_note=(
            "Selects a QECO-family sub-policy per episode from the active-user density regime: "
            "sparse/general/dense -> configurable (energy weight, gate) modes"
        ),
    )

    def __init__(self, env, config):
        super().__init__(env, config)
        self.regime_policy = adapt2_regime_policy_map()
        self._policy_modes = {
            regime: self._resolve_policy_modes(policy_name)
            for regime, policy_name in self.regime_policy.items()
        }
        self.current_regime = None
        self.current_policy = None
        self._apply_regime(self.active_users)

    @staticmethod
    def _resolve_policy_modes(policy_name: str) -> tuple[str, str]:
        normalized = normalize_algorithm_name(policy_name)
        adapter_class = ALGORITHM_REGISTRY.get(normalized)
        if adapter_class is None:
            raise ValueError(f"qeco-adapt2 sub-policy is not registered: {policy_name}")
        if adapter_class is QecoAdapt2Adapter or issubclass(adapter_class, QecoAdapt2Adapter):
            raise ValueError("qeco-adapt2 cannot use itself as a sub-policy")
        if adapter_class is QecoAdapter:
            return ("qeco", "none")
        if not issubclass(adapter_class, QecoAdaptAblationBase):
            raise ValueError(
                f"qeco-adapt2 sub-policy must share the QECO D3QN/LSTM backbone; got {policy_name}"
            )
        return (adapter_class.energy_weight_mode, adapter_class.gate_mode)

    def _apply_regime(self, active_users: int) -> None:
        regime = classify_density_regime(active_users)
        energy_weight_mode, gate_mode = self._policy_modes[regime]
        self.current_regime = regime
        self.current_policy = self.regime_policy[regime]
        self.energy_weight_mode = energy_weight_mode
        self.gate_mode = gate_mode

    def on_episode_start(self, episode: int, active_users: int) -> None:
        super().on_episode_start(episode, active_users)
        self._apply_regime(active_users)

    def episode_selection(self) -> dict | None:
        return {
            "episode": self.episode_index,
            "active_users": self.active_users,
            "users_per_edge": users_per_edge(self.active_users),
            "regime": self.current_regime,
            "policy": self.current_policy,
            "energy_weight_mode": self.energy_weight_mode,
            "gate_mode": self.gate_mode,
            "energy_weight": self._energy_weight(),
            "gating_strength": self._gate_strength(),
        }

    def metadata(self) -> dict:
        payload = super().metadata()
        payload["qeco_adapt2"] = {
            "density_metric": "active_users / num_edges",
            "regime_thresholds": {
                "sparse_max_users_per_edge": SharedExperiment.ADAPT2_SPARSE_MAX_USERS_PER_EDGE,
                "dense_min_users_per_edge": SharedExperiment.ADAPT2_DENSE_MIN_USERS_PER_EDGE,
            },
            "regime_policy": dict(self.regime_policy),
            "regime_modes": {
                regime: {"energy_weight_mode": modes[0], "gate_mode": modes[1]}
                for regime, modes in self._policy_modes.items()
            },
            "regimes": list(DENSITY_REGIMES),
            "initial_selection": self.episode_selection(),
        }
        return payload


ALGORITHM_REGISTRY = {
    TangWongAdapter.info.name: TangWongAdapter,
    QecoAdapter.info.name: QecoAdapter,
    QecoRewardWeightOnlyAdapter.info.name: QecoRewardWeightOnlyAdapter,
    QecoGateOnlyAdapter.info.name: QecoGateOnlyAdapter,
    QecoFixedWeightFixedGateAdapter.info.name: QecoFixedWeightFixedGateAdapter,
    QecoAdaptiveWeightFixedGateAdapter.info.name: QecoAdaptiveWeightFixedGateAdapter,
    QecoFixedWeightAdaptiveGateAdapter.info.name: QecoFixedWeightAdaptiveGateAdapter,
    QecoAdaptAdapter.info.name: QecoAdaptAdapter,
    QecoAdaptFullAdapter.info.name: QecoAdaptFullAdapter,
    QecoAdaptGeneralAdapter.info.name: QecoAdaptGeneralAdapter,
    QecoAdaptiveGateAdapter.info.name: QecoAdaptiveGateAdapter,
    QecoAdapt2Adapter.info.name: QecoAdapt2Adapter,
}

ADAPT2_FAMILY_ALGORITHMS = (
    QecoAdapter.info.name,
    QecoAdaptGeneralAdapter.info.name,
    QecoAdaptiveGateAdapter.info.name,
    QecoAdapt2Adapter.info.name,
)


def available_channel_algorithm_names() -> tuple[str, ...]:
    return tuple(ALGORITHM_REGISTRY.keys())


def channel_algorithm_catalog() -> dict[str, dict[str, str]]:
    return {
        name: {
            "name": adapter_class.info.name,
            "label": adapter_class.info.label,
            "paper_family": adapter_class.info.paper_family,
            "source_type": adapter_class.info.source_type,
            "implementation_module": f"{adapter_class.__module__}.{adapter_class.__name__}",
            "implementation_note": adapter_class.info.implementation_note,
        }
        for name, adapter_class in ALGORITHM_REGISTRY.items()
    }


def get_channel_algorithm(algorithm_name: str, env, config) -> ChannelAlgorithmAdapter:
    normalized = normalize_algorithm_name(algorithm_name)
    try:
        adapter_class = ALGORITHM_REGISTRY[normalized]
    except KeyError as exc:
        raise ValueError(f"Unknown channel-aware algorithm: {algorithm_name}") from exc
    return adapter_class(env, config)
