#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent

for path in (str(ROOT_DIR),):
    if path not in sys.path:
        sys.path.insert(0, path)

from qeco_runtime.Config import Config
from channel_env import ChannelAwareMEC
from algorithm_adapters import available_channel_algorithm_names, get_channel_algorithm, normalize_algorithm_name
from experiment_support import (
    DensitySchedule,
    SharedExperiment,
    apply_active_user_mask,
    apply_global_seed,
    build_task_arrivals,
    compute_env_metrics,
    create_run_dir,
    save_common_outputs,
    write_experiment_snapshot,
    write_progress_checkpoint,
)


def channel_run_dir(algorithm_name: str) -> Path:
    safe_name = normalize_algorithm_name(algorithm_name).replace("-", "_")
    return create_run_dir(safe_name)


def installed_version(distribution: str) -> str | None:
    """Return a distribution version without importing the package runtime."""
    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def run_channel_common(
    algorithm_name: str,
    episodes: int,
    dry_run: bool = False,
    validate_runtime: bool = False,
) -> Path:
    algorithm_name = normalize_algorithm_name(algorithm_name)
    apply_global_seed(SharedExperiment.RANDOM_SEED, use_tensorflow=True)
    run_dir = channel_run_dir(algorithm_name)

    env = ChannelAwareMEC(Config.N_UE, Config.N_EDGE, Config.N_TIME, Config.N_COMPONENT, Config.MAX_DELAY)
    algorithm = get_channel_algorithm(algorithm_name, env, Config)
    density_schedule = DensitySchedule(episodes, agent_pool_users=Config.N_UE, seed=SharedExperiment.RANDOM_SEED)
    if episodes > 0:
        algorithm.on_episode_start(0, density_schedule.active_count(0))
    density_summary = density_schedule.as_dict()
    execution_kind = "dry_run" if dry_run else "runtime_validation" if validate_runtime else "experiment"
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "tensorflow": installed_version("tensorflow"),
        "matplotlib": installed_version("matplotlib"),
        "protobuf": installed_version("protobuf"),
    }
    write_experiment_snapshot(
        run_dir,
        extra={
            "algorithm": algorithm.name.upper(),
            "algorithm_adapter": algorithm.metadata(),
            "evaluation_mode": "channel_aware_common_env",
            "execution_kind": execution_kind,
            "runtime_versions": runtime_versions,
            "run_episodes": episodes,
            "experiment_family": SharedExperiment.EXPERIMENT_FAMILY or None,
            "density_mode": density_schedule.mode,
            "density_choices": list(density_schedule.choices),
            "density_subset": density_schedule.subset_mode,
            "path_model": "user -> AP cluster -> channel/resource -> edge server",
            "action_space": "0=local, 1..N_EDGE=edge offloading; AP/channel resource selected by internal access tree",
            "n_actions": env.n_actions,
            "n_features": env.n_features,
            "n_lstm_state": env.n_lstm_state,
            "n_channel": env.n_channel,
            "n_access_points": env.n_ap,
            "ap_cluster_size": env.ap_cluster_size,
            "ap_to_edge": env.ap_to_edge.tolist(),
            "state_layout": "base QECO features + one edge_access_score per edge",
            "access_allocation": "selected edge traverses user AP-cluster and channel/resource nodes by best normalized effective rate",
            "source_environment": "QECO-Adapt independent channel-aware runtime",
        },
    )
    if density_schedule.mode == "random":
        with open(run_dir / "density_schedule.json", "w") as f:
            json.dump(density_summary, f, indent=2)

    print("[CHANNEL COMMON]", flush=True)
    print(f"- algorithm: {algorithm_name}", flush=True)
    print(f"- experiment_family: {SharedExperiment.EXPERIMENT_FAMILY or '(legacy)'}", flush=True)
    print(f"- density_mode: {density_schedule.mode}", flush=True)
    if density_schedule.mode == "random":
        print(
            f"- density_choices: {list(density_schedule.choices)} "
            f"(agent pool {Config.N_UE}, subset={density_schedule.subset_mode})",
            flush=True,
        )
    print(f"- users: {Config.N_UE}", flush=True)
    print(f"- edges: {Config.N_EDGE}", flush=True)
    print(f"- access_points: {env.n_ap}", flush=True)
    print(f"- ap_cluster_size: {env.ap_cluster_size}", flush=True)
    print(f"- channels: {env.n_channel}", flush=True)
    print(f"- actions: {env.n_actions}", flush=True)
    print(f"- features: {env.n_features}", flush=True)
    print(f"- lstm_state: {env.n_lstm_state}", flush=True)
    print(f"- episodes: {episodes}", flush=True)
    print(f"- run_dir: {run_dir}", flush=True)
    if dry_run:
        return run_dir

    agents = algorithm.build_agents()
    if validate_runtime:
        print(f"[RUNTIME VALIDATION] {algorithm.metadata()['agent_execution']}", flush=True)
        algorithm.close_agents()
        return run_dir

    episode_metrics = {
        "qoe": [],
        "delay": [],
        "energy": [],
        "drop": [],
        "completion_rate": [],
        "drop_rate": [],
        "completed_task": [],
        "arrived_task": [],
        "active_users": [],
    }
    episode_selections = []
    rl_step = 0

    for episode in range(episodes):
        active_users = density_schedule.active_count(episode)
        algorithm.on_episode_start(episode, active_users)
        selection = algorithm.episode_selection()
        if selection is not None:
            episode_selections.append(selection)
        if episode % 20 == 0:
            regime_note = f" regime={selection['regime']} policy={selection['policy']}" if selection else ""
            print(
                f"[{algorithm_name.upper()} channel] starting episode {episode + 1}/{episodes} "
                f"active_users={active_users}{regime_note}",
                flush=True,
            )
        bitarrive_size, bitarrive_dens = build_task_arrivals(env, Config, np)
        if density_schedule.mode == "random":
            bitarrive_size, bitarrive_dens = apply_active_user_mask(
                bitarrive_size, bitarrive_dens, density_schedule.active_mask(episode)
            )
        history = [
            [
                {
                    "observation": np.zeros(env.n_features),
                    "lstm": np.zeros(env.n_lstm_state),
                    "action": 0,
                    "observation_": np.zeros(env.n_features),
                    "lstm_": np.zeros(env.n_lstm_state),
                }
                for _ in range(env.n_ue)
            ]
            for _ in range(env.n_time)
        ]
        reward_indicator = np.zeros([env.n_time, env.n_ue])
        observation_all, lstm_state_all = env.reset(bitarrive_size, bitarrive_dens)
        done = False

        while not done:
            action_all = np.zeros([env.n_ue], dtype=int)
            for ue_index in range(env.n_ue):
                observation = np.squeeze(observation_all[ue_index, :])
                if observation[0] == 0:
                    action_all[ue_index] = 0
                else:
                    action_all[ue_index] = algorithm.select_action(agents[ue_index], observation)
                    agents[ue_index].do_store_action(episode, env.time_count, action_all[ue_index])

            observation_all_, lstm_state_all_, done = env.step(action_all)

            for ue_index in range(env.n_ue):
                agents[ue_index].update_lstm(lstm_state_all_[ue_index, :])
                history[env.time_count - 1][ue_index]["observation"] = observation_all[ue_index, :]
                history[env.time_count - 1][ue_index]["lstm"] = np.squeeze(lstm_state_all[ue_index, :])
                history[env.time_count - 1][ue_index]["action"] = action_all[ue_index]
                history[env.time_count - 1][ue_index]["observation_"] = observation_all_[ue_index]
                history[env.time_count - 1][ue_index]["lstm_"] = np.squeeze(lstm_state_all_[ue_index, :])

                update_index = np.where((1 - reward_indicator[:, ue_index]) * env.process_delay[:, ue_index] > 0)[0]
                for time_index in update_index:
                    reward = algorithm.reward(env, ue_index, time_index)
                    agents[ue_index].store_transition(
                        history[time_index][ue_index]["observation"],
                        history[time_index][ue_index]["lstm"],
                        history[time_index][ue_index]["action"],
                        reward,
                        history[time_index][ue_index]["observation_"],
                        history[time_index][ue_index]["lstm_"],
                    )
                    agents[ue_index].do_store_reward(episode, time_index, reward)
                    agents[ue_index].do_store_delay(episode, time_index, env.process_delay[time_index, ue_index])
                    agents[ue_index].do_store_energy(
                        episode,
                        time_index,
                        env.ue_comp_energy[time_index, ue_index],
                        env.ue_tran_energy[time_index, ue_index],
                        env.edge_comp_energy[time_index, ue_index],
                        env.ue_idle_energy[time_index, ue_index],
                    )
                    reward_indicator[time_index, ue_index] = 1

            rl_step += 1
            observation_all = observation_all_
            lstm_state_all = lstm_state_all_

            if (rl_step > 200) and (rl_step % 10 == 0):
                for agent in agents:
                    if hasattr(agent, "memory_counter") and agent.memory_counter > agent.n_lstm_step:
                        agent.learn()

        metrics = compute_env_metrics(env, np)
        metrics["active_users"] = float(active_users)
        for key in episode_metrics:
            episode_metrics[key].append(metrics[key])
        episodes_completed = episode + 1
        if episodes_completed % 20 == 0 or episodes_completed == episodes:
            write_progress_checkpoint(
                run_dir,
                algorithm.name,
                episodes_completed,
                episodes,
                metrics,
            )
            print(
                f"[{algorithm_name.upper()} channel] completed episode {episodes_completed}/{episodes} "
                f"qoe={metrics['qoe']:.4f} completion_rate={metrics['completion_rate']:.4f}",
                flush=True,
            )

    save_common_outputs(
        run_dir,
        episode_metrics,
        metadata={
            "algorithm": algorithm.name.upper(),
            "algorithm_adapter": algorithm.metadata(),
            "evaluation_mode": "channel_aware_common_env",
            "execution_kind": execution_kind,
            "runtime_versions": runtime_versions,
            "common_axis": "episode",
            "run_episodes": episodes,
            "comparison_points": SharedExperiment.COMPARISON_POINTS,
            "task_success_metrics": {
                "completion_rate": "completed_task / arrived_task",
                "drop_rate": "unfinished_task / arrived_task",
                "completed_task": "arrived_task - unfinished_task",
            },
            "n_channel": env.n_channel,
            "n_access_points": env.n_ap,
            "ap_cluster_size": env.ap_cluster_size,
            "path_model": "user -> AP cluster -> channel/resource -> edge server",
            "action_space": "local_or_edge_with_internal_access_tree",
            "state_layout": "base QECO features + one edge_access_score per edge",
            "experiment_family": SharedExperiment.EXPERIMENT_FAMILY or None,
            "density": {
                "mode": density_schedule.mode,
                "subset": density_schedule.subset_mode,
                "agent_pool_users": density_schedule.agent_pool_users,
                "choices": list(density_schedule.choices),
                "active_users_series": "ActiveUsers.txt",
                "regime_counts": {
                    regime: density_summary["regimes"].count(regime) for regime in ("sparse", "general", "dense")
                },
            },
        },
    )
    if episode_selections:
        with open(run_dir / "regime_selection.json", "w") as f:
            json.dump(
                {
                    "algorithm": algorithm.name,
                    "regime_policy": getattr(algorithm, "regime_policy", None),
                    "episodes": episode_selections,
                },
                f,
                indent=2,
            )
    algorithm.close_agents()
    return run_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run channel-aware QECO-Adapt extension experiments.")
    algorithm_choices = sorted(
        set(available_channel_algorithm_names())
        | {name.replace("-", "_") for name in available_channel_algorithm_names()}
    )
    parser.add_argument("algorithm", choices=algorithm_choices)
    parser.add_argument("--episodes", type=int, default=SharedExperiment.QECO_EPISODES)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--validate-runtime",
        action="store_true",
        help="Build all UE agents in the shared TensorFlow graph/session, report the runtime layout, then close it.",
    )
    args = parser.parse_args()

    run_dir = run_channel_common(
        args.algorithm,
        args.episodes,
        dry_run=args.dry_run,
        validate_runtime=args.validate_runtime,
    )
    print(f"Channel-aware run directory: {run_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
