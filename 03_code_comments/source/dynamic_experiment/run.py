"""CPU-only, paired dynamic-population pilot with frozen specialist evaluation."""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np

from .config import DynamicConfig
from .selector import DwellSelector, calibrate, context_bin
from .provenance import source_identity

TRAIN_POLICIES = ("fixed", "adaptive", "contextual")
EVAL_ARMS = (*TRAIN_POLICIES, "switching")


def atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def history_array(histories):
    return np.asarray([list(h) for h in histories], dtype=np.float32)


def append_context(observations, recurrent, context):
    repeated = np.tile(context, (len(observations), 1))
    return np.concatenate([observations, repeated], axis=1), np.concatenate([recurrent, repeated], axis=1)


def training_reward(env, ue, origin_step, weight):
    from experiment_support import qeco_adapt_reward_function
    return float(qeco_adapt_reward_function(
        env.process_delay[origin_step, ue], env.max_delay,
        env.unfinish_task[origin_step, ue], env.ue_energy_state[ue],
        env.ue_comp_energy[origin_step, ue], env.ue_tran_energy[origin_step, ue],
        env.edge_comp_energy[origin_step, ue], env.ue_idle_energy[origin_step, ue], weight))


def common_task_qoe(env, ue, step):
    from experiment_support import qoe_function
    return float(qoe_function(env.process_delay[step, ue], env.max_delay,
        env.unfinish_task[step, ue], env.ue_energy_state[ue],
        env.ue_comp_energy[step, ue], env.ue_tran_energy[step, ue],
        env.edge_comp_energy[step, ue], env.ue_idle_energy[step, ue]))


def run_episode(config, bank, trace, arm, *, training=False, exploration=0,
                mapping=("fixed", "fixed", "fixed"), step_sink=None, step_prefix=None,
                diagnostic_policies=None):
    from algorithm_adapters.registry import QecoAdaptiveWeightFixedGateAdapter, QecoAdaptAdapter
    from experiment_support import compute_env_metrics, qeco_adapt_energy_weight
    from qeco_runtime.Config import Config
    from .environment import TraceMEC
    from .learner import Transition

    env = TraceMEC(trace, num_edges=3, max_delay=config.max_delay)
    observations, recurrent = env.reset_trace()
    context = env.context()
    observations, recurrent = append_context(observations, recurrent, context)
    histories = [deque([np.zeros(recurrent.shape[1], dtype=np.float32)
                        for _ in range(config.history_steps - 1)] + [row.copy()],
                       maxlen=config.history_steps) for row in recurrent]
    gates = {"fixed": QecoAdaptiveWeightFixedGateAdapter(env, Config),
             "adaptive": QecoAdaptAdapter(env, Config)}
    selector = DwellSelector(mapping, config.selector_dwell)
    pending = {}
    task_records = []
    action_seconds, context_selector_seconds = 0.0, 0.0
    decisions, switches, gates_rejected = 0, 0, 0
    previous_policy = None
    saturation = np.zeros(3, dtype=int)
    delayed_across_switch = 0
    for step in range(config.total_steps):
        context_started = time.perf_counter()
        context = env.context()
        saturation += np.asarray(context) >= 1 - 1e-12
        chosen = selector.select(context, step) if arm == "switching" else arm
        if diagnostic_policies is not None:
            chosen = diagnostic_policies[step]
        switches += int(previous_policy is not None and previous_policy != chosen)
        previous_policy = chosen
        bin_id = context_bin(context)
        count = int(trace.active_mask[step].sum())
        for gate in gates.values():
            gate.on_episode_start(0, count)
        window = history_array(histories)
        actions = np.zeros(config.pool_users, dtype=int)
        proposal_count, rejected = 0, 0
        context_elapsed = time.perf_counter() - context_started
        context_selector_seconds += context_elapsed
        started = time.perf_counter()
        for ue in range(config.pool_users):
            if trace.task_sizes[step, ue] <= 0:
                continue
            proposed = bank.choose(chosen, ue, observations[ue], window[ue], exploration=exploration)
            actions[ue] = gates[chosen]._apply_access_gate(observations[ue], proposed) if chosen in gates else proposed
            proposal_count += 1
            rejected += int(actions[ue] != proposed)
        elapsed = time.perf_counter() - started
        action_seconds += elapsed
        decisions += proposal_count
        gates_rejected += rejected
        next_obs, next_recurrent, done = env.step(actions)
        next_context = env.context()
        next_obs, next_recurrent = append_context(next_obs, next_recurrent, next_context)
        for history, row in zip(histories, next_recurrent):
            history.append(row.copy())
        next_window = history_array(histories)
        weight = qeco_adapt_energy_weight(count)
        for ue in range(config.pool_users):
            transition = Transition(
                origin_expert=chosen, origin_energy_weight=weight,
                state=observations[ue].copy(), history_window=window[ue].copy(),
                action=int(actions[ue]), next_state=next_obs[ue].copy(),
                next_history_window=next_window[ue].copy(), origin_done=bool(done), context_bin=bin_id)
            if trace.task_sizes[step, ue] > 0:
                pending[(step, ue)] = (transition, switches)
            elif training:
                # Idle steps are real transitions too, including the terminal
                # drain step. Their sole executable action is local/no-op (0).
                bank.store(chosen, ue, transition, 0.0)
        completed_now = []
        for (origin, ue), (transition, origin_generation) in pending.items():
            if env.process_delay[origin, ue] <= 0:
                continue
            if training:
                bank.store(transition.origin_expert, ue, transition,
                           training_reward(env, ue, origin, transition.origin_energy_weight))
            common = common_task_qoe(env, ue, origin)
            task_records.append({"policy": arm, "context_bin": transition.context_bin,
                                 "qoe": common, "origin_step": origin,
                                 "completion_step": step, "origin_expert": transition.origin_expert})
            delayed_across_switch += int(origin_generation != switches)
            completed_now.append((origin, ue))
        for key in completed_now:
            del pending[key]
        if training and (step + 1) % config.learn_every == 0:
            bank.learn(chosen)
        if step_sink:
            step_sink.write(json.dumps({**(step_prefix or {}), "step": step,
                "active_users": count, "context": [float(v) for v in context],
                "context_bin": bin_id, "policy": chosen, "decisions": proposal_count,
                "gate_rejections": rejected, "action_seconds": elapsed,
                "context_selector_seconds": context_elapsed,
                "controller_seconds": context_elapsed + elapsed,
                "completed_now": len(completed_now)}, allow_nan=False) + "\n")
        observations = next_obs
        if done != (step == config.total_steps - 1):
            raise AssertionError("Environment episode length differs from contract")
    arrived = int(np.count_nonzero(trace.task_sizes))
    if pending or len(task_records) != arrived:
        raise AssertionError(f"Unclassified arrivals: pending={len(pending)}, classified={len(task_records)}, arrived={arrived}")
    metrics = compute_env_metrics(env, np)
    if not all(np.isfinite(v) for v in metrics.values()):
        raise AssertionError("Non-finite episode metrics")
    recomputed = float(np.mean([r["qoe"] for r in task_records])) if task_records else 0.0
    if not np.isclose(metrics["qoe"], recomputed, rtol=1e-12, atol=1e-12):
        raise AssertionError("Common QoE recomputation differs")
    metrics.update({"switch_count": switches, "gate_rejections": gates_rejected,
                    "decisions": decisions, "action_seconds": action_seconds,
                    "seconds_per_decision": action_seconds / max(decisions, 1),
                    "context_selector_seconds": context_selector_seconds,
                    "controller_seconds": context_selector_seconds + action_seconds,
                    "controller_seconds_per_step": (context_selector_seconds + action_seconds) / config.total_steps,
                    "controller_seconds_per_decision": (context_selector_seconds + action_seconds) / max(decisions, 1),
                    "delayed_across_switch": delayed_across_switch,
                    "context_saturated_steps": saturation.tolist()})
    return metrics, task_records


def execute(config, seed, output):
    identity = source_identity(Path(__file__).resolve().parents[1])
    expected_identity = os.environ.get("QECO_SOURCE_IDENTITY_HASH")
    if expected_identity is not None and expected_identity != identity.identity_hash:
        raise RuntimeError("Runner source identity differs from campaign source identity")
    # Set CPU and topology before importing the legacy TF/environment modules.
    os.environ.update(CUDA_VISIBLE_DEVICES="-1", QECO_AGENT_SESSION_MODE="per-agent",
        TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1",
        OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="1",
        QECO_NUM_USERS=str(config.pool_users), QECO_NUM_EDGES="3",
        QECO_NUM_ACCESS_POINTS="6", QECO_CHANNEL_COUNT="4", QECO_AP_CLUSTER_SIZE="3")
    import tensorflow.compat.v1 as tf
    from .environment import TraceMEC
    from .learner import SpecialistBank
    from .traces import make_trace
    from experiment_support import SharedExperiment

    if tf.config.list_physical_devices("GPU"):
        raise RuntimeError("CPU-only experiment unexpectedly has visible GPUs")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    manifest = {"seed": seed, "revision": identity.revision, "config_hash": config.digest(),
                "source_identity": identity.payload(), "source_identity_hash": identity.identity_hash,
                "config": config.payload(), "status": "running", "protocol": "dynamic-specialists-v1",
                "tensorflow": tf.__version__, "numpy": np.__version__,
                "cpu_only": True, "evaluation_updates": False,
                "resolved_environment": {name: getattr(SharedExperiment, name)
                    for name in dir(SharedExperiment) if name.isupper()
                    and isinstance(getattr(SharedExperiment, name), (int, float, str, tuple, list, bool))}}
    atomic_json(output / "manifest.json", manifest)
    first_trace = make_trace(config, seed, "train", 0)
    shape_env = TraceMEC(first_trace, num_edges=3, max_delay=config.max_delay)
    print(f"[DYNAMIC] building 3 x {config.pool_users} independent UE learners on CPU", flush=True)
    bank = SpecialistBank(config, seed, shape_env.n_actions, shape_env.n_features + 3, shape_env.n_lstm_state + 3)
    print("[DYNAMIC] learner construction complete", flush=True)
    initial = bank.snapshot()
    rows, calibration_records = [], []
    frozen = None
    try:
        with (output / "episodes.jsonl").open("w") as episodes_file, (output / "steps.jsonl").open("w") as steps_file:
            def episode(phase, index, arm, mapping=("fixed",) * 3):
                trace = make_trace(config, seed, phase, index)
                fraction = index / max(config.train_episodes - 1, 1)
                exploration = config.exploration_start + fraction * (config.exploration_end - config.exploration_start) if phase == "train" else 0
                before = time.monotonic()
                metrics, tasks = run_episode(config, bank, trace, arm,
                    training=phase == "train", exploration=exploration, mapping=mapping,
                    step_sink=steps_file, step_prefix={"phase": phase, "episode": index, "arm": arm})
                row = {"phase": phase, "episode": index, "arm": arm,
                       "trace_hash": trace.digest(), "wall_seconds": time.monotonic() - before, **metrics}
                rows.append(row)
                episodes_file.write(json.dumps(row, allow_nan=False) + "\n")
                episodes_file.flush()
                steps_file.flush()
                atomic_json(output / "progress.json", {"phase": phase, "episode": index + 1, "arm": arm,
                    "completed_rollouts": len(rows), "elapsed_seconds": time.monotonic() - started})
                print(f"[DYNAMIC] {phase} {arm} episode={index + 1} qoe={metrics['qoe']:.4f} completion={metrics['completion_rate']:.4f} switches={metrics['switch_count']}", flush=True)
                return tasks

            for arm in TRAIN_POLICIES:
                for index in range(config.train_episodes):
                    episode("train", index, arm)
            frozen = bank.snapshot()
            for policy in TRAIN_POLICIES:
                if (frozen["policies"][policy]["update_count"] <= 0
                        or frozen["policies"][policy]["trainable_weight_digest"] == initial["policies"][policy]["trainable_weight_digest"]):
                    raise AssertionError(f"No measured learning update for {policy}")
            atomic_json(output / "trained_state.json", {"initial": initial, "trained": frozen})
            for arm in ("fixed", "adaptive"):
                for index in range(config.calibration_episodes):
                    calibration_records.extend(episode("calibration", index, arm))
            mapping, evidence = calibrate(calibration_records, config.calibration_min_tasks)
            atomic_json(output / "calibration.json", {"mapping": mapping, "bins": evidence,
                "degenerate_single_policy": len(set(mapping)) == 1, "source_phase": "calibration"})
            for arm in EVAL_ARMS:
                if bank.snapshot() != frozen:
                    raise AssertionError("Frozen learner state changed before evaluation arm")
                for index in range(config.eval_episodes):
                    episode("evaluation", index, arm, mapping)
                if bank.snapshot() != frozen:
                    raise AssertionError("Frozen evaluation mutated weights/replay/update state")
            diagnostic_evidence = None
            if config.diagnostic:
                # Separate diagnostic labels never replace calibrated pilot labels.
                diagnostic_evidence = runtime_diagnostics(config, bank, seed, frozen, mapping)
                atomic_json(output / "runtime_diagnostics.json", diagnostic_evidence)
        evaluation = [r for r in rows if r["phase"] == "evaluation"]
        for index in range(config.eval_episodes):
            if len({r["trace_hash"] for r in evaluation if r["episode"] == index}) != 1:
                raise AssertionError("Held-out exogenous traces differ across arms")
        summaries = {arm: {metric: float(np.mean([r[metric] for r in evaluation if r["arm"] == arm]))
            for metric in ("qoe", "delay", "energy", "completion_rate", "drop_rate", "drop", "switch_count", "seconds_per_decision", "controller_seconds_per_step", "controller_seconds_per_decision")}
            for arm in EVAL_ARMS}
        manifest.update(status="complete", elapsed_seconds=time.monotonic() - started,
            max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            metrics=summaries, learner_state=frozen, calibrated_mapping=mapping,
            completed_rollouts=len(rows), expected_rollouts=3 * config.train_episodes + 2 * config.calibration_episodes + 4 * config.eval_episodes,
            diagnostics=diagnostic_evidence,
            limitations=["Short functional pilot, not convergence or significance evidence",
                "Frozen evaluation tests selection, not continual parameter adaptation",
                "Bank has twice a single policy's deployed parameter count",
                "Inherited in-flight transmission capacity retention; no physical mobility claim"])
        if manifest["completed_rollouts"] != manifest["expected_rollouts"]:
            raise AssertionError("Incomplete rollouts")
        atomic_json(output / "summary.json", manifest)
        atomic_json(output / "manifest.json", manifest)
        checksums = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in output.iterdir() if p.is_file()}
        atomic_json(output / "checksums.json", checksums)
        print(f"[DYNAMIC] COMPLETE seed={seed} output={output}", flush=True)
        return manifest
    finally:
        bank.close()


def runtime_diagnostics(config, bank, seed, frozen, mapping):
    from .traces import make_trace
    trace = make_trace(config, seed, "diagnostic", 0)
    # Reversed arm order reuses the identical exogenous realization and weights.
    first = {}
    for arm in EVAL_ARMS:
        first[arm], _ = run_episode(config, bank, trace, arm, mapping=mapping)
    for arm in reversed(EVAL_ARMS):
        repeated, _ = run_episode(config, bank, trace, arm, mapping=mapping)
        for key in ("qoe", "delay", "energy", "completion_rate", "drop"):
            if repeated[key] != first[arm][key]:
                raise AssertionError(f"Arm order changed {arm}/{key}")
    # A scripted context stream proves the selector can choose both banks.
    selector = DwellSelector(("adaptive", "fixed", "adaptive"), 2)
    policies = [selector.select(c, t) for t, c in enumerate(
        [[0, 0, 0]] * 3 + [[0.7, 0.4, 0.4]] * 3 + [[0, 0, 0]] * 3)]
    if set(policies) != {"adaptive", "fixed"}:
        raise AssertionError("Scripted selector did not exercise both specialists")
    diagnostic_policies = tuple("adaptive" if (step // 2) % 2 == 0 else "fixed"
                                for step in range(config.total_steps))
    switched, _ = run_episode(config, bank, trace, "switching", diagnostic_policies=diagnostic_policies)
    if switched["switch_count"] == 0 or switched["delayed_across_switch"] == 0:
        raise AssertionError("Real bank diagnostic did not exercise switching across pending task rewards")
    if bank.snapshot() != frozen:
        raise AssertionError("Diagnostics mutated frozen state")
    return {"arm_order_invariant": True, "frozen_unchanged": True,
            "arm_order_checked": list(EVAL_ARMS), "calibrated_mapping": list(mapping),
            "scripted_selector_policies": policies, "diagnostic_only": True,
            "real_bank_switches": switched["switch_count"],
            "real_delayed_completions_after_switch": switched["delayed_across_switch"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be an unsigned 32-bit integer")
    execute(DynamicConfig.load(args.config), args.seed, args.output_dir)


if __name__ == "__main__":
    main()
