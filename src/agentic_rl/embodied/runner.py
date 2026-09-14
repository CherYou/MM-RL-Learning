"""CPU training/evaluation on real FetchReach, with separate evaluation seeds."""

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import torch
import yaml

from ..data import ROOT
from ..logging import RunLogger
from .agents import Agent
from .replay import Replay, encode, file_hash, load_dataset, make_env, prepare_dataset, transition


def absolute(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def evaluation_seeds(config, episodes=None):
    first = config.get("eval_seed", 10000)
    return list(range(first, first + (episodes or config.get("eval_episodes", 10))))


def actor_hash(agent):
    digest = hashlib.sha256()
    for value in agent.actor.state_dict().values():
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


def evaluate(agent, config, *, episodes=None):
    env = make_env(config.get("reward_type", "sparse"))
    records = []
    try:
        for seed in evaluation_seeds(config, episodes):
            observation, _ = env.reset(seed=seed)
            total, ever_success = 0.0, False
            for step in range(env.spec.max_episode_steps):
                action = agent.act(encode(observation))
                observation, reward, terminated, truncated, info = env.step(action)
                total += float(reward)
                ever_success |= bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            records.append(
                {
                    "seed": seed,
                    "return": total,
                    "length": step + 1,
                    "success": ever_success,
                    "final_success": bool(info.get("is_success", False)),
                }
            )
    finally:
        env.close()
    metrics = {
        "eval/return": np.mean([row["return"] for row in records]),
        "eval/success_rate": np.mean([row["success"] for row in records]),
        "eval/final_success_rate": np.mean([row["final_success"] for row in records]),
        "eval/length": np.mean([row["length"] for row in records]),
    }
    return {key: float(value) for key, value in metrics.items()}, records


def validate_config(config):
    if config["algorithm"] not in {"sac", "td3", "her", "iql"}:
        raise ValueError("Expected sac, td3, her or iql")
    if config.get("backend") != "embodied" or config.get("device") != "cpu":
        raise ValueError("These teaching examples use backend=embodied, device=cpu")
    for key in ("steps", "batch_size", "hidden_dim", "replay_capacity", "eval_episodes", "policy_delay"):
        if config.get(key, 1) < 1:
            raise ValueError(f"{key} must be positive")
    if not 0 < config.get("expectile", 0.7) < 1:
        raise ValueError("expectile must lie strictly between zero and one")
    if config.get("max_weight", 100) <= 0 or config.get("alpha", 0.2) <= 0:
        raise ValueError("max_weight and alpha must be positive")
    if not 0 <= config.get("discount", 0.98) <= 1 or not 0 < config.get("tau", 0.005) <= 1:
        raise ValueError("discount/tau outside valid bounds")
    if config.get("reward_type", "sparse") not in {"sparse", "dense"}:
        raise ValueError("Unsupported reward type")
    if config["algorithm"] != "iql":
        # At most one reset per step, making this a conservative disjointness check.
        train_range = range(config["seed"], config["seed"] + config["steps"] + 1)
        if any(seed in train_range for seed in evaluation_seeds(config)):
            raise ValueError("Training and evaluation seed ranges may overlap")


def train(config):
    validate_config(config)
    torch.set_num_threads(config.get("cpu_threads", 2))
    torch.manual_seed(config["seed"])
    rng = np.random.default_rng(config["seed"])
    replay = Replay(config["replay_capacity"], config["seed"] + 1)
    dataset_metadata = None
    if config["algorithm"] == "iql":
        rows, dataset_metadata = load_dataset(
            absolute(config["dataset"]), config["reward_type"], evaluation_seeds(config)
        )
        if len(rows) > config["replay_capacity"]:
            raise ValueError("Offline replay capacity must fit the entire fixed dataset")
        replay.rows.extend(rows)
    path = absolute(config["output"])
    logger = RunLogger(path, config)
    agent = Agent(config)
    initial_hash = actor_hash(agent)
    initial_vector = torch.nn.utils.parameters_to_vector(agent.actor.parameters()).detach().clone()
    env = None
    try:
        baseline, baseline_records = evaluate(agent, config)
        logger.log(0, baseline)
        logger.trajectories(0, baseline_records)
        if config["algorithm"] != "iql":
            env = make_env(config["reward_type"])
            observation, _ = env.reset(seed=config["seed"])
        episode, episode_rows, episode_return = 0, [], 0.0
        environment_steps, interval_metrics = 0, {}
        for step in range(1, config["steps"] + 1):
            if env is not None:
                if step <= config.get("warmup_steps", 1000):
                    action = rng.uniform(-1, 1, 4).astype(np.float32)
                else:
                    action = agent.act(encode(observation), explore=True, rng=rng)
                following, reward, terminated, truncated, info = env.step(action)
                episode_rows.append(
                    transition(
                        observation,
                        action,
                        reward,
                        following,
                        terminated,
                        truncated,
                        episode,
                        len(episode_rows),
                    )
                )
                environment_steps += 1
                episode_return += float(reward)
                observation = following
                # Flush a partial final episode too; all hindsight goals are already observed.
                if terminated or truncated or step == config["steps"]:
                    replay.add_episode(
                        episode_rows,
                        reward_function=env.unwrapped.compute_reward,
                        her_k=config.get("her_k", 4) if config["algorithm"] == "her" else 0,
                    )
                    logger.trajectories(
                        step,
                        [
                            {
                                "episode": episode,
                                "return": episode_return,
                                "length": len(episode_rows),
                                "final_success": bool(info["is_success"]),
                                "partial": not (terminated or truncated),
                            }
                        ],
                    )
                    episode += 1
                    episode_rows, episode_return = [], 0.0
                    if step < config["steps"]:
                        observation, _ = env.reset(seed=config["seed"] + episode)
            ready = len(replay) >= config["batch_size"]
            if ready and (env is None or step >= config.get("warmup_steps", 1000)):
                recent = agent.update(replay.sample(config["batch_size"]))
                for key, value in recent.items():
                    interval_metrics.setdefault(key, []).append(value)
            if step % config.get("log_every", 50) == 0 or step == config["steps"]:
                logger.log(
                    step,
                    {
                        **{key: float(np.mean(values)) for key, values in interval_metrics.items()},
                        "train/replay_size": len(replay),
                        "train/environment_steps": environment_steps,
                        "train/updates": agent.updates,
                    },
                )
                interval_metrics.clear()
        if not agent.updates:
            raise RuntimeError("No optimization occurred; increase steps or reduce batch/warmup size")
        final, final_records = evaluate(agent, config)
        logger.log(config["steps"], final)
        logger.trajectories(config["steps"], final_records)
        final_vector = torch.nn.utils.parameters_to_vector(agent.actor.parameters()).detach()
        change = float(torch.linalg.vector_norm(final_vector - initial_vector))
        if not torch.isfinite(final_vector).all() or change == 0:
            raise RuntimeError("Actor did not change or contains nonfinite parameters")
        checkpoint = path / "checkpoint-final"
        checkpoint.mkdir()
        torch.save(
            {**agent.state_dict(), "torch_rng": torch.get_rng_state(), "numpy_rng": rng.bit_generator.state},
            checkpoint / "agent.pt",
        )
        # New object, reading disk: ensure evaluation uses exactly the saved actor.
        loaded = Agent(config)
        loaded.load_state_dict(torch.load(checkpoint / "agent.pt", map_location="cpu", weights_only=True))
        reloaded, reloaded_records = evaluate(loaded, config)
        if actor_hash(loaded) != actor_hash(agent) or reloaded_records != final_records:
            raise RuntimeError("Checkpoint reload failed deterministic evaluation parity")
        result = {
            "algorithm": config["algorithm"],
            "environment": "FetchReach-v4",
            "reward_type": config["reward_type"],
            "device": "cpu",
            "steps": config["steps"],
            "updates": agent.updates,
            "training_environment_steps": environment_steps,
            "initial_actor_sha256": initial_hash,
            "final_actor_sha256": actor_hash(agent),
            "actor_change_l2": change,
            "checkpoint_sha256": file_hash(checkpoint / "agent.pt"),
            "reload_exact": True,
            "baseline": baseline,
            "final": final,
            "reloaded": reloaded,
            "eval_episodes": final_records,
            "dataset": dataset_metadata,
            "replay_transitions": len(replay),
            "her_transitions": sum(bool(row["relabeled"]) for row in replay.rows),
            "versions": {
                key: importlib.metadata.version(key)
                for key in ("torch", "gymnasium", "gymnasium-robotics", "mujoco")
            },
            "scope": "Functional real-physics validation; short runs do not establish benchmark learning performance",
        }
        (path / "validation.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )
        return path
    finally:
        if env is not None:
            env.close()
        logger.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--output", default="data/embodied/fetch-reach/train.npz")
    prepare.add_argument("--episodes", type=int, default=40)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--reward-type", choices=("sparse", "dense"), default="sparse")
    training = sub.add_parser("train")
    training.add_argument("config")
    training.add_argument("--smoke", action="store_true")
    training.add_argument("--steps", type=int)
    training.add_argument("--output")
    training.add_argument("--dataset")
    evaluation = sub.add_parser("eval")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--episodes", type=int, default=10)
    evaluation.add_argument("--seed", type=int, default=20000)
    evaluation.add_argument("--output")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        print(
            json.dumps(
                prepare_dataset(
                    absolute(args.output),
                    episodes=args.episodes,
                    seed=args.seed,
                    reward_type=args.reward_type,
                ),
                indent=2,
            )
        )
    elif args.command == "train":
        config = yaml.safe_load(absolute(args.config).read_text())
        if args.smoke:
            config.update(steps=300, warmup_steps=50, batch_size=32, hidden_dim=64, eval_episodes=3)
        for key in ("steps", "output", "dataset"):
            if getattr(args, key) is not None:
                config[key] = getattr(args, key)
        config.setdefault("output", f"runs/embodied-{config['algorithm']}-{time.time_ns()}")
        print(f"Saved run: {train(config)}")
    else:
        if args.episodes < 1:
            raise ValueError("episodes must be positive")
        checkpoint = absolute(args.checkpoint)
        state = torch.load(checkpoint / "agent.pt", map_location="cpu", weights_only=True)
        config = {**state["config"], "eval_seed": args.seed, "eval_episodes": args.episodes}
        validate_config(config)
        if config["algorithm"] == "iql":
            load_dataset(absolute(config["dataset"]), config["reward_type"], evaluation_seeds(config))
        torch.set_num_threads(config.get("cpu_threads", 2))
        agent = Agent(config)
        agent.load_state_dict(state)
        metrics, records = evaluate(agent, config)
        result = {"checkpoint": str(checkpoint), "metrics": metrics, "episodes": records}
        if args.output:
            output = absolute(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
