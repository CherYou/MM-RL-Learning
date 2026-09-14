"""Episode-aware replay and locally collected FetchReach demonstration data."""

from collections import deque
from copy import deepcopy
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
import torch


FIELDS = ("observation", "achieved_goal", "desired_goal")


def make_env(reward_type="sparse"):
    import gymnasium as gym
    import gymnasium_robotics

    gym.register_envs(gymnasium_robotics)
    return gym.make("FetchReach-v4", reward_type=reward_type)


def encode(observation):
    return np.concatenate((observation["observation"], observation["desired_goal"])).astype(np.float32)


def transition(observation, action, reward, following, terminated, truncated, episode, time):
    return {
        **{key: np.array(observation[key], dtype=np.float32, copy=True) for key in FIELDS},
        **{"next_" + key: np.array(following[key], dtype=np.float32, copy=True) for key in FIELDS},
        "action": np.array(action, dtype=np.float32, copy=True),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "episode": int(episode),
        "time": int(time),
        "relabeled": False,
        "future_time": -1,
    }


def relabel_future(episode, reward_function, rng, goals_per_transition=4):
    """Add hindsight copies using next achieved goals from this episode only.

    The future strategy includes the next state of the selected transition.
    FetchReach termination does not depend on goal success. For other tasks a
    goal-dependent terminal condition must also be recomputed before reuse.
    """
    if not episode:
        return []
    if len({row["episode"] for row in episode}) != 1:
        raise ValueError("HER must never mix episodes")
    if any(row["terminated"] or row["truncated"] for row in episode[:-1]):
        raise ValueError("Episode contains an interior boundary")
    if any(b["time"] != a["time"] + 1 for a, b in zip(episode, episode[1:])):
        raise ValueError("Episode transitions must be contiguous and ordered")
    augmented = []
    for index, row in enumerate(episode):
        for _ in range(goals_per_transition):
            future = episode[int(rng.integers(index, len(episode)))]
            goal = future["next_achieved_goal"]
            copy = deepcopy(row)
            copy["desired_goal"] = goal.copy()
            copy["next_desired_goal"] = goal.copy()
            copy["reward"] = float(reward_function(copy["next_achieved_goal"], goal, {}))
            copy["relabeled"] = True
            copy["future_time"] = future["time"] + 1
            augmented.append(copy)
    return augmented


class Replay:
    def __init__(self, capacity, seed):
        self.rows = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.rows)

    def add_episode(self, episode, *, reward_function=None, her_k=0):
        self.rows.extend(episode)
        if her_k:
            self.rows.extend(relabel_future(episode, reward_function, self.rng, her_k))

    def sample(self, size):
        if not self.rows:
            raise ValueError("Cannot sample an empty replay buffer")
        chosen = [self.rows[int(i)] for i in self.rng.integers(0, len(self), size=size)]
        states = np.stack([encode(row) for row in chosen])
        following = np.stack([encode({key: row["next_" + key] for key in FIELDS}) for row in chosen])
        arrays = (
            states,
            np.stack([row["action"] for row in chosen]),
            np.array([row["reward"] for row in chosen]),
            following,
            np.array([row["terminated"] for row in chosen]),
        )
        return tuple(torch.as_tensor(array, dtype=torch.float32) for array in arrays)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_dataset(path, *, episodes=40, seed=42, reward_type="sparse"):
    """Actual MuJoCo rollouts from a noisy proportional controller and random policy.

    This creates programmatic demonstrations, not human data or a benchmark copy.
    """
    path = Path(path)
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        raise FileExistsError(f"Choose a new dataset path; already exists: {path}")
    if episodes < 1:
        raise ValueError("episodes must be positive")
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    env = make_env(reward_type)
    rows, successes = [], []
    try:
        for episode in range(episodes):
            observation, _ = env.reset(seed=seed + episode)
            scripted = episode % 4 != 0
            success = False
            for time in range(env.spec.max_episode_steps):
                if scripted:
                    movement = 8 * (observation["desired_goal"] - observation["achieved_goal"])
                    action = np.r_[movement, 0] + rng.normal(0, 0.1, 4)
                else:
                    action = rng.uniform(-1, 1, 4)
                action = np.clip(action, -1, 1).astype(np.float32)
                following, reward, terminated, truncated, info = env.step(action)
                rows.append(
                    transition(observation, action, reward, following, terminated, truncated, episode, time)
                )
                observation = following
                success |= bool(info.get("is_success", False))
                if terminated or truncated:
                    break
            successes.append(success)
    finally:
        env.close()
    arrays = {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
    np.savez_compressed(path, **arrays)
    metadata = {
        "format": 1,
        "environment": "FetchReach-v4",
        "reward_type": reward_type,
        "collector": "75% noisy proportional controller; 25% uniform random, by episode",
        "origin": "Locally generated real MuJoCo physics transitions; no human demonstrations",
        "episodes": episodes,
        "transitions": len(rows),
        "episode_seeds": list(range(seed, seed + episodes)),
        "episode_success_rate": float(np.mean(successes)),
        "sha256": file_hash(path),
        "versions": {
            key: importlib.metadata.version(key)
            for key in ("numpy", "gymnasium", "gymnasium-robotics", "mujoco")
        },
        "fields": {
            key: {"shape": list(array.shape), "dtype": str(array.dtype)} for key, array in arrays.items()
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    return metadata


def load_dataset(path, reward_type, eval_seeds):
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata["sha256"] != file_hash(path):
        raise ValueError("Dataset checksum mismatch")
    if metadata["environment"] != "FetchReach-v4" or metadata["reward_type"] != reward_type:
        raise ValueError("Dataset environment/reward does not match the training configuration")
    if set(metadata["episode_seeds"]) & set(eval_seeds):
        raise ValueError("Dataset collection and evaluation seeds overlap")
    with np.load(path, allow_pickle=False) as data:
        arrays = {key: data[key] for key in data.files}
    count = metadata["transitions"]
    required = {
        *FIELDS,
        *("next_" + key for key in FIELDS),
        "action",
        "reward",
        "terminated",
        "truncated",
        "episode",
        "time",
        "relabeled",
        "future_time",
    }
    if set(arrays) != required or any(len(array) != count for array in arrays.values()):
        raise ValueError("Malformed dataset fields or lengths")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Dataset contains nonfinite entries")
    if arrays["action"].shape != (count, 4) or (np.abs(arrays["action"]) > 1).any():
        raise ValueError("FetchReach expects four normalized action coordinates")
    return [{key: array[index] for key, array in arrays.items()} for index in range(count)], metadata
