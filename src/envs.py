"""MiniGrid environment helpers."""

from __future__ import annotations

import gymnasium as gym
import minigrid  # noqa: F401 - registers MiniGrid env ids with Gymnasium
import numpy as np
from gymnasium import spaces


MEMORY_ENVS = [
    "MiniGrid-MemoryS11-v0",
    "MiniGrid-MemoryS13-v0",
    "MiniGrid-MemoryS13Random-v0",
    "MiniGrid-MemoryS17Random-v0",
]


class MiniGridMemoryObsWrapper(gym.Wrapper):
    """Keep compact image + direction and add previous-step side inputs.

    MiniGrid's raw observation is a dict with image, direction, and mission. For
    MemoryEnv the mission text is fixed, so the model receives the semantic grid
    and the agent direction, plus PPO recurrent side inputs.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.num_actions = env.action_space.n
        self.prev_action = np.zeros(self.num_actions, dtype=np.float32)
        self.prev_reward = np.float32(0.0)
        self.episode_start = np.float32(1.0)

        image_space = env.observation_space["image"]
        self.observation_space = spaces.Dict(
            {
                "obs": image_space,
                "direction": spaces.Box(0, 3, shape=(1,), dtype=np.int64),
                "prev_action": spaces.Box(0.0, 1.0, shape=(self.num_actions,), dtype=np.float32),
                "prev_reward": spaces.Box(-np.inf, np.inf, shape=(1,), dtype=np.float32),
                "episode_start": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            }
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_action.fill(0.0)
        self.prev_reward = np.float32(0.0)
        self.episode_start = np.float32(1.0)
        return self._augment_obs(obs), info

    def step(self, action):
        action_int = int(action)
        obs, reward, terminated, truncated, info = self.env.step(action_int)

        self.prev_action.fill(0.0)
        self.prev_action[action_int] = 1.0
        self.prev_reward = np.float32(reward)
        self.episode_start = np.float32(0.0)

        return self._augment_obs(obs), reward, terminated, truncated, info

    def _augment_obs(self, obs):
        return {
            "obs": obs["image"],
            "direction": np.array([obs["direction"]], dtype=np.int64),
            "prev_action": self.prev_action.copy(),
            "prev_reward": np.array([self.prev_reward], dtype=np.float32),
            "episode_start": np.array([self.episode_start], dtype=np.float32),
        }


def make_env(env_id: str, seed: int | None = None, render_mode: str | None = None):
    """Create a MiniGrid env with compact image, direction, and memory inputs."""

    env = gym.make(env_id, render_mode=render_mode)
    env = MiniGridMemoryObsWrapper(env)

    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env
