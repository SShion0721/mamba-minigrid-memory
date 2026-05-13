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


class SpinningPenaltyWrapper(gym.Wrapper):
    """Penalize the agent if it stays in the same position for too long.

    This helps discourage 'spinning in place' or getting stuck.
    """

    def __init__(self, env: gym.Env, max_steps: int = 10, penalty: float = 0.01):
        super().__init__(env)
        self.max_steps = max_steps
        self.penalty = penalty
        self.last_pos = None
        self.steps_in_pos = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Access the underlying agent position from MiniGrid
        unwrapped = self.env.unwrapped
        self.last_pos = getattr(unwrapped, "agent_pos", None)
        if self.last_pos is not None:
            self.last_pos = tuple(self.last_pos)
        self.steps_in_pos = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        unwrapped = self.env.unwrapped
        current_pos = getattr(unwrapped, "agent_pos", None)

        if current_pos is not None:
            current_pos = tuple(current_pos)
            if current_pos == self.last_pos:
                self.steps_in_pos += 1
            else:
                self.steps_in_pos = 0
                self.last_pos = current_pos

        if self.steps_in_pos > self.max_steps:
            reward -= self.penalty

        return obs, reward, terminated, truncated, info


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


def make_env(
    env_id: str,
    seed: int | None = None,
    render_mode: str | None = None,
    spinning_penalty: float = 0.0,
    spinning_threshold: int = 10,
):
    """Create a MiniGrid env with compact image, direction, and memory inputs."""

    env = gym.make(env_id, render_mode=render_mode)
    if spinning_penalty > 0:
        env = SpinningPenaltyWrapper(env, max_steps=spinning_threshold, penalty=spinning_penalty)
    env = MiniGridMemoryObsWrapper(env)

    if seed is not None:
        env.action_space.seed(seed)
        env.observation_space.seed(seed)

    return env
