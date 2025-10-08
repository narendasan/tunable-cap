import random
from typing import Callable

import gymnasium as gym
import numpy as np
import torch


def evaluate(
    model_path: str,
    make_env: Callable,
    env_id: str,
    eval_episodes: int,
    run_name: str,
    policy: Callable,
    device: torch.device = torch.device("cpu"),
    epsilon: float = 0.05,
    capture_video: bool = True,
):
    envs = gym.vector.SyncVectorEnv([make_env(env_id, 0, 0, capture_video, run_name)])

    obs, _ = envs.reset()
    memories = [None] * envs.num_envs
    episodic_returns = []
    while len(episodic_returns) < eval_episodes:
        if random.random() < epsilon:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(envs.num_envs)]
            )
        else:
            actions_and_next_memories, q_values = zip(
                *[policy(o, memory=m) for o, m in zip(obs, memories)]
            )
            (actions, next_memories) = zip(*actions_and_next_memories)
            actions = actions[0]
        next_obs, _, _, _, infos = envs.step(actions)
        if "episode" in infos:
            print(
                f"eval_episode={len(episodic_returns)}, episodic_return={infos['episode']['r']}"
            )
            episodic_returns += [infos["episode"]["r"]]
        obs = next_obs

    return episodic_returns
