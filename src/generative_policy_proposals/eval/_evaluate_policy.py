from collections import namedtuple
from typing import List, Callable
import gymnasium as gym
from slugify import slugify

from generative_policy_proposals._ControllerGenerator import ControllerGenerator

Eval = namedtuple('Eval', ["actions", "score", "lives", "info"])

def evaluate_policy(
    actor: ControllerGenerator,
    env_name: str,
    policy: Callable,
    num_episodes: int = 10,
    accumulate_obs: bool = False,
    num_accumulated_frames: int = 3,
    noop_action: int = 0,
    exp_name: str = ""
) -> List[Eval]:
    env = gym.make(env_name, render_mode="rgb_array", obs_type="grayscale")
    env = gym.wrappers.RecordVideo(
        env,
        episode_trigger=lambda ep: ep % 1 == 0,
        video_folder=f"videos/{exp_name}policy_rollouts_" + env_name.split("/")[1],
        name_prefix=f"eval_{slugify(actor.model_name).replace('-', '_')}_policy"
    )

    evals = []
    for e in range(num_episodes):
        obs, info = env.reset()
        done = False
        step = 0
        score = 0
        action_history = []
        lives = 5
        obs_list = []

        if accumulate_obs:
            for _ in range(num_accumulated_frames):
                obs, reward, terminated, truncated, info = env.step(noop_action)
                done = terminated or truncated
                if done:
                    break
                obs_list.append(obs)
                step += 1

        while not done:
            action = int(policy(obs_list if accumulate_obs else obs_list[-1]))
            print(f"Step {step}:, Action: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            obs_list.pop(0)
            obs_list.append(obs)
            done = terminated or truncated
            score += reward
            lives = info["lives"]
            print(f"Step {step}:, Action: {action}, Score: {score}, Reward: {reward}, info: {info}")
            step += 1

        evals.append(Eval(action_history, score, lives, info))
        print(f"Episode {e} completed.")


    env.close()
    return evals
