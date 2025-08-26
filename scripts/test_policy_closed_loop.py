from typing import List, Dict
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import ale_py
import gymnasium
import csv
import numpy as np
from breakout_utilities import *
from _tmp_nvidia_cosmos_reason1_7b_closed_loop import predict_next_action

gymnasium.register_envs(ale_py)

def load_action_space_dict(csv_path):
    """
    Reads an action space CSV file and loads it into a dictionary.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary where keys are indices and values are dictionaries containing action type and description.
    """
    try:
        with open(csv_path, mode='r', newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            action_space_dict = {row['action'] : {'index':row['index'], 'action': row['action'], 'description': row['description']} for row in reader}
        return action_space_dict
    except FileNotFoundError:
        print(f"Error: The file at {csv_path} does not exist.")
        return {}
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return {}

action_space = load_action_space_dict("./Breakout-v5-action_space.csv")

env_name = "ALE/Breakout-v5"
env = gymnasium.make(env_name, render_mode="rgb_array", obs_type="grayscale")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda ep: ep % 1 == 0,
    video_folder="closed_loop_policy_rollouts_" + env_name.split("/")[1],
    name_prefix="test_policy"
)

for e in range(10):
    obs, info = env.reset()
    done = False
    step = 0

    obs_list = []
    action = 0
    state = np.zeros(1)
    weights = np.random.randn(1,10)

    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(int(action_space["NOOP"]["index"]))
        done = terminated or truncated
        if done:
            break
        obs_list.append(obs)
        step += 1

    while not done:
        action, state = predict_next_action(obs, weights, state)
        print(f"Step {step}:, Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.pop(0)
        obs_list.append(obs)
        done = terminated or truncated
        step += 1

    print(f"Episode {e} completed.")

env.close()
