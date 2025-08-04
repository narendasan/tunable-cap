from typing import List, Dict
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import ale_py
import gymnasium
import csv
import numpy as np

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

import numpy as np

def play_game(frames):
    # Assuming the ball is represented by a fully white pixel
    ball_positions = []
    for frame in frames:
        ball_positions.append(np.where(frame == 255))

    # Check if the ball is moving left or right
    if len(ball_positions) > 1:
        last_ball_pos = ball_positions[-1]
        prev_ball_pos = ball_positions[-2]

        if last_ball_pos[1] < prev_ball_pos[1]:
            return 'LEFT'
        elif last_ball_pos[1] > prev_ball_pos[1]:
            return 'RIGHT'

    # If no movement detected, do nothing
    return 'NOOP'

action_space = load_action_space_dict("./Breakout-v5-action_space.csv")

env_name = "ALE/Breakout-v5"
env = gymnasium.make(env_name, render_mode="rgb_array", obs_type="rgb")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda ep: ep % 1 == 0,
    video_folder="policy_rollouts_" + env_name.split("/")[1],
    name_prefix="test_policy"
)

for e in range(10):
    obs, info = env.reset()
    done = False
    step = 0

    while not done:
        obs_list = []
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(int(action_space["NOOP"]["index"]))
            done = terminated or truncated
            if done:
                break
            obs_list.append(obs)
            step += 1

        action = int(action_space[play_game(obs_list)]["index"])
        #action = env.action_space.sample()
        print(f"Step {step}:, Action: {action}")
        #print(int(action["index"]))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    print(f"Episode {e} completed.")

env.close()
