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

def get_ball_position(frames):
    print(frames[-1].shape)
    #frames = np.array([np.mean(frame, axis=-1) for frame in frames])
    print(frames[-1][-100:-1])
    print(frames[-1][-100:-1].shape)


def get_paddle_position(frame):
    # Implement logic to extract paddle position from the frame
    pass

def get_optimal_action(frames):
    # Get the ball and paddle positions
    ball_pos = get_ball_position(frames)
    paddle_pos = get_paddle_position(frames)

    if ball_pos is None or paddle_pos is None:
        return 'NOOP'

    # Calculate the relative positions
    ball_x, ball_y = ball_pos
    paddle_x, paddle_y = paddle_pos

    # Determine the action based on the relative positions
    if ball_x < paddle_x:
        return 'LEFT'
    elif ball_x > paddle_x:
        return 'RIGHT'
    else:
        return 'NOOP'

action_space = load_action_space_dict("./Breakout-v5-action_space.csv")

env_name = "ALE/Breakout-v5"
env = gymnasium.make(env_name, render_mode="rgb_array", obs_type="grayscale")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda ep: ep % 1 == 0,
    video_folder="policy_rollouts_" + env_name.split("/")[1],
    name_prefix="test_handwritten_policy"
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

        action = int(action_space[get_optimal_action(obs_list)]["index"])
        #action = env.action_space.sample()
        print(f"Step {step}:, Action: {action}")
        #print(int(action["index"]))
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step += 1

    print(f"Episode {e} completed.")

env.close()
