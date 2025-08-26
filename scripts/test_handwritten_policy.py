from typing import List, Dict, Tuple
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import ale_py
import gymnasium
import csv
import numpy as np
import functools

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

def crop_frame(frame: np.ndarray) -> np.ndarray:
    #vertical_crop_with_border = frame[17:-14]
    horizontal_crop = frame[:,8:-8]
    return horizontal_crop

def get_ball_position(frame: np.ndarray) -> np.ndarray:
    """
    Returns an estimate of the ball position for a given frame
    Args:
        frame (np.ndarray): The input frame
    Returns:
        np.ndarray: The estimated ball position as a 2D array
    """
    crop = frame[93:-21]
    row_indices, col_indices = np.nonzero(crop)
    return np.mean(np.column_stack((row_indices, col_indices)), axis=0)


def get_paddle_position(frame: np.ndarray) -> np.ndarray:
    """
    Returns an estimate of the paddle position for a give frame
    Args:
        frame (np.ndarray): The input frame
    Returns:
        np.ndarray: The estimated paddle position as a 2D array
    """
    paddle_rows = frame[-21:-16]
    row_indices, col_indices = np.nonzero(paddle_rows)
    paddle = np.mean(np.column_stack((row_indices, col_indices)), axis=0)
    return paddle

def frame_differences(frames: List[np.ndarray]) -> np.ndarray:
    """
    Overlays the changes between consecutive frames.
    """
    return functools.reduce(lambda a, b: a + (frames[0] != b), frames[1:], frames[0] != frames[1])

def get_optimal_action(frames):
    import matplotlib.pyplot as plt
    import functools
    frame_diff = frame_differences(frames)
    paddle_crop = frame_diff[-21:-16]
    plt.imsave(f"paddle_diff.png", paddle_crop, cmap='gray')
    # Get the ball and paddle positions
    frames = [crop_frame(f) for f in frames]
    ball_pos = [get_ball_position(f) for f in frames]

    paddle_pos = [get_paddle_position(f) for f in frames]
    print(paddle_pos)
    # 2. Unzip the coordinates into separate x and y lists
    x_coords, y_coords = zip(*paddle_pos)

    # 3. Create the scatter plot
    plt.scatter(x_coords, y_coords, color='blue', marker='o')

    # 4. Add labels and a title for clarity
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Scatter Plot of Coordinates")
    plt.grid(True) # Adds a grid for easier reading

    # 5. Save the plot to a file
    plt.savefig("scatter_plot.png")

    print("Plot saved as scatter_plot.png")

    breakpoint()

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
        action = int(action_space["FIRE"]["index"])
        env.step(action)
        for i in range(10):
            obs, reward, terminated, truncated, info = env.step(int(action_space["RIGHT"]["index"]))
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
