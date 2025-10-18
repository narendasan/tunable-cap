# test_policy_open_loop.py
from typing import List, Dict
from PIL import Image
import requests
from transformers import AutoProcessor, AutoModel
import torch
import ale_py
import gymnasium
import csv
import numpy as np
from generative_policy_proposals.controller_utils.breakout_utilities import *

from _tmp_omlab_vlm_r1_qwen2_5vl_3b_ovd_0321_open_loop import predict_next_action

# from generative_policy_proposals._ControllerGenerator import ControllerGenerator, generate_and_load_policy, regenerate_policy
from generative_policy_proposals._ControllerGeneratorOpenAI import ControllerGenerator, generate_and_load_policy, regenerate_policy
from generative_policy_proposals import action_spaces, generate_random_rollouts
from generative_policy_proposals.controller_utils.breakout_utilities import UTILITY_SPECS
from generative_policy_proposals.action_spaces._action_spaces import BREAKOUT_ACTION_SPACE


gymnasium.register_envs(ale_py)
env_name = "ALE/Breakout-v5"
env = gymnasium.make(env_name, render_mode="rgb_array", obs_type="grayscale")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda ep: ep % 1 == 0,
    video_folder="videos/open_loop_policy_rollouts_manual_2" + env_name.split("/")[1],
    # added _2 for testing
    name_prefix="test_policy"
)

action_space = BREAKOUT_ACTION_SPACE

for e in range(10):
    obs, info = env.reset()
    done = False
    step = 0

    obs_list = []
    for i in range(3):
        obs, reward, terminated, truncated, info = env.step(int(action_space["NOOP"]["index"]))
        done = terminated or truncated
        if done:
            break
        obs_list.append(obs)
        step += 1

    while not done:
        action = int(predict_next_action(obs_list))
        print(f"Step {step}:, Action: {action}")
        obs, reward, terminated, truncated, info = env.step(action)
        obs_list.pop(0)
        obs_list.append(obs)
        done = terminated or truncated
        step += 1

    print(f"Episode {e} completed.")

env.close()
