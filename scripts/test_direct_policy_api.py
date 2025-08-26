from typing import List, Dict, Any
from PIL import Image
import requests
import transformers
from transformers import AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration
import torch
import ale_py
import gymnasium
import csv
import numpy as np
from qwen_vl_utils import process_vision_info
import re
import random

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

import openai
import base64
import io
import cv2
import itertools

class GameActor:
    def __init__(
        self,
        system_prompt="You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>. You are tasked to generate code to play a game based on videos you watch",
        model="llava:13b", #"gpt-4o",
        temperature=0.6,
        max_tokens=4096,
        action_history_ring_buf_size: int = 25
    ):
        self.client = openai.OpenAI()
        #openai.OpenAI(
        #    base_url = 'http://localhost:11434/v1',
        #    api_key='ollama', # required, but unused
        #)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt
        self.message_history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        self.end_of_qa = 0
        self.action_history_len = action_history_ring_buf_size * 2

    def _encode_image(self, image_array: np.ndarray) -> str:
        """Convert numpy image array to base64 string"""
        image = Image.fromarray(image_array)
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"

    def _generate(self, messages: List[Dict[Any, Any]]) -> str:
        response = self.client.responses.create(
            model=self.model,
            reasoning={"effort": "high"},
            input=messages,
        )

        output_text = response.output_text

        self.message_history.append({
            "role": "assistant",
            "content": output_text
        })
        return output_text

    def question_answering(self, video_paths: List[str], images: List[np.ndarray], prompt: str):
        # Note: OpenAI API doesn't directly support video files, so we'll just use text and images
        # Videos would need to be processed into frames first

        videos = []
        for video_path in video_paths:
            video = cv2.VideoCapture(video_path)
            base64Frames = []
            while video.isOpened():
                success, frame = video.read()
                if not success:
                    break
                _, buffer = cv2.imencode(".jpg", frame)
                base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

            video.release()
            print(len(base64Frames), "frames read.")
            videos.append(base64Frames)

        video_frames = itertools.chain.from_iterable(
            [
                [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{frame}"
                    }
                    for frame in v[0::50]
                ] for v in videos
            ]
        )

        image_msgs = [{
            "type": "input_image",
            "image_url": self._encode_image(img)
        } for img in images]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            prompt
                        ),
                    }, *video_frames, *image_msgs
                ]
            }
        ]
        self.message_history += messages
        output_text = self._generate(self.message_history)
        self.end_of_qa = len(self.message_history) - 1
        return output_text

    def act(self, images: List[np.ndarray], prompt: str):
        print(prompt)
        content = [{"type": "text", "text": prompt}]

        image_msgs = [{
            "type": "input_image",
            "image_url": self._encode_image(img)
        } for img in images]

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": (
                            prompt
                        ),
                    }, *images
                ]
            }
        ]

        if len(self.message_history) + len(messages) + 1 - self.end_of_qa > self.action_history_len:
            self.message_history.pop(self.end_of_qa + 1)
            self.message_history.pop(self.end_of_qa + 1)

        self.message_history += messages
        output_text = self._generate(self.message_history)
        return output_text

system_prompt = "You are a helpful assistant. Your job is to figure out how to play this game called breakout where the objective is the maximize the score. Answer the questions in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>"
ACTION_HISTORY_LEN = 15
actor = GameActor(
    system_prompt=system_prompt,
    action_history_ring_buf_size=ACTION_HISTORY_LEN
)

action_space = load_action_space_dict("./Breakout-v5-action_space.csv")
print(action_space)

env_name = "ALE/Breakout-v5"
env = gymnasium.make(env_name, render_mode="rgb_array", obs_type="rgb")
env = gymnasium.wrappers.RecordVideo(
    env,
    episode_trigger=lambda ep: ep % 1 == 0,
    video_folder="policy_rollouts_" + env_name.split("/")[1],
    name_prefix="test_direct_policy"
)

rollouts = ["./policy_rollouts_Breakout-v5/random_policy-episode-{}.mp4".format(i) for i in range(0, 50, 30)]
visual_recognition_prompt = "Here are a few videos of the game breakout being played, ultimately we want play the game ourselves. Start by identify all the key mechanics. Be specific. Use at most 100 words." #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, [], visual_recognition_prompt)
print(f"<PROMPT>: {visual_recognition_prompt}")
print(f"<RESPONSE>: {visual_question_answer}")
#breakpoint()
actions = [f"'{a}' - {act['description']}" for a, act in action_space.items()]
random.shuffle(actions)
actions_str = str(actions[1:-1])
strategy_recognition_prompt = "The video is no longer relevant, let's start developing a strategy to play the real game. Describe the ideal strategy if you were playing this game. Be specific. Use at most 100 words." #ATARI-GPT
strategy_recognition_prompt += "Your available actions are given below in the format `[name of action] - [description of action]`: " + actions_str
strategy_question_answer = actor.question_answering(rollouts, [], strategy_recognition_prompt)
print(f"<PROMPT>: {strategy_recognition_prompt}")
print(f"<RESPONSE>: {strategy_question_answer}")
#breakpoint()

score = 0.0
action_names = list(action_space.keys())
def get_action_prompt(score, actions, lives):
    action_prompt = "Your current score is {score}, you have {lives} lives left, the last few actions you took were {actions}, The images are the current state of the game. What is the best action to take in this situation to increase your score?."
    if len(actions) == ACTION_HISTORY_LEN and all([action == actions[0] for action in actions]):
        action_prompt += f"You have been repeating your action choice for too long!!! YOU GOT TO HIT THE BALL WITH THE PADDLE! Try a different action! STOP SAYING {actions[0]} AND TRY SOMETHING LIKE {random.choice(list(set(action_names) - set(actions[0])))} ONCE IN A WHILE!!! YOU ARE NOT INCREASING THE SCORE!!!!/n</think>"
    action_prompt += f" Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>. Your answer should be a single word from this list: {action_names} in the form <answer>\nACTION\n</answer>"
    #action_prompt += "<think>\nIf there is no ball I should say FIRE. If the ball is to the right of the paddle I should say RIGHT. If the ball is to left of the paddle I should say LEFT\n</think>"
    return action_prompt.format(score=score, lives=lives, actions=actions)

for e in range(10):
    obs, info = env.reset()
    done = False
    step = 0
    score = 0
    action_history = []
    lives = 5

    while not done:
        random.shuffle(action_names)
        trace = actor.act([obs], prompt=get_action_prompt(score, action_history, lives)).strip()
        print(f"SAW: {trace}")
        keys_in_trace = [(action, action in trace) for action in action_space.keys()]
        num_keys = sum(1 for i in [i[1] for i in keys_in_trace] if i)
        while num_keys != 1:
            random.shuffle(action_names)
            trace = actor.act([obs], prompt=f"You did not respond with a valid answer, Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>. Your answer should be a single word a single word from this list: {action_names} in the form <answer>\nACTION\n</answer>.").strip()
            print(f"RECOVERY SAW: {trace}")
            keys_in_trace = [(action, action in trace) for action in action_space.keys()]
            num_keys = sum(1 for i in [i[1] for i in keys_in_trace] if i)

        action_str = "NOOP"
        for s,i in keys_in_trace:
            if i:
                action_str = s
        print(action_str)

        action_history.append(action_str)
        if len(action_history) > ACTION_HISTORY_LEN:
            action_history.pop(0)

        action = int(action_space[action_str]["index"])
        #action = env.action_space.sample()
        #print(int(action["index"]))
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        lives = info["lives"]
        print(f"Step {step}:, Action: {action}, Score: {score}, Reward: {reward}, info: {info}")
        done = terminated or truncated
        step += 3

    print(f"Episode {e} completed.")

env.close()
