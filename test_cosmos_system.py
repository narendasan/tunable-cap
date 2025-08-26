from collections import namedtuple
from typing import List, Dict, Any, Optional
from PIL import Image
import requests
import transformers
from transformers import AutoProcessor, AutoModel, Qwen2_5_VLForConditionalGeneration
import torch
import ale_py
import gymnasium
import csv
from qwen_vl_utils import process_vision_info
from breakout_utilities import UTILITY_SPECS
from slugify import slugify
import re
import importlib
import datetime
import logging
import os

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

gymnasium.register_envs(ale_py)

MODEL = "nvidia/Cosmos-Reason1-7B"
ENV_NAME = "ALE/Breakout-v5"
ACTION_HISTORY_LEN = 50
NUM_FRAMES_FOR_OBS = 3
NUM_CODEGEN_RETRIES = 50
NUM_REFINEMENT_EPISODES = 5

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

class GameActor:
    def __init__(self, system_prompt="You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>. You are tasked to generate code to play a game based on videos you watch", device="cuda"):
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.generation_config = transformers.GenerationConfig(
            do_sample=True,
            max_new_tokens=4096,
            repetition_penalty=1.05,
            temperature=0.6,
            top_p=0.95,
        )
        self.device = device
        self.model = self.model.to(self.device)
        self.system_prompt = system_prompt
        self.fps = 12
        self.max_pixels = 1920
        self.message_history = [{
            "role": "system",
            "content": [{
                "type": "text",
                "text": self.system_prompt,
            }],
        }]
        #self.current_code_msg = []

    def _generate(self, messages: List[Dict[Any, Any]]) -> str:
        _LOGGER.debug(messages)
        text_list = [self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )]
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=text_list,
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt",
            padding=True,
            #**video_kwargs,
        )
        inputs = inputs.to(self.device)
        generated_ids = self.model.generate(**inputs, generation_config=self.generation_config)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        self.message_history.append({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": output_text,
            }],
        })
        return output_text

    def question_answering(self, video_paths: List[str], prompt: str):
        video_content = [
            {
                "type": "video",
                "video": video_path,
                "fps": self.fps,
                "max_pixels": self.max_pixels
            } for video_path in video_paths
        ]

        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }] + video_content
            }
        ]
        self.message_history += messages
        output_text = self._generate(self.message_history)
        return output_text[0]

    def generate_gameplay_code(self, video_paths: List[str], prompt="Generate Python code to play this game based on the video."):
        video_content = [
            {
                "type": "video",
                "video": video_path,
                "fps": self.fps,
                "max_pixels": self.max_pixels
            } for video_path in video_paths
        ]

        messages = [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": prompt
                }] + video_content
            }
        ]
        output_text = self._generate(self.message_history + messages)
        #self.current_code_msg = output_text

        return output_text[0]

def generate_random_policy(env_name: str="ALE/Breakout-v5", num_episodes: int = 50) -> None:
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env = gymnasium.wrappers.RecordVideo(
        env,
        episode_trigger=lambda ep: ep % 10 == 0,
        video_folder="policy_rollouts_" + env_name.split("/")[1],
        name_prefix="random_policy"
    )

    for e in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()

Eval = namedtuple('Eval', ["actions", "score", "lives", "info"])

def evaluate_policy(num_episodes: int = 10) -> List[Eval]:
    env = gymnasium.make(ENV_NAME, render_mode="rgb_array", obs_type="grayscale")
    env = gymnasium.wrappers.RecordVideo(
        env,
        episode_trigger=lambda ep: ep % 1 == 0,
        video_folder="policy_rollouts_" + ENV_NAME.split("/")[1],
        name_prefix=f"eval_{slugify(MODEL).replace('-', '_')}_policy"
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
            score += reward
            lives = info["lives"]
            print(f"Step {step}:, Action: {action}, Score: {score}, Reward: {reward}, info: {info}")
            step += 1

        evals.append(Eval(action_history, score, lives, info))
        print(f"Episode {e} completed.")


    env.close()
    return evals

def get_code_from_markdown(text, *, language: str = "python") -> list[str]:
    """Outputs extracted code blocks from a list of strings of markdown text"""
    if language:
        # Pattern for a specific language (e.g., ```python ... ```)
        pattern = rf"```(?:{language})\n(.*?)```"
    else:
        # Pattern for any code block (``` ... ``` or ```language ... ```)
        pattern = r"```(?:\w+)?\n(.*?)```"

    code_blocks = re.findall(pattern, text, re.DOTALL)
    return [block.strip() for block in code_blocks]

    # out = re.search(r'^```(?:py|python)\n([\s\S]*?)```$', text).group(1)
    # print(out)
    # return [out]

generate_random_policy(env_name=ENV_NAME, num_episodes=50)

action_space = load_action_space_dict("./Breakout-v5-action_space.csv")
actions = [[act["index"], act["description"]] for a, act in action_space.items()]
actions_str = str(actions)

system_prompt = "You are a helpful assistant. Your job is to figure out how to play a game based on provided videos. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>"

actor = GameActor(
    system_prompt=system_prompt
)

rollouts = ["./policy_rollouts_Breakout-v5/random_policy-episode-{}.mp4".format(i) for i in range(0, 20, 10)]
visual_recognition_prompt = "Here are a few videos of the game breakout being played, ultimately we want to write some code to play the game. Start by identify all the key elements in this image. Be specific. Use at most 100 words." #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, visual_recognition_prompt)
_LOGGER.info(f"<PROMPT>: {visual_recognition_prompt}")
_LOGGER.info(f"<RESPONSE>: {visual_question_answer}")
#breakpoint()
strategy_recognition_prompt = "The video is no longer relevant, let's start developing a strategy to play the real game. Describe the ideal strategy if you were playing this game. Be specific. Use at most 100 words." #ATARI-GPT
strategy_recognition_prompt += "Your available actions are given below in the format `[action] - [description of action]`: " + actions_str
strategy_question_answer = actor.question_answering([], strategy_recognition_prompt)
_LOGGER.info(f"<PROMPT>: {strategy_recognition_prompt}")
_LOGGER.info(f"<RESPONSE>: {strategy_question_answer}")
#breakpoint()

# breakdown_prompt = """Given your analysis, decompose the different modules you
#     would need to build to implement your strategy to **play** the game and get the highest
#     score. Each module should have a clear single role. Present the design as a ordered
#     list of modules to be built followed by a detailed specification."""

# breakdown_question_answer = actor.question_answering(rollouts, breakdown_prompt)
# print(f"<PROMPT>: {breakdown_prompt}")
# print(f"<RESPONSE>: {breakdown_question_answer}")
#breakpoint()

def generate_and_load_policy(prompt: str, videos: Optional[List[str]]=None) -> str:
    videos_ = []
    if videos is not None:
        videos_ = videos

    global predict_next_action
    policy = actor.generate_gameplay_code(videos_, prompt=prompt)
    _LOGGER.debug(policy)

    ACTIONS = list([a["index"] for a in action_space.values()])
    if not all([a in policy for a in ACTIONS]):
        regenerate_code_prompt = "Not all actions are supported in your code. Your code must have a case for each action in the action space " + actions_str
        policy = actor.generate_gameplay_code([], prompt=regenerate_code_prompt)
        _LOGGER.debug(policy)

    retries = NUM_CODEGEN_RETRIES
    while True:
        try:
            code = ["from breakout_utilities import *\n", "import cv2\n"] + get_code_from_markdown(policy)
            with open(f"_tmp_{slugify(MODEL).replace('-', '_')}.py", "w") as f:
                f.writelines(code)
            if not os.path.exists(f"logs/{slugify(MODEL).replace('-', '_')}"):
                os.makedirs(f"logs/{slugify(MODEL).replace('-', '_')}")
            with open(f"logs/{slugify(MODEL).replace('-', '_')}/_tmp_{datetime.datetime.now()}.py.log", "w") as f:
                f.writelines(code)
            gen_mod = importlib.import_module(f"_tmp_{slugify(MODEL).replace('-', '_')}")
            predict_next_action = getattr(gen_mod, "predict_next_action")
            break
        except Exception as e:
            _LOGGER.error(e)
            regenerate_code_prompt = "I am unable to import the `predict_next_action` function "
            regenerate_code_prompt += " saw this error in your code " + str(e)
            policy = actor.generate_gameplay_code([], prompt=regenerate_code_prompt)
            _LOGGER.debug(policy)

        if retries == 0:
            _LOGGER.error("Failed to regenerate code")
            exit(-1)
        retries -= 1

    return policy

code_prompt = "Given your plan, write the python code that plays the game to maximize the score. The code should take" \
    + str(NUM_FRAMES_FOR_OBS) \
    + """ frames of the game provided as a grayscale numpy array shaped
    Box(0, 255, (210, 160), np.uint8) for each frame as input and
    returns the action to take in the next time step.""" \
    + "Your available actions are given below in the format [[<action>, <description>]]: " + actions_str \
    + "For example to launch the ball to start the game or if the ball is not visible you should return `1`, to move the paddle left or right you should return `3` or `2` respectively. " \
    + """There should be a case for each action in your function. All of the code necessary must be written by you, you cannot
    call any external functions other than numpy or assume implementation. You cannot assume anything about the configuration of the game.
    No placeholder code or examples are allowed. You must be able to handle all possible game states. It is suggested that you use information across
    multiple frames to make your decisions. The entry point for your code should be `def predict_next_action(frames: np.ndarray) -> int:`"""

code_prompt += "You are provided the following utility functions to help you implement your policy: " + str(UTILITY_SPECS)
code_prompt += "Only generate python code, do not generate any comments"

generate_and_load_policy(code_prompt)
assert(predict_next_action is not None)

def reset_env():
    env = gymnasium.make(ENV_NAME, render_mode="rgb_array", obs_type="grayscale")
    env = gymnasium.wrappers.RecordVideo(
        env,
        episode_trigger=lambda ep: ep % 1 == 0,
        video_folder="policy_rollouts_" + ENV_NAME.split("/")[1],
        name_prefix=f"refine_{slugify(MODEL).replace('-', '_')}_policy"
    )

    obs, info = env.reset()
    done = False
    step = 0
    score = 0
    obs_history = [obs]
    action_history = []
    lives = 5

    for i in range(NUM_FRAMES_FOR_OBS - 1):
        obs, reward, terminated, truncated, info = env.step(int(action_space["NOOP"]["index"]))
        action_history.append(int(action_space["NOOP"]["index"]))
        obs_history.append(obs)

    return (env, obs, done, step, score, lives, obs_history, action_history)

def regenerate_policy(qa_prompt: str, videos: List[str], code_prompt: str):
    regenerate_strategy_answer = actor.question_answering(videos, qa_prompt)
    _LOGGER.info(f"<PROMPT>: {qa_prompt}")
    _LOGGER.info(f"<RESPONSE>: {regenerate_strategy_answer}")
    retries = NUM_CODEGEN_RETRIES
    regenerate_prompt = code_prompt
    while True:
        generate_and_load_policy(regenerate_prompt)
        try:
            action_ = predict_next_action(obs_history)
            _LOGGER.debug("Test action: " + action_)
            action = int(action_)
            break
        except Exception as e:
            breakpoint()
            _LOGGER.debug(f"Error occurred while predicting next action: {e}")
            regenerate_prompt = f"While executing your code, I hit to following error: {e}. Please try again."
        if retries == 0:
            breakpoint()
            _LOGGER.error("Failed to regenerate code")
            exit(-1)
        retries -= 1

for r in range(NUM_REFINEMENT_EPISODES):
    env, obs, done, step, score, lives, obs_history, action_history = reset_env()

    while not done:
        action = 0
        try:
            action_ = predict_next_action(obs_history)
            _LOGGER.debug(action_)
            action = int(action_)
        except Exception as e:
            _LOGGER.error(f"Error occurred while predicting next action: {e}")
            regenerate_prompt = f"While executing your code, I hit to following error: {e}. Please try again."

            retries = NUM_CODEGEN_RETRIES
            while True:
                generate_and_load_policy(regenerate_prompt)
                try:
                    action_ = predict_next_action(obs_history)
                    _LOGGER.debug(action_)
                    action = int(action_)
                    break
                except Exception as e:
                    _LOGGER.error(f"Error occurred while predicting next action: {e}")
                    regenerate_prompt = f"While executing your code, I hit to following error: {e}. Please try again."
                if retries == 0:
                    _LOGGER.error("Failed to regenerate code")
                    exit(-1)
                retries -= 1

        #action = env.action_space.sample()
        #print(int(action["index"]))
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        lives = info["lives"]
        _LOGGER.info(f"Step {step}:, Action: {action}, Score: {score}, Reward: {reward}, info: {info}")
        done = terminated or truncated
        step += 1
        action_history.append(action)
        if len(action_history) > ACTION_HISTORY_LEN:
            action_history.pop(0)

        obs_history.append(obs)
        if len(obs_history) > NUM_FRAMES_FOR_OBS:
            obs_history.pop(0)

        if len(action_history) == ACTION_HISTORY_LEN and all([action_history[0] == i for i in action_history]):
            env.close()
            _LOGGER.info(f"No meaningful change in actions: {action_history}")
            regenerate_strategy_prompt = f"I have been seeing your code run the same action too many times repeatedly, Your current score is {score}, you have {lives} lives left, the last few actions you took were {action_history}. Video of your code running is provided as well. What issues do you see?"
            regenerate_policy(regenerate_strategy_prompt, [f"./policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-0.mp4"], code_prompt)
            # Restart the refinement episode
            env, obs, done, step, score, lives, obs_history, action_history = reset_env()
            _LOGGER.info("Restarting refinement episode")

    env.close()
    refine_prompt = f"Your current score is {score}, you have {lives} lives left, the last few actions you took were {action_history}. Video of your code running is provided as well. How can you improve this code?"
    regenerate_policy(refine_prompt, [f"./policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-{r}.mp4"], code_prompt)
    print(f"Refinement Episode {r} completed.")
    breakpoint()
    os.rename(f"./policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-0.mp4", f"./policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-{r+10}.mp4")
exit()

evals = evaluate_policy(num_episodes=10)
print(evals)

rollouts = ["./policy_rollouts_Breakout-v5/eval_policy-episode-{}.mp4".format(i) for i in range(0, 10, 5)]
visual_recognition_prompt = "Here are some videos of breakout being played with your policy, what are the issues that you can see in the videos?" #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, visual_recognition_prompt)
print(f"<PROMPT>: {visual_recognition_prompt}")
print(f"<RESPONSE>: {visual_question_answer}")

code_prompt = "Regenerate the code to fix the issues you found in the videos"
policy = actor.generate_gameplay_code([], prompt=code_prompt)
print(policy)
