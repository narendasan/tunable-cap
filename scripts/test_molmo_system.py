from typing import List, Dict, Any
from PIL import Image
import requests
import transformers
from transformers import AutoModelForImageTextToText, AutoProcessor, GenerationConfig
import torch
import ale_py
import gymnasium
import csv
from qwen_vl_utils import process_vision_info

gymnasium.register_envs(ale_py)

MODEL = "allenai/MolmoAct-7B-D-Pretrain-0812"


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
        self.model = AutoModelForImageTextToText.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16)
        self.processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True, use_fast=True)
        #self.processor.tokenizer.padding_side = "left"
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
        self.message_history = []
        # [{
        #     "role": "system",
        #     "content": [{
        #         "type": "text",
        #         "text": self.system_prompt,
        #     }],
        # }]

    def _generate(self, messages: List[Dict[Any, Any]]) -> str:
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
        output_text = self._generate(messages)
        return output_text[0]

# Example usage:
# actor = EagleGameActor(model, processor)
# code = actor.generate_gameplay_code("https://example.com/gameplay_image.jpg")
# print(code)


#import gymnasium as gym
#import ale_py

#gym.register_envs(ale_py)

#env = gym.make('ALE/Breakout-v5')
#obs, info = env.reset()
#obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#env.close()

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


generate_random_policy(num_episodes=50)

action_space = load_action_space_dict("./Breakout-v5-action_space.csv")
actions = [[a, act["description"]] for a, act in action_space.items()]
actions_str = str(actions)

system_prompt = "You are a helpful assistant. Your job is to figure out how to play a game based on provided videos. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>"

actor = GameActor(
    system_prompt=system_prompt
)

rollouts = ["./policy_rollouts_Breakout-v5/random_policy-episode-{}.mp4".format(i) for i in range(0, 50, 30)]
visual_recognition_prompt = "Here are a few videos of the game breakout being played, ultimately we want to write some code to play the game. Start by identify all the key elements in this image. Be specific. Use at most 100 words." #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, visual_recognition_prompt)
print(f"<PROMPT>: {visual_recognition_prompt}")
print(f"<RESPONSE>: {visual_question_answer}")
#breakpoint()
strategy_recognition_prompt = "Describe the ideal strategy if you were playing this game. Be specific. Use at most 100 words." #ATARI-GPT
strategy_question_answer = actor.question_answering(rollouts, strategy_recognition_prompt)
print(f"<PROMPT>: {strategy_recognition_prompt}")
print(f"<RESPONSE>: {strategy_question_answer}")
#breakpoint()

# breakdown_prompt = """Given your analysis, decompose the different modules you
#     would need to build to implement your strategy to **play** the game and get the highest
#     score. Each module should have a clear single role. Present the design as a ordered
#     list of modules to be built followed by a detailed specification."""

# breakdown_question_answer = actor.question_answering(rollouts, breakdown_prompt)
# print(f"<PROMPT>: {breakdown_prompt}")
# print(f"<RESPONSE>: {breakdown_question_answer}")
#breakpoint()

NUM_FRAMES_FOR_OBS = 3
code_prompt = "Given your plan, write the python code that plays the game to maximize the score. The code should take" \
    + str(NUM_FRAMES_FOR_OBS) \
    + """ frames of the game provided as a grayscale numpy array shaped
    Box(0, 255, (210, 160), np.uint8) for each frame as input and
    returns the action_name to take in the next time step.""" \
    + "Your available actions are given below in the format [[<action_name>, <description>]]: " + actions_str \
    + "For example to launch the ball to start the game or if the ball is not visible you should return `FIRE`, to move the paddle left or right you should return `LEFT` or `RIGHT` respectively. " \
    + """There should be a case for each action in your function. All of the code necessary must be written by you, you cannot
    call any external functions other than numpy or assume implementation. You cannot assume anything about the configuration of the game.
    No placeholders are allowed. You must be able to handle all possible game states. It is suggested that you use information across
    multiple frames to make your decisions"""

code_prompt += "Hint: The ball is kind of gray color try to use the motion of the ball and only focus on the middle section of the screen to identify the ball's position. Similarly for the paddle only look at the bottom of the screen to determine its position"
policy = actor.generate_gameplay_code([], prompt=code_prompt)
print(policy)

ACTIONS = list(action_space.keys())
if not all([a in policy for a in ACTIONS]):
    regenerate_code_prompt = "Not all actions are supported in your code. Your code must have a case for each action in the action space " + actions_str
    policy = actor.generate_gameplay_code([], prompt=regenerate_code_prompt)
    print(policy)

rollouts = ["./policy_rollouts_Breakout-v5/test_policy-episode-{}.mp4".format(i) for i in range(0, 10, 5)]
visual_recognition_prompt = "Here are some videos of breakout being played with your policy, what are the issues that you can see in the videos?" #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, visual_recognition_prompt)
print(f"<PROMPT>: {visual_recognition_prompt}")
print(f"<RESPONSE>: {visual_question_answer}")

code_prompt = "Regenerate the code to fix the issues you found in the videos"
policy = actor.generate_gameplay_code([], prompt=code_prompt)
print(policy)
