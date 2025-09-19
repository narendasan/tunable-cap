from typing import List, Dict, Any, Optional, Tuple, Callable
import torch
import ale_py
import gymnasium
from qwen_vl_utils import process_vision_info
from slugify import slugify
import importlib
import datetime
import logging
import os
import numpy as np
from generative_policy_proposals._utils import get_code_from_markdown
import sys
from openai import OpenAI

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level= logging.INFO)

gymnasium.register_envs(ale_py)


class ControllerGenerator:
    def __init__(
        self, 
        model="Qwen/Qwen2-VL-2B-Instruct", 
        system_prompt="You are a helpful assistant. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>. You are tasked to generate code to play a game based on videos you watch", device="cuda"):
        base_url = "http://127.0.0.1:8000/v1"
        api_key = "EMPTY"
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True, use_fast=True)
        self.processor.tokenizer.padding_side = "left"
        self.model = model
        self.client = OpenAI(base_url, api_key)
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
        self.message_history += messages
        output_text = self._generate(self.message_history)
        #self.current_code_msg = output_text

        return output_text[0]

def generate_and_load_policy(actor: ControllerGenerator, action_space: Dict[str, Any], prompt: str, videos: Optional[List[str]]=None, num_codegen_retries: int = 10, exp_name: str = "", dir_prefix: str = "codegen") -> Tuple[Callable, str]:
    actions = [[act["index"], act["description"]] for a, act in action_space.items()]
    actions_str = str(actions)

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

    retries = num_codegen_retries
    while True:
        try:
            code = ["from typing import Optional, Tuple\n", "from generative_policy_proposals.controller_utils.breakout_utilities import *\n"] + get_code_from_markdown(policy)
            os.makedirs(f"{dir_prefix}", exist_ok=True)
            with open(f"{dir_prefix}/_tmp_{slugify(actor.model_name).replace('-', '_')}_{exp_name}.py", "w") as f:
                f.writelines(code)
            if not os.path.exists(f"logs/{slugify(actor.model_name).replace('-', '_')}"):
                os.makedirs(f"logs/{slugify(actor.model_name).replace('-', '_')}")
            with open(f"logs/{slugify(actor.model_name).replace('-', '_')}/_tmp_{datetime.datetime.now()}_{exp_name}.py.log", "w") as f:
                f.writelines(code)
            spec = importlib.util.spec_from_file_location(f"_tmp_{slugify(actor.model_name).replace('-', '_')}_{exp_name}", f"{dir_prefix}/_tmp_{slugify(actor.model_name).replace('-', '_')}_{exp_name}.py")
            policy_mod = importlib.util.module_from_spec(spec)
            sys.modules["_tmp_{slugify(actor.model_name).replace('-', '_')}_{exp_name}"] = policy_mod
            spec.loader.exec_module(policy_mod)
            predict_next_action = getattr(policy_mod, "predict_next_action")
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

    return predict_next_action, policy

def regenerate_policy(actor: ControllerGenerator, action_space: Dict[str, Any], qa_prompt: str, videos: List[str], code_prompt: str, num_codegen_retries: int = 10, exp_name: str = "", dir_prefix: str = "codegen") -> Tuple[Callable, str]:
    regenerate_strategy_answer = actor.question_answering(videos, qa_prompt)
    _LOGGER.info(f"<PROMPT>: {qa_prompt}")
    _LOGGER.info(f"<RESPONSE>: {regenerate_strategy_answer}")
    retries = num_codegen_retries
    regenerate_prompt = code_prompt
    obs = np.random.randint(low=0, high=256, size=(210, 160), dtype=np.uint8)
    weights = np.random.randn(1,10)
    memory = None
    while True:
        predict_next_action, policy = generate_and_load_policy(actor, action_space, regenerate_prompt, num_codegen_retries=num_codegen_retries, exp_name=exp_name, dir_prefix=dir_prefix)
        try:
            action_ = predict_next_action(obs, weights, memory)
            _LOGGER.debug("Test action: " + action_)
            action = int(action_)
            break
        except Exception as e:
            _LOGGER.debug(f"Error occurred while predicting next action: {e}")
            regenerate_prompt = f"While executing your code, I hit to following error: {e}. Please try again."
        if retries == 0:
            _LOGGER.error("Failed to regenerate code")
            exit(-1)
        retries -= 1
    return predict_next_action, policy
