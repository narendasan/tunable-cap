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
        model="google/gemma-3-12b-it:free", 
        system_prompt="You are a helpful assistant. Answer the question in the following format: <think>\n"
                        "your reasoning\n</think>\n\n"
                        "<answer>\nyour answer\n</answer>. "
                        "You are tasked to generate code to play a game based on videos you watch",
        base_url = "https://openrouter.ai/api/v1",
        api_key = os.getenv("GEMMA3_12B_API_KEY")
        ):
        self.model_name = model
        self.client = OpenAI(
            base_url=base_url, 
            api_key=api_key, 
            default_headers ={"HTTP-Referer": "http://localhost", "X-Title": "tunable-cap"}
            )
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
        self.default_temperature = 0.6
        self.default_top_p = 0.96
        self.default_max_tokens = 4096
        
        #self.current_code_msg = []

    # OpenAI compatible, text only
    def _generate(self, messages: List[Dict[Any, Any]]) -> str:
        _LOGGER.debug(messages)

        # 1. No chat_template needed: OpenAI API accepts chat format directly.
        #    If messages contain parts (text + images), we ignore images/videos for now.
        oa_messages: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                # flatten to just text parts for now
                content = "\n".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            elif isinstance(content, dict) and content.get("type") == "text":
                content = content.get("text", "")
            elif not isinstance(content, str):
                content = str(content)
            oa_messages.append({"role": role, "content": content})

        # 2–6. Skip processor/vision/tensors/device/generate:
        #       the server handles all of that internally.

        # 7. Call the API directly
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=oa_messages,
            temperature=getattr(self, "default_temperature", 0.6),
            top_p=getattr(self, "default_top_p", 0.95),
            max_tokens=getattr(self, "default_max_tokens", 300),
        )

        output_text = resp.choices[0].message.content if resp.choices else ""

        # 8. Append to history and return
        self.message_history.append({
            "role": "assistant",
            "content": output_text,
        })
        return output_text

    # Text only, with video placeholder
    def question_answering(self, video_paths: List[str], prompt: str) -> str:
        if video_paths:
            placeholder = "\n".join(f"[VIDEO:{p}]" for p in video_paths)
            content = f"{prompt}\n{placeholder}"
        else:
            content = prompt

        self.message_history.append({"role": "user", "content": content})
        return self._generate(self.message_history)


    # Text only, with video placeholder
    def generate_gameplay_code(self, video_paths: List[str], prompt: str = "Generate Python code to play this game based on the video.") -> str:
        # While videos aren't sent yet, include placeholders so the model "sees" that clips exist.
        if video_paths:
            placeholder = "\n".join(f"[VIDEO:{p}]" for p in video_paths)
            content = f"{prompt}\n{placeholder}"
        else:
            content = prompt

        self.message_history.append({"role": "user", "content": content})

        output_text = self._generate(self.message_history)
        # self.current_code_msg = output_text
        return output_text

def generate_and_load_policy(actor,
                            action_space: Dict[str, Any],
                            prompt: str,
                            videos: Optional[List[str]] = None,
                            num_codegen_retries: int = 10,
                            exp_name: str = "",
                            dir_prefix: str = "codegen") -> Tuple[Callable, str]:
    actions = [[act["index"], act["description"]] for _, act in action_space.items()]
    actions_str = str(actions)

    videos_ = videos or []

    # OpenAI path: returns a STRING (not a list)
    policy: str = actor.generate_gameplay_code(videos_, prompt=prompt)
    _LOGGER.debug(policy)

    ACTIONS = [a["index"] for a in action_space.values()]
    # Compare as strings to be safe if indices are ints
    if not all(str(a) in policy for a in ACTIONS):
        regenerate_code_prompt = (
            "Not all actions are supported in your code. Your code must have a case for each action in the action space "
            + actions_str
        )
        policy = actor.generate_gameplay_code([], prompt=regenerate_code_prompt)
        _LOGGER.debug(policy)

    retries = num_codegen_retries
    while True:
        try:
            code = [
                "from typing import Optional, Tuple\n",
                "from generative_policy_proposals.controller_utils.breakout_utilities import *\n",
            ] + get_code_from_markdown(policy)

            os.makedirs(dir_prefix, exist_ok=True)

            safe = slugify(actor.model_name).replace("-", "_")
            tmp_mod_name = f"_tmp_{safe}_{exp_name}"
            tmp_py = f"{dir_prefix}/{tmp_mod_name}.py"

            with open(tmp_py, "w") as f:
                f.writelines(code)

            log_dir = f"logs/{safe}"
            os.makedirs(log_dir, exist_ok=True)
            with open(f"{log_dir}/{datetime.datetime.now()}_{exp_name}.py.log", "w") as f:
                f.writelines(code)

            spec = importlib.util.spec_from_file_location(tmp_mod_name, tmp_py)
            policy_mod = importlib.util.module_from_spec(spec)
            sys.modules[tmp_mod_name] = policy_mod
            spec.loader.exec_module(policy_mod)  # type: ignore[attr-defined]

            predict_next_action = getattr(policy_mod, "predict_next_action")
            break
        except Exception as e:
            _LOGGER.error(e)
            regenerate_code_prompt = (
                "I am unable to import the `predict_next_action` function; "
                "saw this error in your code " + str(e)
            )
            policy = actor.generate_gameplay_code([], prompt=regenerate_code_prompt)
            _LOGGER.debug(policy)

        if retries == 0:
            _LOGGER.error("Failed to regenerate code")
            exit(-1)
        retries -= 1

    return predict_next_action, policy


def regenerate_policy(actor,
                    action_space: Dict[str, Any],
                    qa_prompt: str,
                    videos: List[str],
                    code_prompt: str,
                    num_codegen_retries: int = 10,
                    exp_name: str = "",
                    dir_prefix: str = "codegen") -> Tuple[Callable, str]:
    # OpenAI path: returns a STRING
    regenerate_strategy_answer = actor.question_answering(videos, qa_prompt)
    _LOGGER.info(f"<PROMPT>: {qa_prompt}")
    _LOGGER.info(f"<RESPONSE>: {regenerate_strategy_answer}")

    retries = num_codegen_retries
    regenerate_prompt = code_prompt

    # simple smoke inputs for the generated policy
    obs = np.random.randint(low=0, high=256, size=(210, 160), dtype=np.uint8)
    weights = np.random.randn(1, 10)
    memory = None

    while True:
        predict_next_action, policy = generate_and_load_policy(
            actor, action_space, regenerate_prompt,
            num_codegen_retries=num_codegen_retries,
            exp_name=exp_name, dir_prefix=dir_prefix
        )
        try:
            action_ = predict_next_action(obs, weights, memory)
            _LOGGER.debug("Test action: " + str(action_))
            action = int(action_)  # ensure it’s castable
            break
        except Exception as e:
            _LOGGER.debug(f"Error occurred while predicting next action: {e}")
            regenerate_prompt = f"While executing your code, I hit the following error: {e}. Please try again."

        if retries == 0:
            _LOGGER.error("Failed to regenerate code")
            exit(-1)
        retries -= 1

    return predict_next_action, policy

