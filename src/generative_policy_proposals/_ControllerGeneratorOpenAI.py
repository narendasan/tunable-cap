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
from ._ControllerGenerator import ControllerGenerator
import sys
from openai import OpenAI

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level= logging.INFO)

gymnasium.register_envs(ale_py)


class ControllerGeneratorOpenAI(ControllerGenerator):
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