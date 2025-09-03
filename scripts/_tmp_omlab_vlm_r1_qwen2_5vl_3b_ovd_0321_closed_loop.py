from typing import Optional, Tuple
from generative_policy_proposals.controller_utils.breakout_utilities import *
import numpy as np
from typing import Optional

class MemoryClass:
    def __init__(self):
        self.state_history = []
        self.frame_history = []

    def update_state(self, distance_to_paddle, frame):
        self.state_history.append(distance_to_paddle)
        self.frame_history.append(frame)

def calculate_distance(ball_pos, paddle_pos):
    return np.linalg.norm(ball_pos - paddle_pos)

def predict_next_action(frame: np.ndarray, weights: np.ndarray, memory: Optional[MemoryClass]) -> Tuple[int, MemoryClass]:
    if not np.any(frame):
        return 1, memory

    if not provided_is_ball_in_frame(frame):
        return 1, memory

    ball_pos = provided_get_ball_position(frame)
    paddle_pos = provided_get_paddle_position(frame)

    distance_to_paddle = calculate_distance(ball_pos, paddle_pos)

    if distance_to_paddle > 0.5:
        return 2, memory
    elif distance_to_paddle < 0.1:
        return 3, memory

    if memory is not None:
        memory.update_state(distance_to_paddle, frame)

    return 0, memory