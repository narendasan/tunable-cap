from typing import Optional, Tuple
from generative_policy_proposals.controller_utils.breakout_utilities import *
import numpy as np
from typing import Optional, Tuple

class MemoryState:
    def __init__(self):
        self.last_frame = None
        self.last_ball_pos = None
        self.last_paddle_pos = None
        self.score = 0
        self.frame_count = 0

def predict_next_action(frame: np.ndarray, weights: np.ndarray, biases: np.ndarray, memory: Optional[MemoryState] = None) -> Tuple[int, MemoryState]:
    if memory is None:
        memory = MemoryState()

    memory.frame_count += 1

    ball_pos = provided_get_ball_position(frame)
    paddle_pos = provided_get_paddle_position(frame)

    if memory.last_ball_pos is not None:
        dx = ball_pos[0] - memory.last_ball_pos[0]
        dy = ball_pos[1] - memory.last_ball_pos[1]
    else:
        dx, dy = 0, 0

    if memory.last_paddle_pos is not None:
        paddle_dx = paddle_pos[0] - memory.last_paddle_pos[0]
    else:
        paddle_dx = 0

    features = np.array([
        [dx],  # Change in ball x position
        [dy],  # Change in ball y position
        [paddle_dx],  # Change in paddle x position
        [memory.last_ball_pos[0] if memory.last_ball_pos is not None else 0],  # Last ball x position
        [memory.last_ball_pos[1] if memory.last_ball_pos is not None else 0],  # Last ball y position
        [paddle_pos[0]],  # Current paddle x position
        [memory.last_paddle_pos[0] if memory.last_paddle_pos is not None else 0],  # Last paddle x position
        [memory.frame_count],  # Frame count
        [memory.score],  # Score
        [int(provided_is_ball_in_frame(frame))]  # Ball presence
    ])

    weighted_features = weights @ features + biases
    action = np.argmax(weighted_features)

    if action == 0:
        return 0, memory
    elif action == 1:
        return 1, memory
    elif action == 2:
        return 2, memory
    elif action == 3:
        return 3, memory

    return 0, memory
