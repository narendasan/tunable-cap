from typing import Optional, Tuple
from generative_policy_proposals.controller_utils.breakout_utilities import *
import numpy as np
from typing import Optional, Tuple
import torch


class MemoryState:
    def __init__(self):
        self.last_frame = None
        self.last_ball_pos = None
        self.last_paddle_pos = None
        self.score = 0
        self.frame_count = 0

    def __str__(self):
        return f"MemoryState(last_ball_pos={self.last_ball_pos}, last_paddle_pos={self.last_paddle_pos}, score={self.score}, frame_count={self.frame_count})"


# Added score by hand
def predict_next_action(
    frame: np.ndarray, *, model: torch.nn.Module, memory: Optional[MemoryState] = None
) -> Tuple[int, MemoryState]:
    if memory is None:
        memory = MemoryState()

    memory.frame_count += 1

    ball_pos = provided_get_ball_position(frame)
    paddle_pos = provided_get_paddle_position(frame)

    if provided_is_ball_in_frame(frame) and memory.last_ball_pos is not None:
        dx = ball_pos[0] - memory.last_ball_pos[0]
        dy = ball_pos[1] - memory.last_ball_pos[1]
    else:
        dx, dy = 0, 0

    if memory.last_paddle_pos is not None:
        paddle_dx = paddle_pos[0] - memory.last_paddle_pos[0]
    else:
        paddle_dx = 0

    features = np.array(
        [
            [dx],  # Change in ball x position
            [dy],  # Change in ball y position
            [paddle_dx],  # Change in paddle x position
            [
                ball_pos[0] / 160 if provided_is_ball_in_frame(frame) else -1
            ],  # ball x position
            [
                ball_pos[1] / 210 if provided_is_ball_in_frame(frame) else -1
            ],  # ball y position
            [
                memory.last_ball_pos[0] / 160
                if memory.last_ball_pos is not None
                else -1
            ],  # Last ball x position
            [
                memory.last_ball_pos[1] / 210
                if memory.last_ball_pos is not None
                else -1
            ],  # Last ball y position
            [paddle_pos[0] / 160],  # Current paddle x position
            [
                memory.last_paddle_pos[0] / 160
                if memory.last_paddle_pos is not None
                else 0
            ],  # Last paddle x position
            # [memory.frame_count],  # Frame count
            # [memory.score],  # Score
            [int(provided_is_ball_in_frame(frame))],  # Ball presence
        ]
    )

    # print(features)
    assert not np.any(np.isnan(features)), (
        f"Features are NaN: features:{features}, memory:{memory}, ball_pos:{ball_pos}, paddle_pos:{paddle_pos}"
    )

    # weighted_features = weights @ features + biases
    q_values = model(torch.Tensor(features).T.to("cuda"))
    # print(q_values)
    action = torch.argmax(q_values, dim=1).cpu().numpy()
    # action = np.argmax(weighted_features)

    # Added by hand
    memory.last_ball_pos = ball_pos if provided_is_ball_in_frame(frame) else None
    memory.last_paddle_pos = paddle_pos
    memory.last_frame = frame
    # memory.score = score

    # if action == 0:
    #     return 0, memory
    # elif action == 1:
    #     return 1, memory
    # elif action == 2:
    #     return 2, memory
    # elif action == 3:
    #     return 3, memory

    # Added by hand
    # augmented_rewards = paddle[0] - ball_pos[0]
    return (action, memory), q_values


# Handwrtitten based on generated code:
def extract_features(
    frame: np.ndarray, memory: Optional[MemoryState] = None
) -> Tuple[np.ndarray, MemoryState]:
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

    features = np.array(
        [
            [dx],  # Change in ball x position
            [dy],  # Change in ball y position
            [paddle_dx],  # Change in paddle x position
            [
                memory.last_ball_pos[0] if memory.last_ball_pos is not None else 0
            ],  # Last ball x position
            [
                memory.last_ball_pos[1] if memory.last_ball_pos is not None else 0
            ],  # Last ball y position
            [paddle_pos[0]],  # Current paddle x position
            [
                memory.last_paddle_pos[0] if memory.last_paddle_pos is not None else 0
            ],  # Last paddle x position
            [memory.frame_count],  # Frame count
            [memory.score],  # Score
            [int(provided_is_ball_in_frame(frame))],  # Ball presence
        ]
    )

    # Added by hand
    memory.last_ball_pos = ball_pos
    memory.last_paddle_pos = paddle_pos
    memory.last_frame = frame
    memory.score = score
    return features, memory
