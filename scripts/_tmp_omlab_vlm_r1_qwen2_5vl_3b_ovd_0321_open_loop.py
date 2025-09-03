from typing import Optional, Tuple
from generative_policy_proposals.controller_utils.breakout_utilities import *
import numpy as np

def predict_next_action(frames: np.ndarray) -> int:
    # Check if the ball is in the current frame
    if not provided_is_ball_in_frame(frames[-1]):
        return 1  # Launch the ball

    # Get the ball position estimate
    ball_pos = provided_get_ball_position(frames[-1])

    # Get the paddle position estimate
    paddle_pos = provided_get_paddle_position(frames[-1])

    # Check for collisions with the paddle
    if np.any(np.abs(ball_pos - paddle_pos) < 10):
        return 2  # Move the paddle right

    # Check for collisions with the walls
    if ball_pos[0] <= 0 or ball_pos[0] >= 209 or ball_pos[1] <= 0 or ball_pos[1] >= 159:
        return 3  # Move the paddle left

    return 0  # No special action needed
