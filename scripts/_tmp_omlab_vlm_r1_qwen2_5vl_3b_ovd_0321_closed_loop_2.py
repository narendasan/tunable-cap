from typing import Optional, Tuple
from generative_policy_proposals.controller_utils.breakout_utilities import *
import numpy as np

def predict_next_action(frame: np.ndarray, weights: np.ndarray, memory: np.ndarray) -> Tuple[int, np.ndarray]:
    def is_ball_in_frame(frame):
        return np.any(provided_is_ball_in_frame(frame))

    def get_ball_position(frame):
        return provided_get_ball_position(frame)

    def get_paddle_position(frame):
        return provided_get_paddle_position(frame)

    # Check if the ball is in the frame
    if not is_ball_in_frame(frame):
        return 1, memory  # Launch a new ball

    # Get the current ball position
    ball_pos = get_ball_position(frame)

    # Get the current paddle position
    paddle_pos = get_paddle_position(frame)

    # Calculate the distance between the ball and the paddle
    dx = ball_pos[0] - paddle_pos[0]
    dy = ball_pos[1] - paddle_pos[1]

    # Determine the direction of the ball relative to the paddle
    if dx > 0:
        dir_x = 2  # Right
    elif dx < 0:
        dir_x = 3  # Left
    else:
        dir_x = 0  # No movement

    if dy > 0:
        dir_y = 2  # Up
    elif dy < 0:
        dir_y = 3  # Down
    else:
        dir_y = 0  # No movement

    # Update the memory with the current state
    memory = np.append(memory, [dx, dy, dir_x, dir_y])

    # Choose the best action based on the current state
    if dir_x == 0 and dir_y == 0:
        return 0, memory  # No action needed
    elif dir_x != 0:
        return 2, memory  # Move the paddle right
    elif dir_y != 0:
        return 3, memory  # Move the paddle left
    else:
        return 1, memory  # Fire the ball