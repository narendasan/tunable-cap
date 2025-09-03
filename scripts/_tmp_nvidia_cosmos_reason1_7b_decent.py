from breakout_utilities import *
import cv2
import numpy as np

def predict_next_action(frames: np.ndarray) -> int:
    # Check if the ball is present in the current frame
    if not provided_is_ball_in_frame(frames[-1]):
        return 1  # Launch a new ball to start the game

    # Estimate the ball's position
    ball_pos = provided_get_ball_position(frames[-1])
    paddle_pos = provided_get_paddle_position(frames[-1])

    # Calculate the required joystick movement
    if ball_pos[1] > paddle_pos[1]:
        return 2  # Move paddle upward (rightward)
    elif ball_pos[1] < paddle_pos[1]:
        return 3  # Move paddle downward (leftward)
    else:
        return 0  # No action needed; ball is centered
