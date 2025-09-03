from typing import Optional
from breakout_utilities import *
import numpy as np
from typing import List

def predict_next_action(frames: List[np.ndarray], weights: np.ndarray, state: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Predicts the next action in the Breakout game based on the provided frames, weights, and state.

    :param frames: A list of the current game frames (grayscale, shape (210, 160)) representing the game state.
    :param weights: A 2D weight array (shape (1, 10)) used to influence the prediction of actions.
    :param state: A previous state array (empty or initialized) that may contain information from past actions.

    :return: A tuple containing the predicted action (int) and an updated state array (np.ndarray).
    """
    # Ensure frames are processed correctly
    if len(frames) != 3:
        raise ValueError("Exactly three frames are required for prediction.")

    # Extract relevant data from frames
    paddle_positions = [provided_get_paddle_position(frame) for frame in frames]
    ball_positions = [provided_get_ball_position(frame) for frame in frames]
    print(ball_positions)

    # Check for ball presence
    if not any(ball_positions):
        return 1
        raise ValueError("No ball detected in the frames.")

    # Estimate ball trajectory
    ball_trajectory = []
    for i in range(1, 3):
        ball_position = ball_positions[i]
        ball_velocity = (ball_position - ball_positions[i - 1]) / 25
        ball_trajectory.append(ball_position + ball_velocity)

    # Predict paddle movement
    paddle_target = ball_trajectory[-1]
    paddle_current = paddle_positions[-1]
    direction = 3 if paddle_target[0] > paddle_current[0] else 2 if paddle_target[0] < paddle_current[0] else 0
    if direction == 0:
        predicted_action = 1  # Fire button to launch the ball
    else:
        predicted_action = direction  # Move paddle left/right to intercept

    # Update state with prediction outcome
    state = np.append(state, [predicted_action], axis=0)
    return predicted_action, state
