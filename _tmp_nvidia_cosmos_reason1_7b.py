from breakout_utilities import *
import cv2
import numpy as np

def predict_next_action(frames):
    if not provided_is_ball_in_frame(frames[-1]):
        return 1

    ball_pos = provided_get_ball_position(frames[-1])
    paddle_pos = provided_get_paddle_position(frames[-1])
    diff = provided_frame_differences(frames)

    if ball_pos[0] == paddle_pos[0]:
        return 1

    if ball_pos[1] > paddle_pos[1] and diff[1][1] > 0:
        return 2
    elif ball_pos[1] < paddle_pos[1] and diff[1][1] < 0:
        return 3

    return 1