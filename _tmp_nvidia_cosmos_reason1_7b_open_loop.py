from typing import Optional, Tuple
from breakout_utilities import *
import numpy as np
from typing import List

def predict_next_action(frames: List[np.ndarray]) -> int:
    ball_in_frame = provided_is_ball_in_frame(frames[-1])
    if not ball_in_frame:
        return 1
    ball_pos = provided_get_ball_position(frames[-1])
    paddle_pos = provided_get_paddle_position(frames[-1])
    if ball_pos[0] < paddle_pos[0]:
        return 3
    elif ball_pos[0] > paddle_pos[0]:
        return 2
    else:
        return 1