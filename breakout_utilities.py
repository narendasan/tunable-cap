from typing import List
import functools
import numpy as np
import inspect

def provided_crop_frame(frame: np.ndarray) -> np.ndarray:
    #vertical_crop_with_border = frame[17:-14]
    horizontal_crop = frame[:,8:-8]
    return horizontal_crop

def provided_is_ball_in_frame(frame: np.ndarray) -> bool:
    """
    Checks if the ball is in the frame.
    Args:
        frame (np.ndarray): The input frame
    Returns:
        bool: True if the ball is in the frame, False otherwise
    """
    vertical_crop = provided_crop_frame(frame)
    crop = vertical_crop[93:-21]
    in_frame = bool(np.any(crop))
    print(in_frame)
    return in_frame

def provided_get_ball_position(frame: np.ndarray) -> np.ndarray:
    """
    Returns an estimate of the ball position for a given frame
    Args:
        frame (np.ndarray): The input frame
    Returns:
        np.ndarray: The estimated ball position as a 2D array, if there is a ball in the frame, if not [nan, nan] will be returned
    """
    vertical_crop = provided_crop_frame(frame)
    crop = vertical_crop[93:-21]
    row_indices, col_indices = np.nonzero(crop)
    return np.mean(np.column_stack((row_indices, col_indices)), axis=0)


def provided_get_paddle_position(frame: np.ndarray) -> np.ndarray:
    """
    Returns an estimate of the paddle position for a given frame
    Args:
        frame (np.ndarray): The input frame
    Returns:
        np.ndarray: The estimated paddle position as a 2D array
    """
    vertical_crop = provided_crop_frame(frame)
    paddle_rows = vertical_crop[-21:-16]
    row_indices, col_indices = np.nonzero(paddle_rows)
    paddle = np.mean(np.column_stack((row_indices, col_indices)), axis=0)
    return paddle

def provided_frame_differences(frames: List[np.ndarray]) -> np.ndarray:
    """
    Overlays the changes between consecutive frames.
    Args:
        frames (List[np.ndarray]): The input frames
    Returns:
        np.ndarray: A numpy array that contains all the differences between consecutive frames
    """
    return functools.reduce(lambda a, b: a + (frames[0] != b), frames[1:], frames[0] != frames[1])

UTILITY_SPECS = {
    "provided_is_ball_in_frame": {"signature":inspect.signature(provided_is_ball_in_frame), "docstring": provided_is_ball_in_frame.__doc__},
    "provided_get_ball_position": {"signature":inspect.signature(provided_get_ball_position), "docstring": provided_get_ball_position.__doc__},
    "provided_get_paddle_position": {"signature":inspect.signature(provided_get_paddle_position), "docstring": provided_get_paddle_position.__doc__},
    "provided_frame_differences": {"signature":inspect.signature(provided_frame_differences), "docstring": provided_frame_differences.__doc__}
}
