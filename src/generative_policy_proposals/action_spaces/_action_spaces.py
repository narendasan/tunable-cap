import os
from generative_policy_proposals.action_spaces._load_action_spaces import load_action_space_dict

BREAKOUT_ACTION_SPACE = load_action_space_dict(f"{os.path.join(os.path.dirname(__file__), 'Breakout-v5-action_space.csv')}")
TETRIS_ACTION_SPACE = load_action_space_dict(f"{os.path.join(os.path.dirname(__file__), 'Tetris-v5-action_space.csv')}")
