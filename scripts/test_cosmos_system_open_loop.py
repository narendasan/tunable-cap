from typing import List
import ale_py
import gymnasium
from slugify import slugify
import logging
import os
from generative_policy_proposals._ControllerGenerator import ControllerGenerator, generate_and_load_policy, regenerate_policy
from generative_policy_proposals import action_spaces, generate_random_rollouts
from generative_policy_proposals.controller_utils.breakout_utilities import UTILITY_SPECS
from generative_policy_proposals.action_spaces._action_spaces import BREAKOUT_ACTION_SPACE

_LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

gymnasium.register_envs(ale_py)

MODEL = "nvidia/Cosmos-Reason1-7B"
ENV_NAME = "ALE/Breakout-v5"
ACTION_HISTORY_LEN = 50
NUM_FRAMES_FOR_OBS = 3
NUM_CODEGEN_RETRIES = 50
NUM_REFINEMENT_EPISODES = 5
action_space = BREAKOUT_ACTION_SPACE
actions = [[act["index"], act["description"]] for a, act in action_space.items()]
actions_str = str(actions)

generate_random_rollouts(env_name=ENV_NAME, num_episodes=50)

system_prompt = "You are a helpful assistant. Your job is to figure out how to play a game based on provided videos. Answer the question in the following format: <think>\nyour reasoning\n</think>\n\n<answer>\nyour answer\n</answer>"

actor = ControllerGenerator(
    model=MODEL,
    system_prompt=system_prompt
)

rollouts = ["./videos/random_policy_rollouts_Breakout-v5/random_policy-episode-{}.mp4".format(i) for i in range(0, 20, 10)]
visual_recognition_prompt = "Here are a few videos of the game breakout being played, ultimately we want to write some code to play the game. Start by identify all the key elements in this image. Be specific. Use at most 100 words." #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, visual_recognition_prompt)
_LOGGER.info(f"<PROMPT>: {visual_recognition_prompt}")
_LOGGER.info(f"<RESPONSE>: {visual_question_answer}")
#breakpoint()
strategy_recognition_prompt = "The video is no longer relevant, let's start developing a strategy to play the real game. Describe the ideal strategy if you were playing this game. Be specific. Use at most 100 words." #ATARI-GPT
strategy_recognition_prompt += "Your available actions are given below in the format `[action] - [description of action]`: " + actions_str
strategy_question_answer = actor.question_answering([], strategy_recognition_prompt)
_LOGGER.info(f"<PROMPT>: {strategy_recognition_prompt}")
_LOGGER.info(f"<RESPONSE>: {strategy_question_answer}")
#breakpoint()

# breakdown_prompt = """Given your analysis, decompose the different modules you
#     would need to build to implement your strategy to **play** the game and get the highest
#     score. Each module should have a clear single role. Present the design as a ordered
#     list of modules to be built followed by a detailed specification."""

# breakdown_question_answer = actor.question_answering(rollouts, breakdown_prompt)
# print(f"<PROMPT>: {breakdown_prompt}")
# print(f"<RESPONSE>: {breakdown_question_answer}")
#breakpoint()

code_prompt = "Given your plan, write the python code that plays the game to maximize the score. The code should take" \
    + str(NUM_FRAMES_FOR_OBS) \
    + """ frames of the game provided as a grayscale numpy array shaped
    Box(0, 255, (210, 160), np.uint8) for each frame as input and
    returns the action to take in the next time step.""" \
    + "Your available actions are given below in the format [[<action>, <description>]]: " + actions_str \
    + "For example to launch the ball to start the game or if the ball is not visible you should return `1`, to move the paddle left or right you should return `3` or `2` respectively. " \
    + """There should be a case for each action in your function. All of the code necessary must be written by you, you cannot
    call any external functions other than numpy or assume implementation. You cannot assume anything about the configuration of the game.
    No placeholder code or examples are allowed. You must be able to handle all possible game states. It is suggested that you use information across
    multiple frames to make your decisions. The entry point for your code should be `def predict_next_action(frames: np.ndarray) -> int:`"""

code_prompt += "You are provided the following utility functions to help you implement your policy: " + str(UTILITY_SPECS)
code_prompt += "Only generate python code, do not generate any comments"

predict_next_action, _ = generate_and_load_policy(actor, action_space, code_prompt, exp_name="open_loop")
assert(predict_next_action is not None)

def reset_env():
    env = gymnasium.make(ENV_NAME, render_mode="rgb_array", obs_type="grayscale")
    env = gymnasium.wrappers.RecordVideo(
        env,
        episode_trigger=lambda ep: ep % 1 == 0,
        video_folder="videos/open_loop_policy_rollouts_" + ENV_NAME.split("/")[1],
        name_prefix=f"refine_{slugify(MODEL).replace('-', '_')}_policy"
    )

    obs, info = env.reset()
    done = False
    step = 0
    score = 0
    obs_history = [obs]
    action_history = []
    lives = 5

    for i in range(NUM_FRAMES_FOR_OBS - 1):
        obs, reward, terminated, truncated, info = env.step(int(action_space["NOOP"]["index"]))
        action_history.append(int(action_space["NOOP"]["index"]))
        obs_history.append(obs)

    return (env, obs, done, step, score, lives, obs_history, action_history)

for r in range(NUM_REFINEMENT_EPISODES):
    env, obs, done, step, score, lives, obs_history, action_history = reset_env()

    while not done:
        action = 0
        try:
            action_ = predict_next_action(obs_history)
            _LOGGER.debug(action_)
            action = int(action_)
        except Exception as e:
            _LOGGER.error(f"Error occurred while predicting next action: {e}")
            regenerate_prompt = f"While executing your code, I hit to following error: {e}. Please try again."

            retries = NUM_CODEGEN_RETRIES
            while True:
                predict_next_action, _ = generate_and_load_policy(actor, action_space, regenerate_prompt, exp_name="open_loop")
                try:
                    action_ = predict_next_action(obs_history)
                    _LOGGER.debug(action_)
                    action = int(action_)
                    break
                except Exception as e:
                    _LOGGER.error(f"Error occurred while predicting next action: {e}")
                    regenerate_prompt = f"While executing your code, I hit to following error: {e}. Please try again."
                if retries == 0:
                    _LOGGER.error("Failed to regenerate code")
                    exit(-1)
                retries -= 1

        #action = env.action_space.sample()
        #print(int(action["index"]))
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward
        lives = info["lives"]
        _LOGGER.info(f"Step {step}:, Action: {action}, Score: {score}, Reward: {reward}, info: {info}")
        done = terminated or truncated
        step += 1
        action_history.append(action)
        if len(action_history) > ACTION_HISTORY_LEN:
            action_history.pop(0)

        obs_history.append(obs)
        if len(obs_history) > NUM_FRAMES_FOR_OBS:
            obs_history.pop(0)

        if len(action_history) == ACTION_HISTORY_LEN and all([action_history[0] == i for i in action_history]):
            env.close()
            _LOGGER.info(f"No meaningful change in actions: {action_history}")
            regenerate_strategy_prompt = f"I have been seeing your code run the same action too many times repeatedly, Your current score is {score}, you have {lives} lives left, the last few actions you took were {action_history}. Video of your code running is provided as well. What issues do you see?"
            predict_next_action, _ = regenerate_policy(actor, action_space, regenerate_strategy_prompt, [f"./videos/open_loop_policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-0.mp4"], code_prompt, exp_name="open_loop")
            # Restart the refinement episode
            env, obs, done, step, score, lives, obs_history, action_history = reset_env()
            _LOGGER.info("Restarting refinement episode")

    env.close()
    refine_prompt = f"Your current score is {score}, you have {lives} lives left, the last few actions you took were {action_history}. Video of your code running is provided as well. How can you improve this code?"
    predict_next_action, _ = regenerate_policy(actor, action_space, refine_prompt, [f"./videos/open_loop_policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-0.mp4"], code_prompt, exp_name="open_loop")
    print(f"Refinement Episode {r} completed.")
    breakpoint()
    os.rename(f"./videos/open_loop_policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-0.mp4", f"./policy_rollouts_Breakout-v5/refine_{slugify(MODEL).replace('-', '_')}_policy-episode-{r+10}.mp4")
exit()

evals = evaluate_policy(num_episodes=10)
print(evals)

rollouts = ["./policy_rollouts_Breakout-v5/eval_policy-episode-{}.mp4".format(i) for i in range(0, 10, 5)]
visual_recognition_prompt = "Here are some videos of breakout being played with your policy, what are the issues that you can see in the videos?" #ATARI-GPT
visual_question_answer = actor.question_answering(rollouts, visual_recognition_prompt)
print(f"<PROMPT>: {visual_recognition_prompt}")
print(f"<RESPONSE>: {visual_question_answer}")

code_prompt = "Regenerate the code to fix the issues you found in the videos"
policy = actor.generate_gameplay_code([], prompt=code_prompt)
print(policy)
