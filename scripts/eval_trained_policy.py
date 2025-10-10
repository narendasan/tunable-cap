from generative_policy_proposals._eval import evaluate
from generative_policy_proposals._policy_optimizer import (
    optimize_policy,
    Args,
    make_env,
    QNetwork,
)
from _tmp_kwai_keye_keye_vl_8b_preview_closed_loop import predict_next_action
import torch
import functools
import time
import gymnasium as gym
import numpy as np

args = Args(
    torch_deterministic=True,
    track=True,
    wandb_project_name="test_policy_optimizer",
    wandb_entity="narendasan",
    exp_name="test_policy_h_tuning_increased_lr",
    capture_video=True,
    save_model=True,
    env_id="ALE/Breakout-v5",
    num_envs=1,
    total_timesteps=1_000_000,
    learning_starts=5000,
    eval_frequency=10000,
    train_frequency=10,
    batch_size=16,
    exploration_fraction=0.3,
    learning_rate=1e-3,
)


q_net_state_dict = torch.load(
    "runs/ALE/Breakout-v5__test_policy_h_tuning_increased_lr__1__1760042382/test_policy_h_tuning_increased_lr.cleanrl_model"
)

envs = gym.vector.SyncVectorEnv(
    [
        make_env(
            "ALE/Breakout-v5",
            args.seed + i,
            i,
            False,
            f"{i}",
        )
        for i in range(args.num_envs)
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
q_network = QNetwork(envs).to(device)
q_network.load_state_dict(q_net_state_dict)
q_policy = functools.partial(predict_next_action, model=q_network)

episodic_returns = evaluate(
    make_env,
    args.env_id,
    eval_episodes=10,
    run_name=f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}-eval",
    policy=q_policy,
    epsilon=args.end_e,
    capture_video=args.capture_video,
)

print(f"Mean episodic return: {np.mean(episodic_returns)}")
