from generative_policy_proposals._policy_optimizer import optimize_policy, Args
from _tmp_kwai_keye_keye_vl_8b_preview_closed_loop import predict_next_action


args = Args(
    torch_deterministic=True,
    track=True,
    wandb_project_name="test_policy_optimizer",
    wandb_entity="narendasan",
    exp_name="test_policy_optimization_additional_features_normalized_extended_training_expanded_network",
    capture_video=True,
    save_model=True,
    env_id="ALE/Breakout-v5",
    num_envs=1,
    total_timesteps=10000000,
    learning_starts=80000,
    eval_frequency=100000,
    train_frequency=4,
)

optimize_policy(predict_next_action, args)
