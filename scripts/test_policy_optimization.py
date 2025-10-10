from generative_policy_proposals._policy_optimizer import optimize_policy, Args
from _tmp_kwai_keye_keye_vl_8b_preview_closed_loop import predict_next_action


args = Args(
    torch_deterministic=True,
    track=True,
    wandb_project_name="test_policy_optimizer",
    wandb_entity="narendasan",
    exp_name="test_policy_h_tuning_lr_5e-2",
    capture_video=True,
    save_model=True,
    env_id="ALE/Breakout-v5",
    num_envs=1,
    total_timesteps=1_000_000,
    learning_starts=5000,
    eval_frequency=10000,
    train_frequency=5,
    batch_size=16,
    exploration_fraction=0.2,
    learning_rate=5e-3,
)

optimize_policy(predict_next_action, args)
