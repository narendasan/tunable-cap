from generative_policy_proposals._policy_optimizer import optimize_policy, Args
from _tmp_kwai_keye_keye_vl_8b_preview_closed_loop import predict_next_action


args = Args(
    torch_deterministic=True,
    track=True,
    wandb_project_name="test_policy_optimizer",
    wandb_entity="narendasan",
    capture_video=True,
    save_model=True,
    env_id="ALE/Breakout-v5",
    n_atoms=10,
)

optimize_policy(predict_next_action, args)
