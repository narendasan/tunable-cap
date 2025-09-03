import gymnasium

def generate_random_rollouts(env_name: str="ALE/Breakout-v5", num_episodes: int = 50, video_directory_prefix="videos/random_policy_rollouts_") -> None:
    env = gymnasium.make(env_name, render_mode="rgb_array")
    env = gymnasium.wrappers.RecordVideo(
        env,
        episode_trigger=lambda ep: ep % 10 == 0,
        video_folder=video_directory_prefix + env_name.split("/")[1],
        name_prefix="random_policy"
    )

    for e in range(num_episodes):
        obs, info = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

    env.close()
