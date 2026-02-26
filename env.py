from metadrive import MetaDriveEnv

env = MetaDriveEnv(config={"use_render": True})
obs, info = env.reset()

print("✅ MetaDrive is working!")
print(f"Observation shape: {obs.shape}")

for i in range(200):
    action = env.action_space.sample()  # random actions
    print(f"Step {i+1}: Action: {action}")
    obs, reward, terminated, truncated, info = env.step(action)
    
    if terminated or truncated:
        obs, info = env.reset()

    print(f"Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

env.close()
print("Done!")