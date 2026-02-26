from metadrive import MetaDriveEnv

env = MetaDriveEnv(config={"use_render": True})
obs, info = env.reset()

print("=== OBSERVATION SPACE ===")
print(env.observation_space)
print(f"Shape: {env.observation_space.shape}")

print("\n=== ACTION SPACE ===")
print(env.action_space)
print(f"Shape: {env.action_space.shape}")
print(f"Low:  {env.action_space.low}")
print(f"High: {env.action_space.high}")

# print("\n=== SAMPLE OBSERVATION ===")
# print(obs)
# print(f"Min value: {obs.min():.4f}")
# print(f"Max value: {obs.max():.4f}")

print("\n=== SAMPLE REWARD + INFO ===")
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward}")
print(f"Info keys: {list(info.keys())}")
print(f"Info: {info}")

env.close()