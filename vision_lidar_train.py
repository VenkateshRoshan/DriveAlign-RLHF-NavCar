import os
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera 

TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ       = 100_000
LOG_DIR         = "./logs_vision/"
MODEL_DIR       = "./models_vision/"
MODEL_NAME      = "ppo_metadrive_vision"

IMAGE_HEIGHT    = 84
IMAGE_WIDTH     = 84

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_env_config(render=False):
    return {
        "use_render":          render,
        "num_scenarios":       20,
        "traffic_density":     0.2,
        "accident_prob":       0.2,
        "map":                 "SCSCSCS",
        "random_lane_width":   True,
        "random_agent_model":  False,
        "stack_size":          1,

        "image_observation":   True,
        "norm_pixel":          True,
        "sensors": {
            "rgb_camera": (RGBCamera, IMAGE_WIDTH, IMAGE_HEIGHT)
        },
        "vehicle_config": {
            "image_source": "rgb_camera"
        },
        "interface_panel":     [],
    }

class DrivingCNNMLP(BaseFeaturesExtractor):
    """
    Two branches:
      CNN  → processes camera frames  (image)
      MLP  → processes lidar/state vector (state)
    Outputs are concatenated into one feature vector for PPO.
    """

    def __init__(self, observation_space, cnn_features=128, mlp_features=64):
        total_features = cnn_features + mlp_features
        super().__init__(observation_space, features_dim=total_features)
        
        # Input image: (B, H, W, C) → permuted to (B, C, H, W) in forward()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),   # → (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (64, 7, 7)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, cnn_features),
            nn.ReLU(),
        )

        lidar_dim = observation_space["state"].shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(lidar_dim, 128),
            nn.ReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, mlp_features),
            nn.ReLU(),
        )

    def forward(self, observations):
        img   = observations["image"]   # (B, H, W, C)
        state = observations["state"]   # (B, lidar_dim)
        
        img = img[..., 0]
        # Convert image to PyTorch format: (B, C, H, W)
        img = img.permute(0, 3, 1, 2).float()

        image_features = self.cnn(img)
        lidar_features = self.mlp(state)

        return torch.cat([image_features, lidar_features], dim=1)

def make_env(render=False):
    def _init():
        env = MetaDriveEnv(config=get_env_config(render=render))
        env = Monitor(env, LOG_DIR)
        return env
    return _init

def inspect():
    print("\n🔍 Inspecting vision + lidar observation...\n")
    env = MetaDriveEnv(config=get_env_config(render=True))
    obs, _ = env.reset()

    print(f"  Observation type   : {type(obs)}")
    print(f"  Observation keys   : {list(obs.keys())}")
    print(f"  Image shape        : {obs['image'].shape}")
    print(f"  State/lidar shape  : {obs['state'].shape}")
    print(f"  Image min/max      : {obs['image'].min():.3f} / {obs['image'].max():.3f}")
    print(f"  Action space       : {env.action_space}")
    print()

    total_reward = 0
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break

    print(f"  Episode length     : {info['episode_length']}")
    print(f"  Total reward       : {total_reward:.4f}")
    print(f"  Route completion   : {info['route_completion']:.4f}")
    print(f"  Crash              : {info['crash']}")
    print(f"  Out of road        : {info['out_of_road']}")
    print()
    env.close()

def train():
    vec_env = DummyVecEnv([make_env(render=False)])
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    policy_kwargs = dict(
        features_extractor_class  = DrivingCNNMLP,
        features_extractor_kwargs = dict(cnn_features=128, mlp_features=64),
        net_arch                  = [128, 64],
    )

    model = PPO(
        policy          = "MultiInputPolicy",
        env             = vec_env,
        policy_kwargs   = policy_kwargs,
        learning_rate   = 3e-4,
        n_steps         = 4096,
        batch_size      = 128,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        device          = "cuda" if torch.cuda.is_available() else "cpu",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
        verbose     = 1,
    )

    print("🚗 Starting Vision + Lidar Training...\n")
    print(f"   Device          : {'GPU ✅' if torch.cuda.is_available() else 'CPU'}")
    print(f"   Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"   Checkpoints     : every {SAVE_FREQ:,} steps → {MODEL_DIR}")
    print(f"   TensorBoard logs: {LOG_DIR}")
    print(f"\n   Monitor live:")
    print(f"   tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = checkpoint_callback,
        progress_bar    = True,
    )

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print(f"\n✅ Training complete. Model saved to: {final_path}")

def evaluate():
    print("\n🎮 Running evaluation with rendering...\n")

    vec_env = DummyVecEnv([make_env(render=True)])

    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model = PPO.load(final_path, env=vec_env)

    obs           = vec_env.reset()
    episode_rewards = []
    total_reward    = 0
    episode         = 1

    print(f"  Running Episode {episode}...")
    for _ in range(5000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]

        if done[0]:
            episode_rewards.append(total_reward)
            print(f"  Episode {episode} reward: {total_reward:.2f}")
            total_reward = 0
            episode += 1
            obs = vec_env.reset()
            if episode > 5:
                break

    print(f"\n  Average reward : {np.mean(episode_rewards):.2f}")
    vec_env.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    elif len(sys.argv) > 1 and sys.argv[1] == "inspect":
        inspect()
    else:
        train()