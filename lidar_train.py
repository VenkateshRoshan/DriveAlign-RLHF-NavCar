import os
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from metadrive import MetaDriveEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ       = 100_000
LOG_DIR         = "./logs/"
MODEL_DIR       = "./models/"
MODEL_NAME      = "ppo_metadrive_phase1"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

class DrivingMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        
        input_dim = observation_space.shape[0]  # lidar/sensor vector size
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            # nn.BatchNorm1d(256),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )
    
    def forward(self, obs):
        return self.network(obs)

def make_env():
    def _init():
        env = MetaDriveEnv(config={
            "use_render":              True,   # headless = faster training
            "num_scenarios":           20,      # variety of tracks
            "traffic_density":         0.2,     # no traffic in phase 1
            "accident_prob":           0.2,
            "map":                     "SCSCSCS",  # simple straight roads
            "random_lane_width":       True,
            "random_agent_model":      False,
            # "vehicle_config": {
            #     "use_saver":           False,
            # },
        })
        env = Monitor(env, LOG_DIR)  # wraps env to log episode stats
        return env
    return _init

def inspect_reward():
    print("\n🔍 Inspecting reward signal for 1 episode...\n")
    env = MetaDriveEnv(config={"use_render": False})
    obs, _ = env.reset()
    total_reward = 0
    for step in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    print(f"  Episode length : {info['episode_length']}")
    print(f"  Total reward   : {total_reward:.4f}")
    print(f"  Route completion: {info['route_completion']:.4f}")
    print(f"  Crash          : {info['crash']}")
    print(f"  Out of road    : {info['out_of_road']}")
    print(f"  Final velocity : {info['velocity']:.4f}")
    print()
    env.close()

def train():
    vec_env = DummyVecEnv([make_env()])
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True)

    # # Eval environment (separate, not normalized by same stats)
    # eval_env = DummyVecEnv([make_env()])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)
    
    policy_kwargs = dict(
        features_extractor_class  = DrivingMLP,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = [64, 64],  # layers AFTER your extractor → action
    )

    # PPO Agent
    model = PPO(
        policy          = "MlpPolicy",   # vector obs → MLP → actions
        env             = vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate   = 3e-4,
        n_steps         = 4096,          # steps per env before update
        batch_size      = 128,
        n_epochs        = 10,
        gamma           = 0.99,          # discount factor
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,          # encourages exploration
        verbose         = 1,
        tensorboard_log = LOG_DIR,
    )

    # Save a checkpoint every SAVE_FREQ steps
    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
        verbose     = 1,
    )

    # # Evaluate agent every 50k steps, save best model
    # eval_callback = EvalCallback(
    #     eval_env,
    #     best_model_save_path = os.path.join(MODEL_DIR, "best"),
    #     log_path             = LOG_DIR,
    #     eval_freq            = 50_000,
    #     n_eval_episodes      = 5,
    #     deterministic        = True,
    #     verbose              = 1,
    # )

    print("🚗 Starting Phase 1 Training...\n")
    print(f"   Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"   Checkpoints     : every {SAVE_FREQ:,} steps → {MODEL_DIR}")
    print(f"   TensorBoard     : {LOG_DIR}")
    print(f"\n   Run this to monitor:")
    print(f"   tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        # callback        = [checkpoint_callback, eval_callback],
        callback        = checkpoint_callback,
        progress_bar    = True,
    )

    # Save final model + normalization stats
    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print(f"\n✅ Training complete. Model saved to: {final_path}")

def evaluate():
    from stable_baselines3.common.vec_env import VecNormalize
    import pickle

    print("\n🎮 Running evaluation with rendering...\n")

    vec_env = DummyVecEnv([lambda: Monitor(MetaDriveEnv(config={
        "use_render":      True,
        "num_scenarios":   10,
        "traffic_density": 0.0,
    }), LOG_DIR)])

    # Load normalization stats from training
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False

    # Load best model
    best_path = os.path.join(MODEL_DIR, "best", "best")
    model = PPO.load(best_path, env=vec_env)

    obs = vec_env.reset()
    episode_rewards = []
    total_reward = 0
    episode = 1

    print(f"  Running Episode {episode}...")
    for _ in range(3000):
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

    print(f"\n  Average reward over {len(episode_rewards)} episodes: {np.mean(episode_rewards):.2f}")
    vec_env.close()

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        inspect_reward()
        train()