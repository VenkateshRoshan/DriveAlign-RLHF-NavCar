import os
import sys

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from metadrive import MetaDriveEnv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.helpers import ResNet18Extractor, N_STACK, SmoothnessWrapper, StateStackWrapper, get_env_config
from src.sim_recorder import VideoRecordingCallback

TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ       = 200_000
LOG_DIR         = "./logs_vision/"
MODEL_DIR       = "./models_vision/"
MODEL_NAME      = "ppo_metadrive_vision"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def make_env(render: bool = False):
    def _init():
        env = MetaDriveEnv(config=get_env_config(render=render))
        env = SmoothnessWrapper(env)
        env = StateStackWrapper(env, n_stack=N_STACK)
        env = Monitor(env, LOG_DIR)
        return env
    return _init

def inspect():
    print("\n Inspecting observation space...\n")

    raw_env    = MetaDriveEnv(config=get_env_config(render=False))
    env        = StateStackWrapper(raw_env, n_stack=N_STACK)
    obs, _     = env.reset()
    raw_obs, _ = raw_env.reset()
    raw_img    = raw_obs["image"]
    raw_state  = raw_obs["state"]

    print("Image: ")
    print(f"  raw image shape      : {raw_img.shape}")
    print(f"  raw image dtype      : {raw_img.dtype}")
    print(f"  raw image min/max    : {raw_img.min():.3f} / {raw_img.max():.3f}")

    print("\nState: ")
    print(f"  raw state dim (1 step)     : {raw_state.shape[0]}")
    print(f"  stacked state dim (×{N_STACK})  : {obs['state'].shape[0]}")
    print(f"\n  Raw state values:")
    print(f"  {'idx':<5} {'value':>10}   signal (approx)")
    print(f"  {'-'*50}")

    known = {
        0: "lateral offset         (lane centre deviation)",
        1: "heading error          (angle to road dir)",
        2: "speed longitudinal     (forward velocity)",
        3: "speed lateral          (sideways velocity)",
        4: "steering angle",
        5: "yaw rate               (angular velocity)",
    }
    for i, v in enumerate(raw_state):
        label = known.get(i, "(lidar / nav / boundary)")
        print(f"  [{i:3d}] {v:>10.4f}   {label}")

    print(f"\nAction space: {env.action_space}\n")

    total_reward = 0
    steps = 0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if term or trunc:
            break

    print(f"Episode stats:")
    print(f"  steps          : {steps}")
    print(f"  total reward   : {total_reward:.4f}")
    print(f"  route complet. : {info.get('route_completion', 'N/A')}")
    print(f"  crash          : {info.get('crash', 'N/A')}")
    print(f"  out of road    : {info.get('out_of_road', 'N/A')}\n")

    env.close()
    raw_env.close()

def train():
    vec_env = DummyVecEnv([make_env(render=False)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,
        norm_reward = True,
        clip_reward = 10.0,
    )

    policy_kwargs = dict(
        features_extractor_class  = ResNet18Extractor,   # ← ResNet18 frozen backbone
        features_extractor_kwargs = dict(
            cnn_features   = 256,
            mlp_features   = 128,
            fusion_hidden  = 256,
            fused_features = 256,
        ),
        net_arch = [256, 128],
    )

    model = PPO(
        policy          = "MultiInputPolicy",
        env             = vec_env,
        policy_kwargs   = policy_kwargs,

        learning_rate   = 1e-4,
        n_steps         = 512,
        batch_size      = 64,
        n_epochs        = 8,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.15,
        ent_coef        = 0.003,
        vf_coef         = 0.5,
        max_grad_norm   = 0.5,
        verbose         = 1,
        tensorboard_log = LOG_DIR,
        device          = "cuda" if torch.cuda.is_available() else "cpu",
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
        verbose     = 1,
    )
    video_cb = VideoRecordingCallback(record_freq=100_000, include_step_zero=True)

    print("Starting Phase 2 Training\n")
    print(f"  Device           : {'GPU' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Backbone         : ResNet18 (frozen, pretrained ImageNet)")
    print(f"  Resolution       : 224×224")
    print(f"  Total timesteps  : {TOTAL_TIMESTEPS:,}")
    print(f"  State stack      : {N_STACK} steps")
    print(f"  Traffic          : {get_env_config()['traffic_density']:.0%} density")
    print(f"  ent_coef         : {model.ent_coef}")
    print(f"  learning_rate    : {model.learning_rate}")
    print(f"  clip_range       : {model.clip_range}")
    print(f"  n_steps          : {model.n_steps}")
    print(f"  batch_size       : {model.batch_size}")
    print(f"\n  TensorBoard: tensorboard --logdir {LOG_DIR}\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, video_cb],
        progress_bar    = True,
    )

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print(f"\nTraining complete. Saved to: {final_path}")

def continue_train(extra_timesteps: int = 500_000):
    vec_env   = DummyVecEnv([make_env(render=False)])
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model = PPO.load(final_path, env=vec_env, device="cuda" if torch.cuda.is_available() else "cpu")

    print(f"Resuming training for {extra_timesteps:,} more steps...\n")

    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME + "_continued",
        verbose     = 1,
    )

    model.learn(
        total_timesteps     = extra_timesteps,
        callback            = checkpoint_cb,
        progress_bar        = True,
        reset_num_timesteps = False,
    )

    save_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model.save(save_path)
    vec_env.save(norm_path)
    print(f"\nContinued training complete. Saved to: {save_path}")


def evaluate():
    print("\n Running evaluation...\n")

    vec_env   = DummyVecEnv([make_env(render=True)])
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
    else:
        print("  No VecNormalize found - running without normalisation")

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model = PPO.load(final_path, env=vec_env)

    obs             = vec_env.reset()
    episode_rewards = []
    episode_lengths = []
    total_reward    = 0.0
    steps           = 0
    episode         = 1
    MAX_EPISODES    = 5

    print(f"  Episode {episode}...")
    for _ in range(10_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps        += 1

        if done[0]:
            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
            print(f"  Episode {episode}"
                  f"  reward: {total_reward:7.2f}"
                  f"  steps: {steps:5d}"
                  f"  route: {info[0].get('route_completion', 0):.2%}"
                  f"  crash: {info[0].get('crash', False)}")
            total_reward = 0.0
            steps        = 0
            episode     += 1
            obs          = vec_env.reset()
            if episode > MAX_EPISODES:
                break

    print(f"\n  Avg reward  : {np.mean(episode_rewards):.2f}")
    print(f"  Avg length  : {np.mean(episode_lengths):.0f} steps")
    vec_env.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    {"inspect": inspect, "eval": evaluate, "continue": continue_train}.get(mode, train)()
