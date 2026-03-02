import os
import sys
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from metadrive import MetaDriveEnv
from metadrive.component.sensors.rgb_camera import RGBCamera
import gymnasium as gym

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
PHASE2_MODEL_DIR  = "./models_vision/"
PHASE2_MODEL_NAME = "ppo_metadrive_vision"
REWARD_MODEL_DIR  = "./reward_model/"

PHASE6_LOG_DIR    = "./logs_rlhf/"
PHASE6_MODEL_DIR  = "./models_rlhf/"
PHASE6_MODEL_NAME = "ppo_rlhf"

TOTAL_TIMESTEPS   = 1_000_000
SAVE_FREQ         = 100_000
NUM_ENVS          = 2
SEGMENT_LENGTH    = 100
IMAGE_HEIGHT      = 84
IMAGE_WIDTH       = 84

METADRIVE_WEIGHT  = 0.4
RLHF_WEIGHT       = 0.6

MAX_TOKEN_LEN     = 256
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(PHASE6_LOG_DIR,   exist_ok=True)
os.makedirs(PHASE6_MODEL_DIR, exist_ok=True)

# ─────────────────────────────────────────────
#  GPT-2 REWARD MODEL
# ─────────────────────────────────────────────
class GPT2RewardModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gpt2       = GPT2Model.from_pretrained("gpt2")
        hidden_size     = self.gpt2.config.hidden_size
        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_ids, attention_mask):
        outputs     = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        score       = self.score_head(last_hidden).squeeze(-1)
        return score

# ─────────────────────────────────────────────
#  REWARD MODEL WRAPPER
#  Always runs on CPU inside subprocesses
#  GPU is reserved for PPO training only
# ─────────────────────────────────────────────
class RewardModelWrapper:
    def __init__(self, device="cpu"):
        self.device    = device
        self.tokenizer = GPT2Tokenizer.from_pretrained(REWARD_MODEL_DIR)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = GPT2RewardModel().to(self.device)

        model_path = os.path.join(REWARD_MODEL_DIR, "reward_model_best.pth")
        if not os.path.exists(model_path):
            model_path = os.path.join(REWARD_MODEL_DIR, "reward_model_final.pth")

        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    def score(self, description):
        encoding = self.tokenizer(
            description,
            max_length     = MAX_TOKEN_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )
        with torch.no_grad():
            s = self.model(
                encoding["input_ids"].to(self.device),
                encoding["attention_mask"].to(self.device),
            ).item()
        return s

# ─────────────────────────────────────────────
#  SEGMENT STATS TRACKER
# ─────────────────────────────────────────────
class SegmentTracker:
    def __init__(self, segment_length):
        self.segment_length = segment_length
        self.reset()

    def reset(self):
        self.speeds      = []
        self.lane_devs   = []
        self.steerings   = []
        self.rewards     = []
        self.crashed     = False
        self.out_of_road = False
        self.step_count  = 0

    def record(self, action, reward, info):
        self.speeds.append(float(info.get("velocity", 0.0)))
        self.lane_devs.append(abs(float(info.get("lateral_dist", 0.0))))
        self.steerings.append(abs(float(action[0])) if len(action) > 0 else 0.0)
        self.rewards.append(float(reward))

        if info.get("crash", False):
            self.crashed = True
        if info.get("out_of_road", False):
            self.out_of_road = True

        self.step_count += 1

    def is_full(self):
        return self.step_count >= self.segment_length

    def build_description(self):
        speeds     = np.array(self.speeds)
        lane_devs  = np.array(self.lane_devs)
        steerings  = np.array(self.steerings)

        avg_speed  = float(np.mean(speeds))
        avg_dev    = float(np.mean(lane_devs))
        smoothness = float(np.mean(np.abs(np.diff(steerings)))) if len(steerings) > 1 else 0.0
        total_rew  = float(np.sum(self.rewards))

        if avg_speed < 3.0:     speed_desc  = "very slow"
        elif avg_speed < 6.0:   speed_desc  = "moderate speed"
        elif avg_speed < 10.0:  speed_desc  = "good speed"
        else:                   speed_desc  = "very fast"

        if avg_dev < 0.1:       lane_desc   = "staying well within the lane"
        elif avg_dev < 0.3:     lane_desc   = "slight lane deviation"
        elif avg_dev < 0.6:     lane_desc   = "significant lane deviation"
        else:                   lane_desc   = "severe lane deviation"

        if smoothness < 0.05:   smooth_desc = "very smooth steering"
        elif smoothness < 0.15: smooth_desc = "mostly smooth steering"
        elif smoothness < 0.3:  smooth_desc = "somewhat jerky steering"
        else:                   smooth_desc = "very aggressive/jerky steering"

        incident = (
            "The agent crashed."          if self.crashed     else
            "The agent went out of road." if self.out_of_road else
            "No incidents."
        )

        return (
            f"The agent drove at {speed_desc} (avg {avg_speed:.1f} m/s) "
            f"with {lane_desc} (avg deviation {avg_dev:.2f}m). "
            f"Steering was {smooth_desc} (smoothness score {smoothness:.3f}). "
            f"{incident} "
            f"Total reward for this segment: {total_rew:.2f}."
        )

# ─────────────────────────────────────────────
#  RLHF ENV WRAPPER
# ─────────────────────────────────────────────
class RLHFEnvWrapper(gym.Wrapper):
    def __init__(self, env, reward_model, segment_length=100):
        super().__init__(env)
        self.reward_model   = reward_model
        self.segment_length = segment_length
        self.tracker        = SegmentTracker(segment_length)
        self.pending_rlhf   = 0.0

    def reset(self, **kwargs):
        self.tracker.reset()
        self.pending_rlhf = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, metadrive_reward, terminated, truncated, info = self.env.step(action)

        self.tracker.record(action, metadrive_reward, info)

        if self.tracker.is_full():
            description       = self.tracker.build_description()
            gpt2_score        = self.reward_model.score(description)  # CPU call
            self.pending_rlhf = gpt2_score / self.segment_length
            self.tracker.reset()

        combined_reward = (
            METADRIVE_WEIGHT * metadrive_reward +
            RLHF_WEIGHT      * self.pending_rlhf
        )

        return obs, combined_reward, terminated, truncated, info

# ─────────────────────────────────────────────
#  CNN + MLP EXTRACTOR  (same as Phase 2)
# ─────────────────────────────────────────────
class DrivingCNNMLP(BaseFeaturesExtractor):
    def __init__(self, observation_space, cnn_features=128, mlp_features=64):
        super().__init__(observation_space, features_dim=cnn_features + mlp_features)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
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
        img   = observations["image"][..., 0].permute(0, 3, 1, 2).float().contiguous()
        state = observations["state"]
        return torch.cat([self.cnn(img), self.mlp(state)], dim=1)

# ─────────────────────────────────────────────
#  ENV CONFIG
# ─────────────────────────────────────────────
def get_env_config(render=False, start_seed=0):
    return {
        "use_render":         render,
        "num_scenarios":      500,
        "start_seed":         start_seed,
        "map":                7,
        "random_lane_width":  True,
        "random_agent_model": False,
        "traffic_density":    0.2,
        "accident_prob":      0.2,
        "stack_size":         1,
        "image_observation":  True,
        "norm_pixel":         True,
        "sensors": {
            "rgb_camera": (RGBCamera, IMAGE_WIDTH, IMAGE_HEIGHT)
        },
        "vehicle_config": {
            "image_source": "rgb_camera"
        },
        "interface_panel":    [],
    }

# ─────────────────────────────────────────────
#  ENV FACTORY
#  Reward model is created INSIDE each subprocess
#  on CPU — avoids CUDA fork error
# ─────────────────────────────────────────────
def make_env(rank, render=False):
    def _init():
        # ✅ Created inside subprocess → no CUDA fork issue
        # ✅ Runs on CPU → GPU stays free for PPO
        reward_model = RewardModelWrapper(device="cpu")

        base_env = MetaDriveEnv(config=get_env_config(
            render     = render,
            start_seed = rank * 100,
        ))
        env = RLHFEnvWrapper(base_env, reward_model, SEGMENT_LENGTH)
        env = Monitor(env, os.path.join(PHASE6_LOG_DIR, f"env_{rank}"))
        return env
    return _init

# ─────────────────────────────────────────────
#  TRAIN
# ─────────────────────────────────────────────
def train():
    print(f"\n🚗 Phase 6: RLHF Fine-tuning")
    print(f"   Device          : {DEVICE}")
    print(f"   Parallel envs   : {NUM_ENVS}")
    print(f"   Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"   Reward mix      : {METADRIVE_WEIGHT}× MetaDrive + {RLHF_WEIGHT}× GPT-2")
    print(f"   Map             : random 7-block per episode")
    print(f"   Scenarios       : 500 (each env gets unique seed range)")
    print(f"   Reward model    : runs on CPU inside each subprocess\n")

    # ── Build parallel envs ───────────────────
    # Reward model is created inside each subprocess
    print("🌍 Building environments...")
    print(f"   env_0 → seeds   0 - 100")
    print(f"   env_1 → seeds 100 - 200")
    print(f"   ...and so on\n")

    vec_env = SubprocVecEnv(
        [make_env(rank=i) for i in range(NUM_ENVS)],
        start_method="fork"
    )
    vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True)

    # ── Load Phase 2 weights ──────────────────
    print("📂 Loading Phase 2 model as starting point...")
    phase2_path = os.path.join(PHASE2_MODEL_DIR, PHASE2_MODEL_NAME + "_final")

    policy_kwargs = dict(
        features_extractor_class  = DrivingCNNMLP,
        features_extractor_kwargs = dict(cnn_features=128, mlp_features=64),
        net_arch                  = [128, 64],
    )

    model = PPO(
        policy          = "MultiInputPolicy",
        env             = vec_env,
        policy_kwargs   = policy_kwargs,
        learning_rate   = 1e-4,
        n_steps         = 512,
        batch_size      = 256,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.1,
        ent_coef        = 0.005,
        verbose         = 1,
        tensorboard_log = PHASE6_LOG_DIR,
        device          = DEVICE,
    )

    if os.path.exists(phase2_path + ".zip"):
        model.set_parameters(
            PPO.load(phase2_path).get_parameters()
        )
        print(f"  ✅ Phase 2 weights loaded from: {phase2_path}")
    else:
        print(f"  ⚠️  Phase 2 model not found at {phase2_path}")
        print(f"      Training from scratch instead.")

    # ── Callbacks ─────────────────────────────
    checkpoint_callback = CheckpointCallback(
        save_freq   = SAVE_FREQ // NUM_ENVS,
        save_path   = PHASE6_MODEL_DIR,
        name_prefix = PHASE6_MODEL_NAME,
        verbose     = 1,
    )

    print(f"\n   Monitor live:")
    print(f"   tensorboard --logdir {PHASE6_LOG_DIR}")
    print(f"   watch -n 1 nvidia-smi\n")

    print("🚀 Starting RLHF fine-tuning...\n")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = checkpoint_callback,
        progress_bar    = True,
    )

    final_path = os.path.join(PHASE6_MODEL_DIR, PHASE6_MODEL_NAME + "_final")
    model.save(final_path)
    vec_env.save(os.path.join(PHASE6_MODEL_DIR, "vecnormalize_rlhf.pkl"))

    print(f"\n✅ RLHF fine-tuning complete!")
    print(f"   Model saved to : {final_path}")
    print(f"\n🚀 Ready for Phase 7 — Metrics + Visualization\n")

# ─────────────────────────────────────────────
#  EVALUATE
# ─────────────────────────────────────────────
def evaluate():
    from stable_baselines3.common.vec_env import DummyVecEnv

    print("\n🎮 Evaluating RLHF model...\n")

    vec_env   = DummyVecEnv([make_env(rank=0, render=True)])
    norm_path = os.path.join(PHASE6_MODEL_DIR, "vecnormalize_rlhf.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False

    final_path = os.path.join(PHASE6_MODEL_DIR, PHASE6_MODEL_NAME + "_final")
    model      = PPO.load(final_path, env=vec_env)

    obs             = vec_env.reset()
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
            episode     += 1
            obs          = vec_env.reset()
            if episode > 5:
                break

    print(f"\n  Average reward : {np.mean(episode_rewards):.2f}")
    vec_env.close()

# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "eval":
        evaluate()
    else:
        train()