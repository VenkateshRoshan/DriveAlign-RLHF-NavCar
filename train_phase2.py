"""
train_phase2.py  -  DriveAlign Phase 2: Vision + Lidar Agent
=============================================================================

Architecture:
  ┌─────────────────────────────────────┐   ┌─────────────────────────────┐
  │         STATE BRANCH                │   │       IMAGE BRANCH          │
  │   (loaded from Phase 1 weights)     │   │   (ResNet18, frozen)        │
  │                                     │   │                             │
  │  ego  → 32                          │   │  RGB (224×224)              │
  │  lidar→ 128   → state_fusion → 128  │   │  → backbone → 512          │
  │  side → 16                          │   │  → cnn_proj  → 256         │
  │  lane → 16                          │   │                             │
  │  nav  → 32                          │   │                             │
  └────────────────┬────────────────────┘   └──────────────┬─────────────┘
                   └──────────────┬──────────────────────────┘
                                  │  concat (384-dim)
                              gated fusion
                                  │
                             final_fusion → 256-dim
                                  │
                            Policy Head (PPO)

Transfer schedule:
  Steps 0 – 200k  : state branch FROZEN  (Phase 1 weights locked)
                    only image branch + fusion trains
                    LR = 1e-4
  Steps 200k+     : state branch UNFROZEN (all weights train jointly)
                    LR drops to 5e-5

Usage:
  python train_phase2.py              # train from scratch (loads Phase 1)
  python train_phase2.py inspect      # inspect obs space
  python train_phase2.py eval         # evaluate with rendering
  python train_phase2.py continue     # resume from latest checkpoint
"""

import os
import sys
import re
import glob

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from metadrive import MetaDriveEnv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "config"))

from config import Config
from src.helpers import SmoothnessWrapper, StateStackWrapper, FullStateWrapper, get_env_config
from src.sim_recorder import VideoRecordingCallback

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING CONFIG
# ─────────────────────────────────────────────────────────────────────────────

TOTAL_TIMESTEPS  = 2_000_000
SAVE_FREQ        =   200_000
UNFREEZE_AT_STEP =   200_000   # freeze state branch until this step
N_ENVS           = 2           # parallel envs (reduced vs Phase 1 — image obs is heavier)
N_STACK          = 1           # same as Phase 1: no stacking, single state vector

LOG_DIR          = "./logs_phase2/"
MODEL_DIR        = "./models_phase2/"
MODEL_NAME       = "ppo_phase2"

# Phase 1 model to load state branch weights from
PHASE1_MODEL_DIR = "./models_phase1/"
PHASE1_MODEL_NAME = "ppo_phase1_final"

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT
# ─────────────────────────────────────────────────────────────────────────────

def make_env(rank: int = 0, render: bool = False):
    """
    Phase 2 env factory.

    Wrapper stack (inside → outside):
        MetaDriveEnv                  raw image + 24-dim state
        FullStateWrapper              reconstructs full 264-dim state from lidar sensor
        SmoothnessWrapper             penalises steering jerk
        Monitor                       logs episode stats
    """
    def _init():
        cfg = get_env_config(render=render, start_seed=rank * 100)
        env = MetaDriveEnv(config=cfg)
        env = FullStateWrapper(env)      # ← restores 264-dim state for Phase 1 transfer
        env = SmoothnessWrapper(env)
        env = Monitor(env, LOG_DIR)
        return env
    return _init


# ─────────────────────────────────────────────────────────────────────────────
# PHASE 2 FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

class Phase2Extractor(BaseFeaturesExtractor):
    """
    Combined image + state extractor for Phase 2.

    State branch:
        Exact replica of Phase 1's MultiHeadExtractor heads.
        Weights are loaded from the saved Phase 1 model.
        Frozen for the first UNFREEZE_AT_STEP timesteps so the
        CNN can warm up without corrupting Phase 1's knowledge.

    Image branch:
        ResNet18 pretrained backbone (frozen — no gradients through it).
        Projection head trains to map 512-dim backbone output to 256-dim.

    Fusion:
        Gated fusion: each branch gates the other before combining.
        Final linear maps to fused_features (256-dim).
    """

    def __init__(
        self,
        observation_space,
        cnn_features:    int = 256,
        state_features:  int = 128,   # must match Phase 1 features_dim
        fusion_hidden:   int = 256,
        fused_features:  int = 256,
    ):
        super().__init__(observation_space, features_dim=fused_features)

        state_dim = observation_space["state"].shape[0]

        # ── State branch: same heads as Phase 1 MultiHeadExtractor ───────────
        # These weights will be loaded from the Phase 1 checkpoint.
        self.ego_head = nn.Sequential(
            nn.Linear(Config.EGO_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        self.lidar_head = nn.Sequential(
            nn.Linear(Config.LIDAR_DIM, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
        )
        self.side_head = nn.Sequential(
            nn.Linear(Config.SIDE_DIM, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.lane_head = nn.Sequential(
            nn.Linear(Config.LANE_DIM, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )
        self.nav_head = nn.Sequential(
            nn.Linear(Config.NAV_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )
        # Phase 1 fusion: 224-dim → 256 → 128 (state_features)
        state_combined_dim = 32 + 128 + 16 + 16 + 32  # = 224
        self.state_fusion = nn.Sequential(
            nn.Linear(state_combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, state_features),
            nn.LayerNorm(state_features),
            nn.ReLU(),
        )

        # ── Image branch: ResNet18 frozen backbone ────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False   # backbone stays frozen throughout

        self.cnn_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, cnn_features),
            nn.LayerNorm(cnn_features),
            nn.ReLU(),
        )

        # ImageNet normalisation (ResNet expects these)
        self.register_buffer(
            "img_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "img_std",  torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        # ── Gated fusion ──────────────────────────────────────────────────────
        combined_dim = cnn_features + state_features   # 256 + 128 = 384
        self.image_gate = nn.Sequential(
            nn.Linear(combined_dim, cnn_features),
            nn.Sigmoid(),
        )
        self.state_gate = nn.Sequential(
            nn.Linear(combined_dim, state_features),
            nn.Sigmoid(),
        )
        self.final_fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fusion_hidden, fused_features),
            nn.LayerNorm(fused_features),
            nn.ReLU(),
        )

    # ── weight transfer ───────────────────────────────────────────────────────

    def load_phase1_weights(self, phase1_path: str):
        """
        Load Phase 1's MultiHeadExtractor weights into the state branch.
        Must be called AFTER the PPO model is created (so the extractor exists).
        """
        device = next(self.parameters()).device
        phase1 = PPO.load(phase1_path, device=device)
        ext    = phase1.policy.features_extractor

        self.ego_head.load_state_dict(ext.ego_head.state_dict())
        self.lidar_head.load_state_dict(ext.lidar_head.state_dict())
        self.side_head.load_state_dict(ext.side_head.state_dict())
        self.lane_head.load_state_dict(ext.lane_head.state_dict())
        self.nav_head.load_state_dict(ext.nav_head.state_dict())
        self.state_fusion.load_state_dict(ext.fusion.state_dict())

        del phase1   # free memory
        print("  ✅ Phase 1 state branch weights loaded into Phase 2 extractor.")

    # ── freeze / unfreeze ─────────────────────────────────────────────────────

    @property
    def _state_modules(self):
        return [
            self.ego_head, self.lidar_head, self.side_head,
            self.lane_head, self.nav_head, self.state_fusion,
        ]

    def freeze_state_branch(self):
        for m in self._state_modules:
            for p in m.parameters():
                p.requires_grad = False
        print("  🔒 State branch frozen.")

    def unfreeze_state_branch(self):
        for m in self._state_modules:
            for p in m.parameters():
                p.requires_grad = True
        print("  🔓 State branch unfrozen — all weights now training.")

    # ── forward ───────────────────────────────────────────────────────────────

    def forward(self, observations: dict) -> torch.Tensor:
        # ── state branch ─────────────────────────────────────────────────────
        state = observations["state"].float()

        ego   = state[:, Config.EGO_START   : Config.EGO_START   + Config.EGO_DIM  ]
        lidar = state[:, Config.LIDAR_START : Config.LIDAR_START + Config.LIDAR_DIM]
        side  = state[:, Config.SIDE_START  : Config.SIDE_START  + Config.SIDE_DIM ]
        lane  = state[:, Config.LANE_START  : Config.LANE_START  + Config.LANE_DIM ]
        nav   = state[:, Config.NAV_START   : Config.NAV_START   + Config.NAV_DIM  ]

        state_feat = self.state_fusion(torch.cat([
            self.ego_head(ego),
            self.lidar_head(lidar),
            self.side_head(side),
            self.lane_head(lane),
            self.nav_head(nav),
        ], dim=1))

        # ── image branch ─────────────────────────────────────────────────────
        img = observations["image"]
        while img.dim() > 4:
            img = img[..., -1]                           # collapse stack dim
        img = img.permute(0, 3, 1, 2).float()            # (B, C, H, W)
        img = (img - self.img_mean) / self.img_std       # ImageNet normalise

        with torch.no_grad():
            feat = self.backbone(img)                    # (B, 512, 1, 1)
        img_feat = self.cnn_proj(feat)                   # (B, cnn_features)

        # ── gated fusion ─────────────────────────────────────────────────────
        combined    = torch.cat([img_feat, state_feat], dim=1)
        gated_img   = img_feat   * self.image_gate(combined)
        gated_state = state_feat * self.state_gate(combined)
        fused       = torch.cat([gated_img, gated_state], dim=1)

        return self.final_fusion(fused)


# ─────────────────────────────────────────────────────────────────────────────
# UNFREEZE CALLBACK
# ─────────────────────────────────────────────────────────────────────────────

class UnfreezeStateCallback(BaseCallback):
    """
    At UNFREEZE_AT_STEP timesteps:
      - Unfreezes the state branch of Phase2Extractor
      - Drops the learning rate from 1e-4 to 5e-5
    """

    def __init__(self, unfreeze_at: int = UNFREEZE_AT_STEP, new_lr: float = 5e-5):
        super().__init__()
        self.unfreeze_at = unfreeze_at
        self.new_lr      = new_lr
        self._done       = False

    def _on_step(self) -> bool:
        if not self._done and self.num_timesteps >= self.unfreeze_at:
            ext = self.model.policy.features_extractor
            ext.unfreeze_state_branch()

            # update LR in the optimiser directly
            for pg in self.model.policy.optimizer.param_groups:
                pg["lr"] = self.new_lr
            self.model.learning_rate = self.new_lr

            print(f"\n  Step {self.num_timesteps:,}: state branch unfrozen, LR → {self.new_lr}")
            self._done = True
        return True


# ─────────────────────────────────────────────────────────────────────────────
# MODES
# ─────────────────────────────────────────────────────────────────────────────

def inspect():
    print("\n── Phase 2 Observation Space Inspection ─────────────────────────────\n")

    cfg = get_env_config(render=False)
    env = MetaDriveEnv(config=cfg)
    env = FullStateWrapper(env)
    env = SmoothnessWrapper(env)

    obs, _ = env.reset()

    img_shape   = obs["image"].shape
    state_shape = obs["state"].shape

    print(f"  Image  shape : {img_shape}   (H × W × C)")
    print(f"  State  shape : {state_shape}")
    print(f"  N_STACK      : {N_STACK}")
    print(f"  State dim    : {state_shape[0]}")

    if state_shape[0] == Config.EGO_DIM + Config.LIDAR_DIM + Config.SIDE_DIM + Config.LANE_DIM + Config.NAV_DIM:
        print(f"\n  ✅ State dim matches Phase 1 ({state_shape[0]}-dim) — full weight transfer possible.")
    else:
        print(f"\n  ⚠  State dim {state_shape[0]} ≠ expected "
              f"{Config.EGO_DIM + Config.LIDAR_DIM + Config.SIDE_DIM + Config.LANE_DIM + Config.NAV_DIM}.")
        print(f"     Check Config slice constants.")

    print(f"\n  Action space : {env.action_space}\n")
    env.close()
    print("── Done ─────────────────────────────────────────────────────────────\n")


def train():
    # ── build envs ────────────────────────────────────────────────────────────
    vec_env = SubprocVecEnv([make_env(rank=i, render=False) for i in range(N_ENVS)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = False,   # image obs — don't normalise pixels
        norm_reward = True,
        clip_reward = 10.0,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    policy_kwargs = dict(
        features_extractor_class  = Phase2Extractor,
        features_extractor_kwargs = dict(
            cnn_features   = 256,
            state_features = 128,
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

    # ── load Phase 1 weights into state branch ────────────────────────────────
    phase1_path = os.path.join(PHASE1_MODEL_DIR, PHASE1_MODEL_NAME)
    if os.path.exists(phase1_path + ".zip"):
        model.policy.features_extractor.load_phase1_weights(phase1_path)
        model.policy.features_extractor.freeze_state_branch()
    else:
        print(f"  ⚠  Phase 1 model not found at {phase1_path}.")
        print(f"     Training Phase 2 from scratch (no transfer).")

    # ── callbacks ─────────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
        verbose     = 1,
    )
    unfreeze_cb = UnfreezeStateCallback(
        unfreeze_at = UNFREEZE_AT_STEP,
        new_lr      = 5e-5,
    )
    video_cb = VideoRecordingCallback(record_freq=100_000, include_step_zero=True)

    total_params    = sum(p.numel() for p in model.policy.parameters())
    trainable_start = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    print("\n── Phase 2 Training ─────────────────────────────────────────────────")
    print(f"  Device          : {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable now   : {trainable_start:,}  (state branch frozen)")
    print(f"  Unfreeze at     : {UNFREEZE_AT_STEP:,} steps")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Parallel envs   : {N_ENVS}")
    print(f"  N_STACK         : {N_STACK}")
    print(f"  TensorBoard     : tensorboard --logdir {LOG_DIR}")
    print(f"─────────────────────────────────────────────────────────────────────\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = [checkpoint_cb, unfreeze_cb, video_cb],
        progress_bar    = True,
    )

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print(f"\n✅ Training complete. Saved to: {final_path}")


def continue_train(extra_timesteps: int = None):
    pattern    = os.path.join(MODEL_DIR, f"{MODEL_NAME}_*_steps.zip")
    ckpt_files = glob.glob(pattern)
    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")

    if not ckpt_files and not os.path.exists(final_path + ".zip"):
        print("  No checkpoint found. Run train() first.")
        return

    def _extract_steps(fp):
        m = re.search(r"_(\d+)_steps\.zip$", fp)
        return int(m.group(1)) if m else -1

    if ckpt_files:
        latest     = max(ckpt_files, key=_extract_steps)
        steps_done = _extract_steps(latest)
        load_path  = latest.replace(".zip", "")
        print(f"\n  Latest checkpoint : {os.path.basename(latest)}")
        print(f"  Steps completed   : {steps_done:,}")
    else:
        load_path  = final_path
        steps_done = 0
        print("\n  Loading _final model.")

    remaining = extra_timesteps if extra_timesteps is not None else max(0, TOTAL_TIMESTEPS - steps_done)
    if remaining <= 0:
        print(f"  Already at {steps_done:,} / {TOTAL_TIMESTEPS:,} steps.")
        return

    print(f"  Remaining steps   : {remaining:,}\n")

    vec_env   = SubprocVecEnv([make_env(rank=i) for i in range(N_ENVS)])
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = True
        vec_env.norm_reward = True
    else:
        vec_env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = PPO.load(load_path, env=vec_env, device=device)
    model.num_timesteps = steps_done

    # if we're past unfreeze point, make sure state branch is unfrozen
    if steps_done >= UNFREEZE_AT_STEP:
        model.policy.features_extractor.unfreeze_state_branch()
        print("  State branch already unfrozen (past unfreeze step).")

    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
        verbose     = 1,
    )
    unfreeze_cb = UnfreezeStateCallback(
        unfreeze_at = UNFREEZE_AT_STEP,
        new_lr      = 5e-5,
    )
    video_cb = VideoRecordingCallback(record_freq=100_000, include_step_zero=False)

    model.learn(
        total_timesteps     = remaining,
        callback            = [checkpoint_cb, unfreeze_cb, video_cb],
        progress_bar        = True,
        reset_num_timesteps = False,
    )

    model.save(final_path)
    vec_env.save(norm_path)
    print(f"\n✅ Continued training complete. Saved to: {final_path}")


def evaluate():
    print("\n── Phase 2 Evaluation ───────────────────────────────────────────────\n")

    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env   = DummyVecEnv([make_env(render=True)])
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print("  VecNormalize : loaded\n")
    else:
        print("  VecNormalize : not found\n")

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    if not os.path.exists(final_path + ".zip"):
        print(f"  ❌ No model at {final_path}. Train first.")
        return

    model = PPO.load(final_path, env=vec_env)

    MAX_EPISODES = 5
    obs          = vec_env.reset()
    ep_rewards   = []
    ep_lengths   = []
    crash_count  = 0
    oor_count    = 0
    total_reward = 0.0
    steps        = 0
    episode      = 1

    for _ in range(50_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps        += 1

        if done[0]:
            rc      = info[0].get("route_completion", 0.0)
            crashed = info[0].get("crash",       False)
            oor     = info[0].get("out_of_road", False)

            ep_rewards.append(total_reward)
            ep_lengths.append(steps)
            if crashed: crash_count += 1
            if oor:     oor_count   += 1

            print(f"  Ep {episode:2d}  reward: {total_reward:8.2f}  "
                  f"steps: {steps:5d}  route: {rc:.1%}  "
                  f"crash: {str(crashed):<5}  oor: {str(oor):<5}")

            total_reward = 0.0
            steps        = 0
            episode     += 1
            obs          = vec_env.reset()
            if episode > MAX_EPISODES:
                break

    print(f"\n  ── Summary ({len(ep_rewards)} episodes) ──")
    print(f"  Avg reward   : {np.mean(ep_rewards):.2f}")
    print(f"  Avg length   : {np.mean(ep_lengths):.0f} steps")
    print(f"  Crash rate   : {crash_count / len(ep_rewards):.0%}")
    print(f"  Out-of-road  : {oor_count   / len(ep_rewards):.0%}")

    vec_env.close()


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    modes = {
        "inspect":  inspect,
        "eval":     evaluate,
        "continue": continue_train,
    }
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    modes.get(mode, train)()