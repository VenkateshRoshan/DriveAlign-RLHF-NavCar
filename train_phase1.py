"""
train_phase1.py  -  DriveAlign Phase 1: Base RL Agent (Vector Observations)
=============================================================================
Multi-head feature extractor:
  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
  │  Ego Head    │  │ Lidar Head   │  │  Side Head   │  │  Lane Head   │  │  Nav Head    │
  │  (9 → 32)    │  │ (240 → 128)  │  │  (4 → 16)    │  │  (4 → 16)    │  │  (5 → 32)    │
  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘
         └─────────────────┴─────────────────┴─────────────────┴─────────────────┘
                                              │
                                    Concat (224-dim)
                                              │
                                    Fusion MLP (224 → 256 → 128)
                                              │
                                       Policy Head (PPO)

Usage:
  python train_phase1.py              # train from scratch
  python train_phase1.py inspect      # inspect observation space + slice boundaries
  python train_phase1.py eval         # evaluate saved model with rendering
  python train_phase1.py continue     # resume from latest checkpoint
"""

import os
import sys
import re
import glob

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from metadrive import MetaDriveEnv

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
sys.path.append(os.path.join(os.path.dirname(__file__), "config"))

from config import Config
from src.helpers import MultiHeadExtractor

# ─
# OBSERVATION SLICE DEFINITIONS
# ─
# MetaDrive flat vector layout (total ~259-dim with the config below):
#
#   [0  :  9 ]  →  Ego state      (9 values)
#   [9  : 249]  →  Lidar rays   (240 values, 240 lasers @ 50m range)
#   [249: 253]  →  Side detector  (4 values, left/right proximity)
#   [253: 257]  →  Lane detector  (4 values, lane line distances)
#   [257: 262]  →  Navigation     (5 values, heading/dist to waypoint)
#
# ⚠  Run `python train_phase1.py inspect` FIRST to verify these on your
#    MetaDrive version. If the total obs dim differs, adjust the constants
#    below to match what inspect() prints.
# ─



# Training config
TOTAL_TIMESTEPS = 1_000_000
SAVE_FREQ       =   100_000
LOG_DIR         = "./logs_phase1/"
MODEL_DIR       = "./models_phase1/"
MODEL_NAME      = "ppo_phase1"

os.makedirs(LOG_DIR,   exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def get_env_config(render: bool = False) -> dict:
    """
    Phase 1 environment - intentionally simple.
    No image, no traffic, fixed map layout.
    Goal: learn basic driving from lidar + state before adding any complexity.
    """
    return {
        "use_render":          render,
        "image_observation":   False,      # flat vector only in Phase 1
        "num_scenarios":       20,         # small variety of tracks
        "map":                 "SCSCSCS",  # straight + curve + straight (simple)
        "random_lane_width":   True,       # slight randomness to prevent overfitting
        "random_agent_model":  False,
        "traffic_density":     0.0,        # NO traffic - learn driving first
        "accident_prob":       0.0,        # NO static obstacles - keep it clean
        "vehicle_config": {
            "lidar": {
                "num_lasers":  Config.LIDAR_DIM,  # 240 rays in a full ring
                "distance":    50,         # 50m range
                "num_others":  0,          # don't encode other vehicles in lidar
            },
            "side_detector": {
                "num_lasers":  Config.SIDE_DIM,   # 4 rays to road edges
                "distance":    50,
            },
            "lane_line_detector": {
                "num_lasers":  Config.LANE_DIM,   # 4 rays to lane lines
                "distance":    20,
            },
        },
        # Reward shaping
        "use_lateral_reward":       True,
        "driving_reward":           1.0,   # reward for forward progress
        "speed_reward":             0.25,   # small speed bonus
        "out_of_road_penalty":      5.0,
        "crash_vehicle_penalty":    5.0,
        "crash_object_penalty":     5.0,
        "horizon":                  10000,  # max steps per episode
        "interface_panel":          [],
    }

def make_env(render: bool = False):
    """Returns a factory function for a single environment instance."""
    def _init():
        env = MetaDriveEnv(config=get_env_config(render=render))
        env = Monitor(env, LOG_DIR)
        return env
    return _init

def inspect():
    """
    Inspect the raw observation vector from MetaDrive.

    Run this BEFORE training to verify:
      1. The total observation dimension
      2. What values live at each index
      3. That the slice constants match your environment version

    If the obs dim printed here ≠ what the slice constants expect,
    update EGO_DIM / LIDAR_DIM / SIDE_DIM / LANE_DIM / NAV_DIM at the
    top of this file accordingly.
    """
    print("\n── Observation Space Inspection \n")

    env   = MetaDriveEnv(config=get_env_config(render=False))
    obs, _ = env.reset()

    total_dim = obs.shape[0]
    print(f"  Total observation dim : {total_dim}")
    print(f"\n  Expected slice layout:")
    print(f"    Ego    [{Config.EGO_START:3d} : {Config.EGO_START + Config.EGO_DIM:3d}]  →  {Config.EGO_DIM} values")
    print(f"    Lidar  [{Config.LIDAR_START:3d} : {Config.LIDAR_START + Config.LIDAR_DIM:3d}]  →  {Config.LIDAR_DIM} values")
    print(f"    Side   [{Config.SIDE_START:3d} : {Config.SIDE_START + Config.SIDE_DIM:3d}]  →  {Config.SIDE_DIM} values")
    print(f"    Lane   [{Config.LANE_START:3d} : {Config.LANE_START + Config.LANE_DIM:3d}]  →  {Config.LANE_DIM} values")
    print(f"    Nav    [{Config.NAV_START:3d} : {Config.NAV_START + Config.NAV_DIM:3d}]  →  {Config.NAV_DIM} values")
    print(f"    Total covered : {Config.EGO_DIM + Config.LIDAR_DIM + Config.SIDE_DIM + Config.LANE_DIM + Config.NAV_DIM}")

    if total_dim != Config.EGO_DIM + Config.LIDAR_DIM + Config.SIDE_DIM + Config.LANE_DIM + Config.NAV_DIM:
        print(f"\n  ⚠  MISMATCH: total obs dim is {total_dim} but slices sum to "
              f"{Config.EGO_DIM + Config.LIDAR_DIM + Config.SIDE_DIM + Config.LANE_DIM + Config.NAV_DIM}.")
        print(f"     Adjust the constants at the top of this file before training.\n")
    else:
        print(f"\n  ✅ Slices match total observation dimension.\n")

    # Print known ego indices
    known = {
        0: "lateral offset         (deviation from lane centre)",
        1: "heading error          (angle to road direction)",
        2: "speed longitudinal     (forward velocity m/s)",
        3: "speed lateral          (sideways velocity m/s)",
        4: "steering angle         (current wheel angle)",
        5: "yaw rate               (angular velocity rad/s)",
    }
    print(f"  Ego state values (first {Config.EGO_DIM} indices):")
    print(f"  {'idx':<5} {'value':>10}   signal")
    print(f"  {'-'*55}")
    for i in range(Config.EGO_DIM):
        label = known.get(i, "(ego state - unidentified)")
        print(f"  [{i:3d}] {obs[i]:>10.4f}   {label}")

    # Print nav values
    print(f"\n  Navigation values (indices {Config.NAV_START}–{Config.NAV_START + Config.NAV_DIM - 1}):")
    for i in range(Config.NAV_DIM):
        idx = Config.NAV_START + i
        print(f"  [{idx:3d}] {obs[idx]:>10.4f}")

    # Quick episode sanity check
    print(f"\n  Running a 500-step random episode to check reward signal...")
    total_reward = 0.0
    steps = 0
    for _ in range(500):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        total_reward += reward
        steps += 1
        if term or trunc:
            break

    print(f"\n  Episode stats (random policy):")
    print(f"    Steps           : {steps}")
    print(f"    Total reward    : {total_reward:.4f}")
    print(f"    Route complete  : {info.get('route_completion', 'N/A')}")
    print(f"    Crash           : {info.get('crash', 'N/A')}")
    print(f"    Out of road     : {info.get('out_of_road', 'N/A')}")
    print(f"    Action space    : {env.action_space}\n")

    env.close()
    print("── Done ───────────────────────\n")


def train():
    """
    Train Phase 1 from scratch.
    Saves checkpoints every SAVE_FREQ steps and a final model on completion.
    """
    vec_env = DummyVecEnv([make_env(render=False)])
    vec_env = VecNormalize(
        vec_env,
        norm_obs    = True,   # normalise flat obs vector (important for lidar scale)
        norm_reward = True,
        clip_obs    = 10.0,
        clip_reward = 10.0,
    )

    policy_kwargs = dict(
        features_extractor_class  = MultiHeadExtractor,
        features_extractor_kwargs = dict(features_dim=128),
        net_arch                  = [128, 64],  # layers after extractor → action
    )

    model = PPO(
        policy          = "MlpPolicy",
        env             = vec_env,
        policy_kwargs   = policy_kwargs,

        learning_rate   = 3e-4,
        n_steps         = 2048,   # steps per env per update
        batch_size      = 128,
        n_epochs        = 10,
        gamma           = 0.99,
        gae_lambda      = 0.95,
        clip_range      = 0.2,
        ent_coef        = 0.01,   # exploration encouragement
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

    total_params    = sum(p.numel() for p in model.policy.parameters())
    trainable       = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)

    print("\n── Phase 1 Training ───────────")
    print(f"  Device          : {'GPU (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    print(f"  Total params    : {total_params:,}")
    print(f"  Trainable       : {trainable:,}")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")
    print(f"  Save every      : {SAVE_FREQ:,} steps → {MODEL_DIR}")
    print(f"  TensorBoard     : tensorboard --logdir {LOG_DIR}")
    print(f"\n  Head dims       : ego→32  lidar→128  side→16  lane→16  nav→32")
    print(f"  Fusion          : 224 → 256 → 128 (features_dim)")
    print(f"  Policy arch     : [128, 64] after extractor")
    print(f"───────────────────────────────\n")

    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = checkpoint_cb,
        progress_bar    = True,
    )

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model.save(final_path)
    vec_env.save(os.path.join(MODEL_DIR, "vecnormalize.pkl"))
    print(f"\n✅ Training complete. Model saved to: {final_path}")


def continue_train(extra_timesteps: int = None):
    """
    Resume training from the latest checkpoint.
    Automatically finds the highest-step .zip in MODEL_DIR.
    Pass extra_timesteps to train for a fixed number of additional steps,
    or leave None to complete up to TOTAL_TIMESTEPS.
    """
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
        latest    = max(ckpt_files, key=_extract_steps)
        steps_done = _extract_steps(latest)
        load_path  = latest.replace(".zip", "")
        print(f"\n  Latest checkpoint : {os.path.basename(latest)}")
        print(f"  Steps completed   : {steps_done:,}")
    else:
        load_path  = final_path
        steps_done = 0
        print("\n  No step checkpoint found — loading _final model.")

    remaining = extra_timesteps if extra_timesteps is not None else max(0, TOTAL_TIMESTEPS - steps_done)

    if remaining <= 0:
        print(f"  Already at {steps_done:,} / {TOTAL_TIMESTEPS:,} steps. Nothing to do.")
        return

    print(f"  Remaining steps   : {remaining:,}\n")

    vec_env   = DummyVecEnv([make_env(render=False)])
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = True
        vec_env.norm_reward = True
        print("  VecNormalize      : loaded from checkpoint")
    else:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True,
                               clip_obs=10.0, clip_reward=10.0)
        print("  VecNormalize      : freshly initialised")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = PPO.load(load_path, env=vec_env, device=device)
    model.num_timesteps = steps_done

    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ,
        save_path   = MODEL_DIR,
        name_prefix = MODEL_NAME,
        verbose     = 1,
    )

    model.learn(
        total_timesteps     = remaining,
        callback            = checkpoint_cb,
        progress_bar        = True,
        reset_num_timesteps = False,  # keeps the step counter
    )

    model.save(final_path)
    vec_env.save(norm_path)
    print(f"\n✅ Continued training complete. Saved to: {final_path}")


def evaluate():
    """
    Evaluate the saved Phase 1 model with rendering enabled.
    Prints per-episode stats: reward, route completion, crash, out-of-road.
    """
    print("\n── Phase 1 Evaluation ─────────\n")

    vec_env   = DummyVecEnv([make_env(render=True)])
    norm_path = os.path.join(MODEL_DIR, "vecnormalize.pkl")

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print("  VecNormalize : loaded\n")
    else:
        print("  VecNormalize : not found — running without normalisation\n")

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    if not os.path.exists(final_path + ".zip"):
        print(f"  ❌ No model found at {final_path}. Train first.")
        return

    model = PPO.load(final_path, env=vec_env)

    MAX_EPISODES    = 10
    obs             = vec_env.reset()
    episode_rewards = []
    episode_lengths = []
    crash_count     = 0
    oor_count       = 0
    total_reward    = 0.0
    steps           = 0
    episode         = 1

    for _ in range(50_000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        steps        += 1

        if done[0]:
            rc      = info[0].get("route_completion", 0.0)
            crashed = info[0].get("crash",       False)
            oor     = info[0].get("out_of_road", False)

            episode_rewards.append(total_reward)
            episode_lengths.append(steps)
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

    print(f"\n  ── Summary ({len(episode_rewards)} episodes) ──")
    print(f"  Avg reward      : {np.mean(episode_rewards):.2f}")
    print(f"  Avg length      : {np.mean(episode_lengths):.0f} steps")
    print(f"  Crash rate      : {crash_count / len(episode_rewards):.0%}")
    print(f"  Out-of-road     : {oor_count   / len(episode_rewards):.0%}")
    print(f"\n  Phase 1 target  : route completion > 80%, crash rate < 20%")

    vec_env.close()

def record():
    try:
        import imageio
        from PIL import Image, ImageDraw, ImageFont
        import mss
        import subprocess
        import time
    except ImportError:
        print("  pip install imageio[ffmpeg] pillow mss")
        return

    os.makedirs("outputs/recordings", exist_ok=True)
    out_path = "outputs/recordings/phase1_episode.mp4"

    FONT_PATHS = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    font_large = font_small = None
    for fp in FONT_PATHS:
        if os.path.exists(fp):
            font_large = ImageFont.truetype(fp, 22)
            font_small = ImageFont.truetype(fp, 15)
            break
    if font_large is None:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    BG_FILL    = (10,  10,  10,  185)
    BORDER_COL = (255, 255, 255,  60)
    TITLE_COL  = (200, 200, 200, 255)
    VAL_COLS   = {
        "reward": (120, 220, 255, 255),
        "speed":  ( 90, 255, 160, 255),
        "route":  (255, 200,  80, 255),
    }

    def draw_hud(frame, cum_reward, speed, route):
        img = Image.fromarray(frame).convert("RGBA")
        W, H = img.size
        PAD  = 18
        BW, BH = 210, 112
        X1 = W - BW - 20;  Y1 = 20
        X2 = X1 + BW;      Y2 = Y1 + BH

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rounded_rectangle([X1, Y1, X2, Y2], radius=10, fill=BG_FILL)
        d.rounded_rectangle([X1, Y1, X2, Y2], radius=10, outline=BORDER_COL, width=1)
        d.text((X1 + PAD, Y1 + 10), "DriveAlign · Phase 1", font=font_small, fill=TITLE_COL)
        d.line([(X1 + PAD, Y1 + 30), (X2 - PAD, Y1 + 30)], fill=(255, 255, 255, 30), width=1)

        for i, (label, value, key) in enumerate([
            ("REWARD", f"{cum_reward:>8.1f}", "reward"),
            ("SPEED",  f"{speed:>6.1f} m/s",  "speed"),
            ("ROUTE",  f"{route:>6.1%}",       "route"),
        ]):
            y = Y1 + 38 + i * 23
            d.text((X1 + PAD,      y), label, font=font_small, fill=TITLE_COL)
            d.text((X1 + PAD + 80, y), value, font=font_large, fill=VAL_COLS[key])

        return np.array(Image.alpha_composite(img, overlay).convert("RGB"))

    # get screen resolution
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        SW, SH  = monitor["width"], monitor["height"]

    norm_path  = os.path.join(MODEL_DIR, "vecnormalize.pkl")
    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")

    cfg = get_env_config(render=True)
    cfg["window_size"] = (SW, SH)
    cfg["show_logo"]   = False
    cfg["show_fps"]    = False

    vec_env = DummyVecEnv([lambda: Monitor(MetaDriveEnv(config=cfg), LOG_DIR)])
    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False

    model    = PPO.load(final_path, env=vec_env, device="cpu")
    obs_vec  = vec_env.reset()

    # force fullscreen and wait for it to settle
    time.sleep(2.0)
    subprocess.run(["wmctrl", "-r", ":ACTIVE:", "-b", "add,fullscreen"], check=False)
    time.sleep(0.5)

    frames     = []
    cum_reward = 0.0

    with mss.mss() as sct:
        region = sct.monitors[1]   # full monitor

        for _ in range(10_000):
            action, _ = model.predict(obs_vec, deterministic=True)
            obs_vec, reward, done, info = vec_env.step(action)
            cum_reward += float(reward[0])

            raw   = np.array(sct.grab(region))
            frame = raw[:, :, 2::-1].copy()   # BGRA → RGB

            speed = float(info[0].get("velocity", 0.0))
            route = float(info[0].get("route_completion", 0.0))
            frames.append(draw_hud(frame, cum_reward, speed, route).astype(np.uint8))

            if done[0]:
                break

    vec_env.close()

    print(f"  Writing {len(frames)} frames...")
    writer = imageio.get_writer(out_path, fps=30, codec="libx264", quality=8, macro_block_size=1)
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"  Saved: {out_path}")

if __name__ == "__main__":
    modes = {
        "inspect":  inspect,
        "eval":     evaluate,
        "continue": continue_train,
        "record":   record,
    }
    mode = sys.argv[1] if len(sys.argv) > 1 else "train"
    modes.get(mode, train)()