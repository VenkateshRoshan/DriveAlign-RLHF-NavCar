import os
import json
import uuid
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from metadrive import MetaDriveEnv
from helpers import (
    PHASE2_MODEL_NAME,
    PHASE2_VECNORM_FILENAME,
    N_STACK,
    StateStackWrapper,
    SmoothnessWrapper,
    compute_reward_features,
    get_env_config,
)

MODEL_DIR       = "./models_vision/"
MODEL_NAME      = PHASE2_MODEL_NAME
SEGMENTS_DIR    = "./segments/"

SEGMENT_LENGTH  = 50        # how many steps per segment (~10 seconds at 10Hz)
NUM_SEGMENTS    = 50         # total segments to record
RENDER          = False      # set True if you want to watch while recording

os.makedirs(SEGMENTS_DIR, exist_ok=True)

def make_env(render=False):
    def _init():
        env = MetaDriveEnv(config=get_env_config(render=render))
        # env = SmoothnessWrapper(env)
        env = StateStackWrapper(env, n_stack=N_STACK)
        env = Monitor(env)
        return env
    return _init

class SegmentRecorder:
    """
    Records a fixed-length chunk of driving experience.
    Buffers step data and flushes to disk when segment is full.
    """

    def __init__(self, segment_length=100, save_dir="./segments/"):
        self.segment_length = segment_length
        self.save_dir       = save_dir
        self.reset_buffer()

    def reset_buffer(self):
        self.states         = []   # lidar/state vectors
        self.actions        = []   # agent actions
        self.rewards        = []   # env rewards at each step
        self.speeds         = []   # vehicle speed
        self.lane_devs      = []   # lane deviation
        self.steerings      = []   # steering angle
        self.step_count     = 0

    def record_step(self, obs, action, reward, info):
        """Call this every env step to buffer data."""
        self.states.append(obs["state"].copy())
        self.actions.append(np.array(action).copy())
        self.rewards.append(float(reward))

        # MetaDrive info dict has these fields
        self.speeds.append(float(info.get("velocity", 0.0)))
        self.lane_devs.append(float(info.get("lateral_dist", 0.0)))
        self.steerings.append(float(action[0]) if len(action) > 0 else 0.0)

        self.step_count += 1

    def is_full(self):
        return self.step_count >= self.segment_length

    def compute_summary(self, crashed, out_of_road):
        """Compute summary stats for the segment."""
        speeds     = np.array(self.speeds)
        lane_devs  = np.abs(np.array(self.lane_devs))
        steerings  = np.abs(np.array(self.steerings))
        rewards    = np.array(self.rewards)

        # Smoothness = how much steering changes between steps (lower = smoother)
        steering_diff = np.abs(np.diff(steerings)) if len(steerings) > 1 else np.array([0.0])

        return {
            "avg_speed":           float(np.mean(speeds)),
            "max_speed":           float(np.max(speeds)),
            "avg_lane_deviation":  float(np.mean(lane_devs)),
            "max_lane_deviation":  float(np.max(lane_devs)),
            "avg_steering":        float(np.mean(steerings)),
            "steering_smoothness": float(np.mean(steering_diff)),   # lower = smoother
            "total_reward":        float(np.sum(rewards)),
            "avg_reward":          float(np.mean(rewards)),
            "steps":               self.step_count,
            "crashed":             bool(crashed),
            "out_of_road":         bool(out_of_road),
        }

    def flush(self, crashed=False, out_of_road=False):
        """
        Save the buffered segment to disk and return the segment ID.
        Saves:
          - states.npy    → state vectors at each step
          - actions.npy   → actions at each step
          - rewards.npy   → rewards at each step
          - stats.json    → numeric summary + features
        """
        segment_id  = str(uuid.uuid4())[:8]
        segment_dir = os.path.join(self.save_dir, f"seg_{segment_id}")
        os.makedirs(segment_dir, exist_ok=True)

        # Save arrays
        np.save(os.path.join(segment_dir, "states.npy"),  np.array(self.states))
        np.save(os.path.join(segment_dir, "actions.npy"), np.array(self.actions))
        np.save(os.path.join(segment_dir, "rewards.npy"), np.array(self.rewards))

        # Compute and save stats
        summary = self.compute_summary(crashed, out_of_road)
        reward_features = compute_reward_features(
            np.array(self.states),
            np.array(self.actions),
            crashed=crashed,
            out_of_road=out_of_road,
        )

        stats = {
            "segment_id":       segment_id,
            "summary":          summary,
            "reward_features":  reward_features.tolist(),
            "score":            None,      # filled manually
        }

        with open(os.path.join(segment_dir, "stats.json"), "w") as f:
            json.dump(stats, f, indent=2)

        # Reset buffer for next segment
        self.reset_buffer()

        return segment_id, segment_dir

def load_model():
    vec_env   = DummyVecEnv([make_env(render=RENDER)])
    norm_path = os.path.join(MODEL_DIR, PHASE2_VECNORM_FILENAME)

    if os.path.exists(norm_path):
        vec_env = VecNormalize.load(norm_path, vec_env)
        vec_env.training    = False
        vec_env.norm_reward = False
        print("  ✅ VecNormalize loaded")
    else:
        print(f"  ⚠️  No {PHASE2_VECNORM_FILENAME} found, running without normalization")

    final_path = os.path.join(MODEL_DIR, MODEL_NAME + "_final")
    model      = PPO.load(final_path, env=vec_env)
    print(f"  ✅ Model loaded from: {final_path}")

    return model, vec_env

def record():
    print("\n🎬 Phase 3: Segment Recording\n")
    print(f"   Segment length : {SEGMENT_LENGTH} steps")
    print(f"   Total segments : {NUM_SEGMENTS}")
    print(f"   Save directory : {SEGMENTS_DIR}\n")

    print("📦 Loading model...")
    model, vec_env = load_model()

    recorder         = SegmentRecorder(segment_length=SEGMENT_LENGTH, save_dir=SEGMENTS_DIR)
    segments_saved   = 0
    obs              = vec_env.reset()
    crashed          = False
    out_of_road      = False

    print(f"\n▶️  Recording started...\n")

    while segments_saved < NUM_SEGMENTS:
        # Agent picks action
        action, _ = model.predict(obs, deterministic=True)

        # Step env
        next_obs, reward, done, info = vec_env.step(action)

        # SB3 VecEnv returns lists — unwrap
        reward_val   = float(reward[0])
        done_val     = bool(done[0])
        info_val     = info[0]
        action_val   = action[0]
        obs_unwrapped = {k: v[0] for k, v in obs.items()} if isinstance(obs, dict) else obs[0]

        # Track incidents
        if info_val.get("crash", False):
            crashed = True
        if info_val.get("out_of_road", False):
            out_of_road = True

        # Record this step
        recorder.record_step(obs_unwrapped, action_val, reward_val, info_val)

        # Check if segment is full OR episode ended
        if recorder.is_full() or done_val:
            seg_id, seg_dir = recorder.flush(crashed=crashed, out_of_road=out_of_road)
            segments_saved += 1
            print(f"  ✅ Segment {segments_saved:>3}/{NUM_SEGMENTS}  │  ID: {seg_id}  │  saved to: {seg_dir}")

            crashed     = False
            out_of_road = False

        # Reset env if episode ended
        if done_val:
            obs = vec_env.reset()
        else:
            obs = next_obs

    vec_env.close()

    print(f"\n✅ Recording complete!")
    print(f"   {segments_saved} segments saved to: {SEGMENTS_DIR}")
    print(f"\n📋 Segment summary:")

    # Print a quick overview of all saved segments
    segment_dirs = sorted([
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    ])

    total_crashes  = 0
    total_oor      = 0
    all_rewards    = []

    for seg_dir in segment_dirs:
        stats_path = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                stats = json.load(f)
            summary = stats["summary"]
            if summary["crashed"]:
                total_crashes += 1
            if summary["out_of_road"]:
                total_oor += 1
            all_rewards.append(summary["total_reward"])

    print(f"   Total segments  : {len(segment_dirs)}")
    print(f"   Crashes         : {total_crashes}")
    print(f"   Out of road     : {total_oor}")
    print(f"   Avg reward      : {np.mean(all_rewards):.2f}")
    print(f"   Best segment    : {np.max(all_rewards):.2f}")
    print(f"   Worst segment   : {np.min(all_rewards):.2f}")
    print(f"\n🚀 Ready for Phase 4 — Manual Numeric Labelling\n")

def inspect_segment(segment_id=None):
    """
    Print the contents of a saved segment.
    Pass a segment ID or leave None to inspect the first one.
    """
    segment_dirs = sorted([
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    ])

    if not segment_dirs:
        print("❌ No segments found. Run recording first.")
        return

    target = segment_dirs[0] if segment_id is None else f"seg_{segment_id}"
    seg_path = os.path.join(SEGMENTS_DIR, target)

    if not os.path.exists(seg_path):
        print(f"❌ Segment not found: {seg_path}")
        return

    print(f"\n🔍 Inspecting segment: {target}\n")

    states  = np.load(os.path.join(seg_path, "states.npy"))
    actions = np.load(os.path.join(seg_path, "actions.npy"))
    rewards = np.load(os.path.join(seg_path, "rewards.npy"))

    with open(os.path.join(seg_path, "stats.json")) as f:
        stats = json.load(f)

    print(f"  States shape  : {states.shape}")
    print(f"  Actions shape : {actions.shape}")
    print(f"  Rewards shape : {rewards.shape}")
    print(f"\n  Summary:")
    for k, v in stats["summary"].items():
        print(f"    {k:<25}: {v}")
    print(f"\n  Reward features:")
    print(f"  {stats.get('reward_features')}")
    print(f"\n  Score         : {stats.get('score')}  (None = not yet labelled)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "inspect":
        seg_id = sys.argv[2] if len(sys.argv) > 2 else None
        inspect_segment(seg_id)
    else:
        record()
