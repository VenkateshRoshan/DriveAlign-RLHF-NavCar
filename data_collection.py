import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_recorder import SegmentRecorder, load_model, get_env_config, make_env, SEGMENTS_DIR, SEGMENT_LENGTH, NUM_SEGMENTS
from feedback import call_mistral, build_prompt, parse_llm_response, save_label
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import data_recorder
data_recorder.RENDER = True

def print_segment_stats(stats):
    """Print what just happened in this segment so you can give informed feedback."""
    s = stats["summary"]
    print("\n" + "─" * 60)
    print("  📊 SEGMENT SUMMARY - What just happened:")
    print("─" * 60)
    print(f"  Speed          : avg {s['avg_speed']:.1f} m/s  |  max {s['max_speed']:.1f} m/s")
    print(f"  Lane deviation : avg {s['avg_lane_deviation']:.2f}m  |  max {s['max_lane_deviation']:.2f}m")
    print(f"  Steering smooth: {s['steering_smoothness']:.3f}  (lower = smoother)")
    print(f"  Total reward   : {s['total_reward']:.2f}")
    print(f"  Crashed        : {'❌ YES' if s['crashed'] else '✅ No'}")
    print(f"  Out of road    : {'❌ YES' if s['out_of_road'] else '✅ No'}")
    print(f"\n  📝 Description:")
    print(f"  {stats['description']}")
    print("─" * 60)

def get_human_feedback():
    """Ask for feedback. Allow skipping."""
    print("\n  💬 What did you think of that driving?")
    print("  (press Enter to skip this segment)\n")
    feedback = input("  Your feedback: ").strip()
    return feedback if feedback else None

def label_with_mistral(stats, human_feedback, stats_path):
    """Call Mistral and save the label immediately."""
    print("\n  🤖 Sending to Mistral for labelling...")
    prompt   = build_prompt(stats["description"], human_feedback)
    response = call_mistral(prompt)

    if response is None:
        print("  ❌ Mistral failed to respond. Skipping label.")
        return

    label_result = parse_llm_response(response)

    if label_result is None:
        print("  ❌ Could not parse Mistral response. Skipping label.")
        return

    save_label(stats_path, stats, label_result, human_feedback)

    emoji = "✅" if label_result["label"] == "good" else "❌"
    print(f"\n  {emoji} Mistral says : {label_result['label'].upper()}  "
          f"(confidence: {label_result['confidence']:.2f})")
    print(f"  💬 Reason      : {label_result['reason']}")

def run():
    print("\n🚗 Live Recording + Feedback Session")
    print(f"   Segments to record : {NUM_SEGMENTS}")
    print(f"   Segment length     : {SEGMENT_LENGTH} steps")
    print(f"   Render             : ON (watch the agent drive)")
    print(f"   Feedback           : Human + Mistral (after each segment)\n")

    print("📦 Loading model...")
    model, vec_env = load_model()

    recorder      = SegmentRecorder(segment_length=SEGMENT_LENGTH, save_dir=SEGMENTS_DIR)
    segments_saved = 0
    obs            = vec_env.reset()
    crashed        = False
    out_of_road    = False

    print("\n▶️  Starting - watch the agent drive in the window.")
    print("   After each segment you'll be asked for feedback.\n")

    while segments_saved < NUM_SEGMENTS:
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, info = vec_env.step(action)

        reward_val    = float(reward[0])
        done_val      = bool(done[0])
        info_val      = info[0]
        action_val    = action[0]
        obs_unwrapped = {k: v[0] for k, v in obs.items()} if isinstance(obs, dict) else obs[0]

        if info_val.get("crash", False):
            crashed = True
        if info_val.get("out_of_road", False):
            out_of_road = True

        recorder.record_step(obs_unwrapped, action_val, reward_val, info_val)

        if recorder.is_full() or done_val:
            # ── Flush segment to disk ──────────────────────
            seg_id, seg_dir = recorder.flush(crashed=crashed, out_of_road=out_of_road)
            segments_saved += 1

            print(f"\n⏸️  Segment {segments_saved}/{NUM_SEGMENTS} complete  │  ID: {seg_id}")

            # ── Load the saved stats ───────────────────────
            import json
            stats_path = os.path.join(seg_dir, "stats.json")
            with open(stats_path) as f:
                stats = json.load(f)

            # ── Show what happened ─────────────────────────
            print_segment_stats(stats)

            # ── Get human feedback ─────────────────────────
            human_feedback = get_human_feedback()

            if human_feedback:
                # ── Label with Mistral ─────────────────────
                label_with_mistral(stats, human_feedback, stats_path)
            else:
                print("  ⏭️  Skipped. No label saved for this segment.")

            print(f"\n▶️  Continuing... ({segments_saved}/{NUM_SEGMENTS} done)\n")

            crashed     = False
            out_of_road = False

        if done_val:
            obs = vec_env.reset()
        else:
            obs = next_obs

    vec_env.close()

    # ── Final summary ──────────────────────────────────────
    import json
    import numpy as np

    print("\n✅ Session complete!\n")

    segment_dirs = sorted([
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    ])

    good = bad = skipped = 0
    for seg_dir in segment_dirs:
        sp = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if not os.path.exists(sp):
            continue
        with open(sp) as f:
            s = json.load(f)
        label = s.get("label")
        if label == "good":
            good += 1
        elif label == "bad":
            bad += 1
        else:
            skipped += 1

    print(f"   Total segments : {segments_saved}")
    print(f"   ✅ Good        : {good}")
    print(f"   ❌ Bad         : {bad}")
    print(f"   ⏭️  Skipped    : {skipped}")
    print(f"\n🚀 Ready for Phase 5 - Reward Model Training\n")

if __name__ == "__main__":
    run()