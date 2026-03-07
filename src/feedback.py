import os
import json
import argparse
from datetime import datetime

SEGMENTS_DIR = "./segments/"


def load_segments(only_unlabelled=True):
    segment_dirs = sorted(
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    )

    segments = []
    for seg_dir in segment_dirs:
        stats_path = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        if only_unlabelled and stats.get("score") is not None:
            continue

        segments.append({
            "dir": os.path.join(SEGMENTS_DIR, seg_dir),
            "stats_path": stats_path,
            "stats": stats,
        })

    return segments


def save_score(stats_path, stats, score):
    stats["score"] = float(score)
    stats["labelled_at"] = datetime.now().isoformat()
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)


def prompt_score(seg_id, summary):
    print(f"\nSegment {seg_id}")
    print(f"  avg_speed          : {summary.get('avg_speed', 0.0):.3f}")
    print(f"  avg_lane_deviation : {summary.get('avg_lane_deviation', 0.0):.3f}")
    print(f"  steering_smoothness: {summary.get('steering_smoothness', 0.0):.3f}")
    print(f"  total_reward       : {summary.get('total_reward', 0.0):.3f}")
    print(f"  crashed            : {summary.get('crashed', False)}")
    print(f"  out_of_road        : {summary.get('out_of_road', False)}")

    while True:
        raw = input("  Enter score [0.0-1.0], or 's' to skip, or 'q' to quit: ").strip().lower()
        if raw == "s":
            return None
        if raw == "q":
            return "quit"
        try:
            score = float(raw)
            if 0.0 <= score <= 1.0:
                return score
        except ValueError:
            pass
        print("  Invalid input. Use a number between 0.0 and 1.0, 's', or 'q'.")


def run_feedback_session():
    print("\n🧠 Phase 4: Manual Numeric Labelling")
    print(f"   Segments dir: {SEGMENTS_DIR}\n")

    segments = load_segments(only_unlabelled=True)
    if not segments:
        print("⚠️  No unlabelled segments found.")
        return

    print(f"  📂 Found {len(segments)} unlabelled segments.\n")

    labelled = 0
    skipped = 0

    for i, seg in enumerate(segments, 1):
        seg_id = seg["stats"].get("segment_id", os.path.basename(seg["dir"]))
        summary = seg["stats"].get("summary", {})

        print(f"[{i:>3}/{len(segments)}]")
        score = prompt_score(seg_id, summary)

        if score == "quit":
            break
        if score is None:
            skipped += 1
            continue

        save_score(seg["stats_path"], seg["stats"], score)
        labelled += 1

    print("\n✅ Labelling complete")
    print(f"   Labelled : {labelled}")
    print(f"   Skipped  : {skipped}\n")


def review():
    segments = sorted(
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    )

    print("\n📋 Labelled Segments Review\n")
    good = bad = unlabelled = 0

    for seg_dir in segments:
        stats_path = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        seg_id = stats.get("segment_id", seg_dir)
        score = stats.get("score")
        summary = stats.get("summary", {})
        reward = summary.get("total_reward", 0.0)
        crashed = summary.get("crashed", False)

        if score is None:
            unlabelled += 1
            print(f"  {seg_id:<12} (unlabelled)")
            continue

        emoji = "✅" if score >= 0.5 else "❌"
        print(f"  {seg_id:<12} {emoji} score={score:.2f} reward={reward:.2f} crashed={crashed}")
        if score >= 0.5:
            good += 1
        else:
            bad += 1

    print("\nSummary:")
    print(f"  Total segments : {len(segments)}")
    print(f"  Good           : {good}")
    print(f"  Bad            : {bad}")
    print(f"  Unlabelled     : {unlabelled}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        nargs="?",
        default="label",
        choices=["label", "review"],
        help="label = manual score entry | review = show all scores",
    )
    args = parser.parse_args()

    if args.mode == "review":
        review()
    else:
        run_feedback_session()
