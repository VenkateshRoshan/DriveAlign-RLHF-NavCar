import os
import json
import requests
import argparse
import numpy as np
from datetime import datetime

SEGMENTS_DIR    = "./segments/"
OLLAMA_URL      = "http://localhost:11434/api/generate"
MISTRAL_MODEL   = "mistral"

def call_mistral(prompt):
    """
    Send a prompt to local Mistral via Ollama.
    Returns the response text.
    """
    payload = {
        "model":  MISTRAL_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.2,
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        print("\n❌ Could not connect to Ollama.")
        print("   Make sure Ollama is running:  ollama serve")
        print("   And Mistral is pulled:        ollama pull mistral\n")
        exit(1)
    except Exception as e:
        print(f"\n❌ Ollama error: {e}\n")
        return None

def build_prompt(segment_description, human_feedback):
    """
    Build the prompt that Mistral will respond to.
    Asks it to label the segment as good/bad based on:
      - what actually happened (description)
      - what the human said (feedback)
    """
    prompt = f"""You are evaluating the driving behaviour of an autonomous driving agent.

Here is a description of what the agent did during a short driving segment:
\"{segment_description}\"

The human evaluator has given the following feedback about driving quality in general:
\"{human_feedback}\"

Based on the segment description and the human's feedback, decide if this segment shows GOOD or BAD driving.

Reply in this exact JSON format and nothing else:
{{
  "label": "good" or "bad",
  "confidence": a number between 0.0 and 1.0,
  "reason": "one sentence explaining why"
}}

Only reply with the JSON. No extra text."""
    return prompt

def parse_llm_response(response_text):
    """
    Parse the JSON response from Mistral.
    Returns dict with label, confidence, reason.
    Falls back gracefully if JSON is malformed.
    """
    try:
        # Strip any markdown code fences if model adds them
        cleaned = response_text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        parsed = json.loads(cleaned)

        # Validate expected keys
        assert "label"      in parsed
        assert "confidence" in parsed
        assert "reason"     in parsed
        assert parsed["label"] in ["good", "bad"]

        return parsed

    except Exception as e:
        print(f"  ⚠️  Could not parse LLM response: {e}")
        print(f"  Raw response: {response_text[:200]}")
        return None

def load_segments(only_unlabelled=True):
    """
    Load all segments from disk.
    If only_unlabelled=True, skip segments already labelled.
    """
    segment_dirs = sorted([
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    ])

    segments = []
    for seg_dir in segment_dirs:
        stats_path = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        # Skip already labelled ones if requested
        if only_unlabelled and stats.get("label") is not None:
            continue

        segments.append({
            "dir":         os.path.join(SEGMENTS_DIR, seg_dir),
            "stats_path":  stats_path,
            "stats":       stats,
        })

    return segments

def save_label(stats_path, stats, label_result, human_feedback):
    """Write the LLM label back into the segment's stats.json."""
    stats["label"]        = label_result["label"]
    stats["confidence"]   = label_result["confidence"]
    stats["llm_reason"]   = label_result["reason"]
    stats["human_feedback"] = human_feedback
    stats["labelled_at"]  = datetime.now().isoformat()

    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

def run_feedback_session():
    print("\n🧠 Phase 4: LLM Feedback Layer")
    print("   Model  : Mistral (local via Ollama)")
    print(f"   Segments dir: {SEGMENTS_DIR}\n")

    # ── Step 1: Get human feedback ──────────────────
    print("─" * 55)
    print("  Type your feedback about the agent's driving.")
    print("  Examples:")
    print("    'the car was too aggressive in turns'")
    print("    'it was driving too slowly'")
    print("    'good driving but slightly off center'")
    print("─" * 55)
    human_feedback = input("\n  Your feedback: ").strip()

    if not human_feedback:
        print("❌ No feedback entered. Exiting.")
        return

    print(f"\n  ✅ Feedback received: \"{human_feedback}\"\n")

    # ── Step 2: Load unlabelled segments ────────────
    segments = load_segments(only_unlabelled=True)

    if not segments:
        print("⚠️  No unlabelled segments found.")
        print("   Either run phase3_record.py first, or all segments are already labelled.")
        return

    print(f"  📂 Found {len(segments)} unlabelled segments to label.\n")

    # ── Step 3: Label each segment with Mistral ──────
    good_count = 0
    bad_count  = 0
    fail_count = 0

    for i, seg in enumerate(segments):
        description = seg["stats"]["summary"]
        desc_text   = seg["stats"].get("description", str(description))
        seg_id      = seg["stats"].get("segment_id", os.path.basename(seg["dir"]))

        print(f"  [{i+1:>3}/{len(segments)}] Segment {seg_id}")
        print(f"          Description: {desc_text[:90]}...")

        # Build prompt and call Mistral
        prompt   = build_prompt(desc_text, human_feedback)
        response = call_mistral(prompt)

        if response is None:
            fail_count += 1
            print(f"          ❌ Failed to get response\n")
            continue

        # Parse response
        label_result = parse_llm_response(response)

        if label_result is None:
            fail_count += 1
            print(f"          ❌ Failed to parse response\n")
            continue

        # Save label back to disk
        save_label(seg["stats_path"], seg["stats"], label_result, human_feedback)

        # Display result
        emoji = "✅" if label_result["label"] == "good" else "❌"
        print(f"          {emoji} Label     : {label_result['label'].upper()}  "
              f"(confidence: {label_result['confidence']:.2f})")
        print(f"          💬 Reason    : {label_result['reason']}\n")

        if label_result["label"] == "good":
            good_count += 1
        else:
            bad_count += 1

    # ── Step 4: Summary ─────────────────────────────
    total_labelled = good_count + bad_count
    print("─" * 55)
    print(f"\n✅ Labelling complete!\n")
    print(f"   Total labelled : {total_labelled}")
    print(f"   Good segments  : {good_count}  ({100*good_count/max(total_labelled,1):.0f}%)")
    print(f"   Bad segments   : {bad_count}  ({100*bad_count/max(total_labelled,1):.0f}%)")
    if fail_count:
        print(f"   Failed         : {fail_count}")

    print(f"\n🚀 Ready for Phase 5 — Reward Model Training\n")

def review():
    """Print a summary of all labelled segments."""
    segments = sorted([
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    ])

    print(f"\n📋 Labelled Segments Review\n")
    print(f"  {'ID':<12} {'Label':<8} {'Conf':<6} {'Reward':<10} {'Crashed':<10} Reason")
    print("  " + "─" * 80)

    good = bad = unlabelled = 0

    for seg_dir in segments:
        stats_path = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        label   = stats.get("label")
        conf    = stats.get("confidence", "-")
        reward  = stats["summary"]["total_reward"]
        crashed = stats["summary"]["crashed"]
        reason  = stats.get("llm_reason", "-")
        seg_id  = stats.get("segment_id", seg_dir)

        if label is None:
            unlabelled += 1
            print(f"  {seg_id:<12} {'(unlabelled)':<8}")
            continue

        emoji = "✅" if label == "good" else "❌"
        conf_str = f"{conf:.2f}" if isinstance(conf, float) else str(conf)
        print(f"  {seg_id:<12} {emoji} {label:<6} {conf_str:<6} {reward:<10.2f} {str(crashed):<10} {reason[:50]}")

        if label == "good":
            good += 1
        else:
            bad += 1

    print(f"\n  Good: {good}  |  Bad: {bad}  |  Unlabelled: {unlabelled}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", default="label",
                        choices=["label", "review"],
                        help="label = run feedback session | review = show all labels")
    args = parser.parse_args()

    if args.mode == "review":
        review()
    else:
        run_feedback_session()