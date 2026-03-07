import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from helpers import REWARD_FEATURE_DIM, compute_reward_features

SEGMENTS_DIR = "./segments/"
MODEL_SAVE_DIR = "./reward_model/"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)


class DrivingSegmentDataset(Dataset):
    """Each sample: numeric trajectory features -> scalar score [0, 1]."""

    def __init__(self, segments):
        self.samples = segments

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg = self.samples[idx]
        return {
            "features": torch.tensor(seg["features"], dtype=torch.float32),
            "score": torch.tensor(float(seg["score"]), dtype=torch.float32),
        }


class GPT2RewardModel(nn.Module):
    """MLP reward model (name kept for compatibility with phase-6 import)."""

    def __init__(self):
        super().__init__()
        self.score_head = nn.Sequential(
            nn.Linear(REWARD_FEATURE_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, features):
        return self.score_head(features).squeeze(-1)


def load_segment_features(seg_dir, stats):
    features = stats.get("reward_features")
    if features is not None and len(features) == REWARD_FEATURE_DIM:
        return np.asarray(features, dtype=np.float32)

    states_path = os.path.join(seg_dir, "states.npy")
    actions_path = os.path.join(seg_dir, "actions.npy")
    if not (os.path.exists(states_path) and os.path.exists(actions_path)):
        return None

    states = np.load(states_path)
    actions = np.load(actions_path)
    summary = stats.get("summary", {})
    return compute_reward_features(
        states,
        actions,
        crashed=bool(summary.get("crashed", False)),
        out_of_road=bool(summary.get("out_of_road", False)),
    )


def load_labelled_segments():
    segment_dirs = sorted(
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    )

    segments = []
    skipped_unlabelled = 0
    skipped_missing = 0

    for seg_name in segment_dirs:
        seg_dir = os.path.join(SEGMENTS_DIR, seg_name)
        stats_path = os.path.join(seg_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        if stats.get("score") is None:
            skipped_unlabelled += 1
            continue

        features = load_segment_features(seg_dir, stats)
        if features is None:
            skipped_missing += 1
            continue

        segments.append(
            {
                "segment_id": stats.get("segment_id", seg_name),
                "features": features,
                "score": float(stats["score"]),
            }
        )

    print(f"  ✅ Loaded  : {len(segments)} labelled segments")
    if skipped_unlabelled:
        print(f"  ⏭️  Skipped : {skipped_unlabelled} unlabelled segments")
    if skipped_missing:
        print(f"  ⚠️  Skipped : {skipped_missing} segments missing trajectory data")

    return segments


def train():
    print("\n🧠 Phase 5: Numeric Reward Model Training")
    print(f"   Device     : {DEVICE}")
    print(f"   Epochs     : {EPOCHS}")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   Input      : {REWARD_FEATURE_DIM} trajectory features")
    print("   Output     : scalar score only (0.0 → bad, 1.0 → good)\n")

    print("📂 Loading labelled segments...")
    segments = load_labelled_segments()

    if len(segments) < 5:
        print("\n❌ Not enough labelled segments.")
        print(f"   You have {len(segments)}. Need at least 5.")
        return

    good_count = sum(1 for s in segments if s["score"] > 0.5)
    bad_count = sum(1 for s in segments if s["score"] <= 0.5)
    print("\n  Label distribution:")
    print(f"    Good : {good_count}")
    print(f"    Bad  : {bad_count}\n")

    if len(segments) >= 10:
        train_segs, val_segs = train_test_split(
            segments,
            test_size=0.2,
            random_state=42,
            stratify=[1 if s["score"] > 0.5 else 0 for s in segments],
        )
    else:
        print("  ⚠️  Less than 10 segments — training on all, no val split.")
        train_segs = segments
        val_segs = []

    train_loader = DataLoader(DrivingSegmentDataset(train_segs), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = (
        DataLoader(DrivingSegmentDataset(val_segs), batch_size=BATCH_SIZE, shuffle=False)
        if val_segs
        else None
    )

    print(f"  Train : {len(train_segs)} samples")
    print(f"  Val   : {len(val_segs)} samples\n")

    print("🤖 Building reward model...")
    model = GPT2RewardModel().to(DEVICE)
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}\n")

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    print("🚀 Training...\n")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            features = batch["features"].to(DEVICE)
            scores = batch["score"].to(DEVICE)

            optimizer.zero_grad()
            pred_scores = model(features)
            loss = loss_fn(pred_scores, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        val_str = ""
        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    features = batch["features"].to(DEVICE)
                    scores = batch["score"].to(DEVICE)
                    pred_scores = model(features)
                    val_loss += loss_fn(pred_scores, scores).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_str = f"  Val loss: {avg_val_loss:.4f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_DIR, "reward_model_best.pth"))

        print(
            f"  Epoch [{epoch + 1:>3}/{EPOCHS}]  "
            f"Train loss: {avg_train_loss:.4f}"
            f"{val_str}"
        )

    final_path = os.path.join(MODEL_SAVE_DIR, "reward_model_final.pth")
    torch.save(model.state_dict(), final_path)

    print("\n✅ Done!")
    print(f"   Final model : {final_path}")
    if val_segs:
        print(f"   Best model  : {os.path.join(MODEL_SAVE_DIR, 'reward_model_best.pth')}")

    plot_loss(train_losses, val_losses)
    print("\n🚀 Ready for Phase 6 — RLHF Fine-tuning\n")


def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", color="blue")
    if val_losses:
        plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Reward Model — Training Loss")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(MODEL_SAVE_DIR, "training_loss.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Loss curve : {save_path}")


def test_inference():
    print("\n🔍 Testing reward model inference...\n")

    model = GPT2RewardModel().to(DEVICE)
    model_path = os.path.join(MODEL_SAVE_DIR, "reward_model_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_SAVE_DIR, "reward_model_final.pth")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_cases = [
        {
            "features": [8.2, 0.8, 0.12, 0.25, 0.03, 0.06, 0.01, 0.0, 0.0],
            "expected": "high score (good driving)",
        },
        {
            "features": [12.1, 3.2, 0.81, 1.30, 0.42, 0.35, 0.09, 1.0, 0.0],
            "expected": "low score (bad driving)",
        },
    ]

    print(f"  {'Features':<58} {'Score':>6}  Expected")
    print("  " + "─" * 90)

    for case in test_cases:
        x = torch.tensor([case["features"]], dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            score = model(x).item()
        short = str(case["features"])
        print(f"  {short:<58} {score:>6.3f}  ← {case['expected']}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_inference()
    else:
        train()
