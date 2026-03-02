import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Model, GPT2Tokenizer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SEGMENTS_DIR   = "./segments/"
MODEL_SAVE_DIR = "./reward_model/"
BATCH_SIZE     = 8
EPOCHS         = 20
LEARNING_RATE  = 2e-5
MAX_TOKEN_LEN  = 256
DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

class DrivingSegmentDataset(Dataset):
    """
    Each sample:
      input  → description + human feedback (text)
      target → scalar score between 0.0 and 1.0
                 good + high confidence → close to 1.0
                 bad  + high confidence → close to 0.0
    """

    def __init__(self, segments, tokenizer):
        self.samples   = segments
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seg = self.samples[idx]

        description    = seg["description"]
        human_feedback = seg.get("human_feedback", "")
        label          = 1 if seg["label"] == "good" else 0
        confidence     = float(seg.get("confidence", 0.5))

        # Score target
        score = confidence if label == 1 else 1.0 - confidence

        # Input text
        input_text = (
            f"Driving behaviour: {description} "
            f"Human feedback: {human_feedback}"
        )

        encoding = self.tokenizer(
            input_text,
            max_length     = MAX_TOKEN_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "score":          torch.tensor(score, dtype=torch.float),
        }

class GPT2RewardModel(nn.Module):
    """
    GPT-2 Small (117M) as text encoder.
    Single output: scalar score 0.0 to 1.0
    This score is used directly as reward in PPO (Phase 6).
    """

    def __init__(self):
        super().__init__()

        self.gpt2       = GPT2Model.from_pretrained("gpt2")
        hidden_size     = self.gpt2.config.hidden_size   # 768

        self.score_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),    # output clamped to 0.0 - 1.0
        )

    def forward(self, input_ids, attention_mask):
        outputs     = self.gpt2(
            input_ids      = input_ids,
            attention_mask = attention_mask,
        )
        # Last token hidden state → sees all previous tokens (causal LM)
        last_hidden = outputs.last_hidden_state[:, -1, :]
        score       = self.score_head(last_hidden).squeeze(-1)   # (B,)
        return score

def load_labelled_segments():
    segment_dirs = sorted([
        d for d in os.listdir(SEGMENTS_DIR)
        if os.path.isdir(os.path.join(SEGMENTS_DIR, d))
    ])

    segments = []
    skipped  = 0

    for seg_dir in segment_dirs:
        stats_path = os.path.join(SEGMENTS_DIR, seg_dir, "stats.json")
        if not os.path.exists(stats_path):
            continue

        with open(stats_path) as f:
            stats = json.load(f)

        if stats.get("label") is None:
            skipped += 1
            continue

        segments.append({
            "segment_id":     stats.get("segment_id", seg_dir),
            "description":    stats.get("description", ""),
            "human_feedback": stats.get("human_feedback", ""),
            "label":          stats["label"],
            "confidence":     stats.get("confidence", 0.5),
        })

    print(f"  ✅ Loaded  : {len(segments)} labelled segments")
    if skipped:
        print(f"  ⏭️  Skipped : {skipped} unlabelled segments")

    return segments

def train():
    print(f"\n🧠 Phase 5: GPT-2 Reward Model Training")
    print(f"   Device     : {DEVICE}")
    print(f"   Epochs     : {EPOCHS}")
    print(f"   Batch size : {BATCH_SIZE}")
    print(f"   Output     : scalar score only (0.0 → bad, 1.0 → good)\n")

    # ── Load data ──────────────────────────────────
    print("📂 Loading labelled segments...")
    segments = load_labelled_segments()

    if len(segments) < 5:
        print(f"\n❌ Not enough labelled segments.")
        print(f"   You have {len(segments)}. Need at least 5.")
        print(f"   Run run.py to collect and label more.")
        return

    good_count = sum(1 for s in segments if s["label"] == "good")
    bad_count  = sum(1 for s in segments if s["label"] == "bad")
    print(f"\n  Label distribution:")
    print(f"    Good : {good_count}")
    print(f"    Bad  : {bad_count}\n")

    # ── Tokenizer ─────────────────────────────────
    print("🔤 Loading GPT-2 tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # ── Train / val split ─────────────────────────
    if len(segments) >= 10:
        train_segs, val_segs = train_test_split(
            segments, test_size=0.2, random_state=42,
            stratify=[s["label"] for s in segments]
        )
    else:
        print("  ⚠️  Less than 10 segments — training on all, no val split.")
        train_segs = segments
        val_segs   = []

    train_dataset = DrivingSegmentDataset(train_segs, tokenizer)
    train_loader  = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    val_loader = None
    if val_segs:
        val_dataset = DrivingSegmentDataset(val_segs, tokenizer)
        val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print(f"  Train : {len(train_segs)} samples")
    print(f"  Val   : {len(val_segs)} samples\n")

    # ── Model ─────────────────────────────────────
    print("🤖 Loading GPT-2 Small (117M)...")
    model = GPT2RewardModel().to(DEVICE)
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}\n")

    # ── Loss + Optimizer ──────────────────────────
    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────
    train_losses  = []
    val_losses    = []
    best_val_loss = float("inf")

    print("🚀 Training...\n")

    for epoch in range(EPOCHS):
        # Train
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            scores         = batch["score"].to(DEVICE)

            optimizer.zero_grad()
            pred_scores = model(input_ids, attention_mask)
            loss        = loss_fn(pred_scores, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Val
        val_str = ""
        if val_loader:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    input_ids      = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)
                    scores         = batch["score"].to(DEVICE)
                    pred_scores    = model(input_ids, attention_mask)
                    val_loss      += loss_fn(pred_scores, scores).item()

            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            val_str = f"  Val loss: {avg_val_loss:.4f}"

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(),
                           os.path.join(MODEL_SAVE_DIR, "reward_model_best.pth"))

        print(f"  Epoch [{epoch+1:>3}/{EPOCHS}]  "
              f"Train loss: {avg_train_loss:.4f}"
              f"{val_str}")

    # ── Save ──────────────────────────────────────
    final_path = os.path.join(MODEL_SAVE_DIR, "reward_model_final.pth")
    torch.save(model.state_dict(), final_path)
    tokenizer.save_pretrained(MODEL_SAVE_DIR)

    print(f"\n✅ Done!")
    print(f"   Final model : {final_path}")
    if val_segs:
        print(f"   Best model  : {os.path.join(MODEL_SAVE_DIR, 'reward_model_best.pth')}")

    plot_loss(train_losses, val_losses)
    print(f"\n🚀 Ready for Phase 6 — RLHF Fine-tuning\n")

def plot_loss(train_losses, val_losses):
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, label="Train Loss", color="blue")
    if val_losses:
        plt.plot(val_losses, label="Val Loss", color="orange")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("GPT-2 Reward Model — Training Loss")
    plt.legend()
    plt.tight_layout()
    save_path = os.path.join(MODEL_SAVE_DIR, "training_loss.png")
    plt.savefig(save_path)
    plt.close()
    print(f"  📊 Loss curve : {save_path}")

def test_inference():
    """
    Simulates exactly how Phase 6 PPO will call the reward model.
    Input  → description text
    Output → scalar score (the reward)
    """
    print("\n🔍 Testing reward model inference...\n")

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_SAVE_DIR)
    tokenizer.pad_token = tokenizer.eos_token

    model      = GPT2RewardModel().to(DEVICE)
    model_path = os.path.join(MODEL_SAVE_DIR, "reward_model_best.pth")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODEL_SAVE_DIR, "reward_model_final.pth")

    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    test_cases = [
        {
            "description":    "The agent drove at good speed (avg 8.2 m/s) with slight lane deviation (avg 0.12m). Steering was very smooth. No incidents. Total reward: 45.2.",
            "human_feedback": "smooth driving, good lane keeping",
            "expected":       "high score (good driving)"
        },
        {
            "description":    "The agent drove at very fast (avg 12.1 m/s) with severe lane deviation (avg 0.81m). Steering was very aggressive. The agent crashed. Total reward: -12.3.",
            "human_feedback": "too aggressive in turns",
            "expected":       "low score (bad driving)"
        },
    ]

    print(f"  {'Description':<55} {'Score':>6}  Expected")
    print("  " + "─" * 80)

    for case in test_cases:
        input_text = (
            f"Driving behaviour: {case['description']} "
            f"Human feedback: {case['human_feedback']}"
        )

        encoding = tokenizer(
            input_text,
            max_length     = MAX_TOKEN_LEN,
            padding        = "max_length",
            truncation     = True,
            return_tensors = "pt",
        )

        with torch.no_grad():
            score = model(
                encoding["input_ids"].to(DEVICE),
                encoding["attention_mask"].to(DEVICE)
            ).item()

        desc_short = case["description"][:55]
        print(f"  {desc_short:<55} {score:>6.3f}  ← {case['expected']}")

    print(f"\n  ✅ This score is exactly what PPO uses as reward in Phase 6\n")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        test_inference()
    else:
        train()