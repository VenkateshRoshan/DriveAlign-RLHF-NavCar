# 🚗 DriveAlign - RLHF-Based Autonomous Driving with Natural Language Feedback

> Training a vision-based autonomous driving agent using Reinforcement Learning from Human Feedback (RLHF), where humans guide the agent using natural language instead of manual reward engineering.

---

## 📌 Project Overview

DriveAlign is an end-to-end RLHF pipeline for autonomous driving inside the **MetaDrive** simulator. The agent learns to drive using a combination of:

- **Reinforcement Learning (PPO)** for base driving behavior
- **Vision (RGB Camera)** so the agent literally sees the road
- **Lidar/State sensors** for spatial awareness
- **Natural Language Feedback** from humans ("you were too aggressive in turns")
- **LLM-powered preference labeling** to convert language into structured reward signals
- **A learned reward model** trained on human preferences
- **RLHF fine-tuning** where the agent optimizes for what the human values, not just lane-following rules

---

## 🧠 Why This Project?

Traditional RL agents optimize for hand-crafted rewards like "stay in lane" or "maintain speed." These are rigid and don't capture nuanced human preferences like smoothness, aggression level, or safety margins.

RLHF solves this by letting humans express preferences in natural language and training the agent to optimize for those preferences directly. This is the same core idea behind how large language models like GPT are aligned - applied here to physical driving behavior.

---

## 🗂️ Project Structure

```
DriveAlign-RLHF-NavCar/
│
├── rlhf_train.py            # Phase 1 - Base RL agent (lidar/vector obs only)
├── vision_lidar_train.py    # Phase 2 - Vision + Lidar combined agent
│
├── models/                  # Saved Phase 1 model checkpoints
│   ├── ppo_metadrive_phase1_final.zip
│   └── vecnormalize.pkl
│
├── models_vision/           # Saved Phase 2 vision model checkpoints
│   ├── ppo_metadrive_vision_final.zip
│   └── vecnormalize.pkl
│
├── logs/                    # TensorBoard logs for Phase 1
├── logs_vision/             # TensorBoard logs for Phase 2
│
└── metadrive-env/           # Python virtual environment
```

---

## ⚙️ Tech Stack

| Component | Tool |
|---|---|
| Simulator | MetaDrive 0.4.3 |
| RL Algorithm | PPO (Stable-Baselines3) |
| Neural Network | PyTorch |
| Vision Encoder | Custom CNN (NatureCNN-style) |
| Lidar Encoder | Custom MLP |
| LLM Feedback | GPT-4o / Mistral (Phase 4) |
| Reward Model | Small NN trained on preferences |
| Logging | TensorBoard |
| Language | Python 3.11 |
| GPU | CUDA-enabled (6GB VRAM) |

---

## 🔄 Full Pipeline - Phase by Phase

### Phase 1 - Base RL Agent (Vector Observations)

The first step is training a vanilla PPO agent using only sensor/lidar data - no vision. The agent receives a 259-dimensional vector containing ego state, lidar readings, and navigation info. It learns to drive using MetaDrive's built-in reward (lane following + speed).

```
Observation (259-dim vector)
        ↓
Custom MLP:
  Linear(259 → 256) → ReLU → LayerNorm
  Linear(256 → 128) → ReLU → Dropout(0.1)
  Linear(128 → 128) → ReLU
        ↓
Policy Head [64 → 64]
        ↓
Action: [steering, throttle] ∈ [-1, 1]
```

**Goal:** Agent reliably reaches the destination without crashing on varied track layouts.

---

### Phase 2 - Vision + Lidar Agent

The observation is upgraded to include a live RGB camera feed alongside the lidar state vector. A combined CNN + MLP architecture processes both streams in parallel and merges them before the policy head.

```
┌─────────────────────────────┐   ┌──────────────────┐
│  Camera Frame (84×84×3)     │   │  State (19-dim)   │
│  squeeze stack dim          │   │  lidar + ego +    │
│  permute → (3, 84, 84)      │   │  navigation       │
│                             │   └────────┬─────────┘
│  Conv2d(3→32, 8×8, s=4)    │            │
│  Conv2d(32→64, 4×4, s=2)   │   Linear(19 → 128) → ReLU
│  Conv2d(64→64, 3×3, s=1)   │   LayerNorm(128)
│  Flatten → Linear → 128-dim│   Linear(128 → 64) → ReLU
└─────────────┬───────────────┘            │
              │                            │
              └──────────┬─────────────────┘
                         │
                   Concat (192-dim)
                         │
                 Linear(192 → 128) → ReLU
                 Linear(128 → 64)  → ReLU
                    ┌────┴────┐
                    │         │
               Actor Head  Critic Head
               (steer,      (value
               throttle)    estimate)
```

**Policy:** `MultiInputPolicy` (handles dict observation spaces natively in SB3)

---

### Phase 3 - Segment Recording

Before RLHF begins, the agent's driving behavior is recorded in segments (5–10 seconds each). Each segment stores:

- Sequence of observations (image + state)
- Actions taken
- Per-step rewards
- Key metrics: lane deviation, crash events, speed, route completion

These segments are what humans will later review and compare.

---

### Phase 4 - LLM Feedback Layer

Instead of clicking "A is better than B," the human types natural language feedback like:

> *"The car was too aggressive in the turn and almost went off road"*
> *"Good lane keeping but braking was too sudden"*

An LLM (GPT-4o or local Mistral) reads this feedback and converts it into structured preference labels:

```
Human: "too aggressive in the turn"
        ↓
LLM processes + maps to recent segments
        ↓
Output: {
  "segment_12": "bad",
  "segment_13": "good",
  "reason": "excessive steering angle in curve"
}
```

This is the novel contribution of the project - replacing manual click-based preferences with natural language understanding.

---

### Phase 5 - Reward Model Training

The structured preference labels from Phase 4 are used to train a small neural network that takes a driving segment as input and outputs a scalar reward score.

```
Driving Segment (obs + actions)
        ↓
Segment Encoder (MLP/CNN)
        ↓
Scalar Score → "how good was this driving?"
```

Training objective: the reward model should score "good" segments higher than "bad" ones, using a Bradley-Terry pairwise ranking loss.

---

### Phase 6 - RLHF Fine-Tuning

The Phase 2 vision agent is retrained using PPO, but MetaDrive's built-in reward is replaced with the learned reward model from Phase 5.

```
Agent drives → segment recorded
        ↓
Learned Reward Model scores the segment
        ↓
PPO updates policy based on human-aligned reward
        ↓
Agent now optimizes for what the human said
```

---

### Phase 7 - Metrics + Visualization

Before vs after RLHF comparison using three key metrics:

| Metric | Before RLHF | After RLHF |
|---|---|---|
| Lane Deviation | Higher | Lower ↓ |
| Crash Rate | Higher | Lower ↓ |
| Smoothness (jerk) | Worse | Better ↑ |
| Route Completion | Partial | Higher ↑ |

TensorBoard plots show these trends over training timesteps, providing visual proof of improvement from human feedback.

---

## 🚀 Getting Started

### 1. Clone and set up environment

```bash
git clone <your-repo-url>
cd DriveAlign-RLHF-NavCar

python3 -m venv metadrive-env
source metadrive-env/bin/activate

pip install --upgrade pip
pip install metadrive-simulator stable-baselines3 tensorboard gymnasium torch torchvision
```

### 2. Run Phase 1 (Base Agent)

```bash
# Train
python rlhf_train.py

# Evaluate
python rlhf_train.py eval

# Monitor
tensorboard --logdir ./logs/
```

### 3. Run Phase 2 (Vision + Lidar)

```bash
# Inspect observations first
python vision_lidar_train.py inspect

# Train
python vision_lidar_train.py

# Evaluate
python vision_lidar_train.py eval

# Monitor
tensorboard --logdir ./logs_vision/
```

---

## 🌍 Environment Configuration

| Config Key | Value | Description |
|---|---|---|
| `num_scenarios` | 20 | Number of different track layouts |
| `map` | `SCSCSCS` | Alternating straight and curved roads |
| `traffic_density` | 0.2 | Light traffic |
| `accident_prob` | 0.2 | Occasional road obstacles |
| `random_lane_width` | True | Lane width varies per episode |
| `image_observation` | True | Enable camera frames |
| `stack_size` | 1 | Single frame (no stacking) |
| `norm_pixel` | True | Pixels normalized to [0, 1] |

---

## 📈 Training Hyperparameters

| Parameter | Value | Reason |
|---|---|---|
| Algorithm | PPO | Stable, sample efficient for continuous control |
| Learning Rate | 3e-4 | Standard PPO default |
| n_steps | 2048 | Rollout buffer size |
| batch_size | 64 | Mini-batch for gradient updates |
| n_epochs | 10 | PPO update epochs per rollout |
| gamma | 0.99 | Long-horizon discount |
| gae_lambda | 0.95 | Advantage estimation smoothing |
| clip_range | 0.2 | PPO clipping for stability |
| ent_coef | 0.01 | Encourages exploration |

---

## 📊 What Good Training Looks Like

In TensorBoard watch for:

- `ep_rew_mean` → should steadily **increase** over timesteps
- `ep_len_mean` → should **increase** (surviving longer before crashing)
- `explained_variance` → should stay **above 0.8** (good value estimation)
- `value_loss` → should **decrease** over time

---

## 🔮 Roadmap

- [x] Phase 1 - Base RL Agent (vector obs)
- [x] Phase 2 - Vision + Lidar Agent
- [ ] Phase 3 - Segment recording infrastructure
- [ ] Phase 4 - LLM feedback layer (GPT-4o / Mistral)
- [ ] Phase 5 - Reward model training
- [ ] Phase 6 - RLHF fine-tuning
- [ ] Phase 7 - Metrics dashboard + before/after comparison

---

## 👤 Author

Built as a research project demonstrating RLHF applied to autonomous driving - combining computer vision, reinforcement learning, and large language models in a unified pipeline.