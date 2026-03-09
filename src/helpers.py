# from collections import deque

# import gymnasium as gym
# import numpy as np
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from metadrive.component.sensors.rgb_camera import RGBCamera
# from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# import sys
# import os

# sys.path.append(os.path.join(os.path.dirname(__file__), "config"))

# from config import Config

# PHASE2_MODEL_NAME       = "ppo_metadrive_vision"
# PHASE2_VECNORM_FILENAME = "vecnormalize.pkl"


# def get_env_config(render: bool = False, start_seed: int = 0) -> dict:
#     return {
#         "use_render": render,
#         "num_scenarios": 500,
#         "start_seed": start_seed,
#         "map": 7,
#         "random_lane_width": True,
#         "random_agent_model": False,
#         "traffic_density": 0.1,
#         "accident_prob": 0.1,
#         "stack_size": 1,
#         "image_observation": True,
#         "norm_pixel": True,
#         "use_AI_protector": False,
#         "sensors": {
#             "rgb_camera": (RGBCamera, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
#         },
#         "vehicle_config": {
#             "image_source": "rgb_camera",
#             "lidar": {
#                 "num_lasers": 240,
#                 "distance": 50,
#                 "num_others": 0,
#             },
#             "side_detector": {
#                 "num_lasers": 4,
#                 "distance": 50,
#             },
#             "lane_line_detector": {
#                 "num_lasers": 4,
#                 "distance": 20,
#             },
#         },
#         "use_lateral_reward": True,
#         "driving_reward": 1.0,
#         "speed_reward": 0.5,
#         "out_of_road_penalty": 5.0,
#         "crash_vehicle_penalty": 5.0,
#         "crash_object_penalty": 5.0,
#         "horizon": 10000,
#         "interface_panel": [],
#     }

# # Phase 1 Model


# class MultiHeadExtractor(BaseFeaturesExtractor):
#     """
#     Separate processing heads for each modality in the flat observation vector.

#     Why separate heads instead of one big MLP:
#       - Each head only receives gradients relevant to its modality.
#         The lidar head learns obstacle geometry without nav signals
#         interfering, and vice versa.
#       - Embedding sizes match signal complexity: lidar gets 128-dim,
#         tiny 4-value detectors get 16-dim.
#       - The lidar head can be transplanted directly into Phase 2's
#         state branch — clean transfer of learned driving knowledge.

#     Slices are defined by the module-level constants at the top of
#     this file. Run inspect() to verify them for your setup.
#     """

#     def __init__(self, observation_space, features_dim: int = 128):
#         super().__init__(observation_space, features_dim)

#         # ── Ego state head (9-dim → 32-dim) ─────────────────────────────────
#         # Processes: lateral offset, heading error, speed (long+lat),
#         #            steering angle, yaw rate, and additional ego values.
#         # Small head — 9 values, but high-signal (the agent's own motion).
#         self.ego_head = nn.Sequential(
#             nn.Linear(Config.EGO_DIM, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#         )

#         # ── Lidar head (240-dim → 128-dim) ──────────────────────────────────
#         # Processes: 240 distance readings in a ring around the vehicle.
#         # Largest head — most data, most spatial complexity.
#         # This is the head that gets transferred to Phase 2.
#         self.lidar_head = nn.Sequential(
#             nn.Linear(Config.LIDAR_DIM, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.LayerNorm(128),
#             nn.ReLU(),
#         )

#         # ── Side detector head (4-dim → 16-dim) ─────────────────────────────
#         # Processes: proximity to left/right road edges.
#         # Tiny head — 4 values encoding binary-ish road boundary info.
#         self.side_head = nn.Sequential(
#             nn.Linear(Config.SIDE_DIM, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.LayerNorm(16),
#             nn.ReLU(),
#         )

#         # ── Lane line detector head (4-dim → 16-dim) ────────────────────────
#         # Processes: distance to nearest lane lines (left and right).
#         # Similar to side head but for lane marking positions specifically.
#         self.lane_head = nn.Sequential(
#             nn.Linear(Config.LANE_DIM, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#             nn.Linear(32, 16),
#             nn.LayerNorm(16),
#             nn.ReLU(),
#         )

#         # ── Navigation head (5-dim → 32-dim) ────────────────────────────────
#         # Processes: heading to next waypoint, distance to destination,
#         #            route completion progress, etc.
#         # Needs enough capacity to learn "where should I go next."
#         self.nav_head = nn.Sequential(
#             nn.Linear(Config.NAV_DIM, 64),
#             nn.LayerNorm(64),
#             nn.ReLU(),
#             nn.Linear(64, 32),
#             nn.LayerNorm(32),
#             nn.ReLU(),
#         )

#         # ── Fusion layer ─────────────────────────────────────────────────────
#         # Combines all head outputs: 32 + 128 + 16 + 16 + 32 = 224-dim
#         combined_dim = 32 + 128 + 16 + 16 + 32  # = 224

#         self.fusion = nn.Sequential(
#             nn.Linear(combined_dim, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(256, features_dim),
#             nn.LayerNorm(features_dim),
#             nn.ReLU(),
#         )

#     def forward(self, obs: torch.Tensor) -> torch.Tensor:
#         # Slice the flat observation into each modality
#         ego   = obs[:, Config.EGO_START   : Config.EGO_START   + Config.EGO_DIM  ]
#         lidar = obs[:, Config.LIDAR_START : Config.LIDAR_START + Config.LIDAR_DIM]
#         side  = obs[:, Config.SIDE_START  : Config.SIDE_START  + Config.SIDE_DIM ]
#         lane  = obs[:, Config.LANE_START  : Config.LANE_START  + Config.LANE_DIM ]
#         nav   = obs[:, Config.NAV_START   : Config.NAV_START   + Config.NAV_DIM  ]

#         # Each head processes its own modality independently
#         ego_feat  = self.ego_head(ego)
#         lidar_feat = self.lidar_head(lidar)
#         side_feat = self.side_head(side)
#         lane_feat = self.lane_head(lane)
#         nav_feat  = self.nav_head(nav)

#         # Concatenate all embeddings and fuse
#         combined = torch.cat([ego_feat, lidar_feat, side_feat, lane_feat, nav_feat], dim=1)
#         return self.fusion(combined)




# class SmoothnessWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env, max_steer_delta=0.08, jerk_coef=0.5, steer_mag_coef=0.02):
#         super().__init__(env)
#         self.max_steer_delta = float(max_steer_delta)
#         self.jerk_coef = float(jerk_coef)
#         self.steer_mag_coef = float(steer_mag_coef)
#         self.prev_steer = 0.0

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self.prev_steer = 0.0
#         return obs, info

#     def step(self, action):
#         a = np.array(action, dtype=np.float32).copy()

#         steer = float(a[0])
#         steer = np.clip(
#             self.prev_steer + np.clip(steer - self.prev_steer, -self.max_steer_delta, self.max_steer_delta),
#             -1.0,
#             1.0,
#         )
#         a[0] = steer

#         obs, reward, term, trunc, info = self.env.step(a)

#         jerk = steer - self.prev_steer
#         reward -= self.jerk_coef * (jerk ** 2) + self.steer_mag_coef * (steer ** 2)

#         self.prev_steer = steer
#         return obs, reward, term, trunc, info


# class StateStackWrapper(gym.Wrapper):
#     def __init__(self, env: gym.Env, n_stack: int = Config.N_STACK):
#         super().__init__(env)
#         self.n_stack = n_stack

#         orig_state_space = env.observation_space["state"]
#         self._buffer = deque(maxlen=n_stack)

#         new_state_space = gym.spaces.Box(
#             low=np.tile(orig_state_space.low, n_stack),
#             high=np.tile(orig_state_space.high, n_stack),
#             dtype=np.float32,
#         )
#         self.observation_space = gym.spaces.Dict({
#             **{k: v for k, v in env.observation_space.spaces.items() if k != "state"},
#             "state": new_state_space,
#         })

#     def reset(self, **kwargs):
#         obs, info = self.env.reset(**kwargs)
#         self._buffer.clear()
#         for _ in range(self.n_stack):
#             self._buffer.append(obs["state"].astype(np.float32))
#         stacked = np.concatenate(list(self._buffer), axis=0)
#         return {**obs, "state": stacked}, info

#     def step(self, action):
#         obs, reward, terminated, truncated, info = self.env.step(action)
#         self._buffer.append(obs["state"].astype(np.float32))
#         stacked = np.concatenate(list(self._buffer), axis=0)
#         return {**obs, "state": stacked}, reward, terminated, truncated, info


# # ── ResNet18 feature extractor (frozen backbone) ──────────────────────────────
# # Pretrained ResNet18 is used as the image encoder.
# # The backbone is FROZEN — no gradients flow through it — so only the
# # projection head + MLP policy head train.  This halves VRAM usage vs
# # fine-tuning and is safe on a 6 GB GPU at 224×224.
# class ResNet18Extractor(BaseFeaturesExtractor):
#     def __init__(
#         self,
#         observation_space,
#         cnn_features:   int = 256,
#         mlp_features:   int = 128,
#         fusion_hidden:  int = 256,
#         fused_features: int = 256,
#     ):
#         super().__init__(observation_space, features_dim=fused_features)

#         state_dim = observation_space["state"].shape[0]

#         # ── frozen ResNet18 backbone ──────────────────────────────────────
#         resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
#         # Strip the final FC classifier — output is (B, 512, 1, 1)
#         self.backbone = nn.Sequential(*list(resnet.children())[:-1])
#         for param in self.backbone.parameters():
#             param.requires_grad = False   # frozen = no gradients = saves VRAM

#         # Projection: 512 → cnn_features
#         self.cnn_proj = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(512, cnn_features),
#             nn.LayerNorm(cnn_features),
#             nn.ReLU(),
#         )

#         # ── state MLP (same as before) ────────────────────────────────────
#         self.mlp = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, mlp_features),
#             nn.LayerNorm(mlp_features),
#             nn.ReLU(),
#         )

#         # ── gated fusion (same as before) ─────────────────────────────────
#         combined_dim = cnn_features + mlp_features
#         self.image_gate = nn.Sequential(
#             nn.Linear(combined_dim, cnn_features),
#             nn.Sigmoid(),
#         )
#         self.state_gate = nn.Sequential(
#             nn.Linear(combined_dim, mlp_features),
#             nn.Sigmoid(),
#         )
#         self.fusion = nn.Sequential(
#             nn.Linear(combined_dim, fusion_hidden),
#             nn.LayerNorm(fusion_hidden),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(fusion_hidden, fused_features),
#             nn.LayerNorm(fused_features),
#             nn.ReLU(),
#         )

#         # ImageNet normalisation constants (ResNet expects these)
#         self.register_buffer(
#             "img_mean",
#             torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
#         )
#         self.register_buffer(
#             "img_std",
#             torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
#         )

#     def forward(self, observations: dict) -> torch.Tensor:
#         # ── image preprocessing ───────────────────────────────────────────
#         img = observations["image"]         # (B, H, W, C, N_STACK) or (B, H, W, C)
#         # Collapse any trailing stack dimension
#         while img.dim() > 4:
#             img = img[..., -1]              # (B, H, W, C)
#         img = img.permute(0, 3, 1, 2).float()   # (B, C, H, W)  values in [0,1]

#         # Apply ImageNet normalisation
#         img = (img - self.img_mean) / self.img_std

#         # ── forward through frozen backbone ──────────────────────────────
#         with torch.no_grad():
#             feat = self.backbone(img)       # (B, 512, 1, 1)
#         img_feat = self.cnn_proj(feat)      # (B, cnn_features)

#         # ── state branch ─────────────────────────────────────────────────
#         state    = observations["state"].float()
#         st_feat  = self.mlp(state)          # (B, mlp_features)

#         # ── gated fusion ─────────────────────────────────────────────────
#         combined   = torch.cat([img_feat, st_feat], dim=1)
#         gated_img  = img_feat * self.image_gate(combined)
#         gated_st   = st_feat  * self.state_gate(combined)
#         fused_input = torch.cat([gated_img, gated_st], dim=1)

#         return self.fusion(fused_input)


# # ── kept for reference / fallback ────────────────────────────────────────────
# class DrivingCNNMLP(BaseFeaturesExtractor):
#     def __init__(
#         self,
#         observation_space,
#         cnn_features: int = 256,
#         mlp_features: int = 128,
#         fusion_hidden: int = 256,
#         fused_features: int = 256,
#     ):
#         super().__init__(observation_space, features_dim=fused_features)

#         state_dim = observation_space["state"].shape[0]

#         self.cnn = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(256, cnn_features),
#             nn.LayerNorm(cnn_features),
#             nn.ReLU(),
#         )

#         self.mlp = nn.Sequential(
#             nn.Linear(state_dim, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, 256),
#             nn.LayerNorm(256),
#             nn.ReLU(),
#             nn.Linear(256, mlp_features),
#             nn.LayerNorm(mlp_features),
#             nn.ReLU(),
#         )

#         combined_dim = cnn_features + mlp_features
#         self.image_gate = nn.Sequential(
#             nn.Linear(combined_dim, cnn_features),
#             nn.Sigmoid(),
#         )
#         self.state_gate = nn.Sequential(
#             nn.Linear(combined_dim, mlp_features),
#             nn.Sigmoid(),
#         )
#         self.fusion = nn.Sequential(
#             nn.Linear(combined_dim, fusion_hidden),
#             nn.LayerNorm(fusion_hidden),
#             nn.ReLU(),
#             nn.Dropout(p=0.1),
#             nn.Linear(fusion_hidden, fused_features),
#             nn.LayerNorm(fused_features),
#             nn.ReLU(),
#         )

#     def forward(self, observations: dict) -> torch.Tensor:
#         img = observations["image"]
#         img = img[..., 0]
#         img = img.permute(0, 3, 1, 2).float()

#         state = observations["state"].float()

#         img_feat = self.cnn(img)
#         st_feat  = self.mlp(state)

#         combined   = torch.cat([img_feat, st_feat], dim=1)
#         gated_img  = img_feat * self.image_gate(combined)
#         gated_st   = st_feat  * self.state_gate(combined)
#         fused_input = torch.cat([gated_img, gated_st], dim=1)

#         return self.fusion(fused_input)


# def compute_reward_features(states, actions, crashed: bool, out_of_road: bool):
#     """Compute numeric segment features from trajectory tensors."""
#     states  = np.asarray(states,  dtype=np.float32)
#     actions = np.asarray(actions, dtype=np.float32)

#     if states.ndim != 2 or states.shape[1] < 6:
#         raise ValueError(f"Expected states shape (T, >=6), got {states.shape}")
#     if actions.ndim == 1:
#         actions = actions.reshape(-1, 1)
#     if actions.ndim != 2 or actions.shape[0] != states.shape[0]:
#         raise ValueError(f"Expected actions shape (T, A) aligned with states, got {actions.shape}")

#     speed    = states[:, 2]
#     lateral  = states[:, 0]
#     yaw_rate = states[:, 5]
#     steering = actions[:, 0]

#     steering_diff = np.diff(steering) if steering.shape[0] > 1 else np.array([0.0], dtype=np.float32)

#     features = np.array(
#         [
#             float(np.mean(speed)),
#             float(np.std(speed)),
#             float(np.mean(np.abs(lateral))),
#             float(np.max(np.abs(lateral))),
#             float(np.mean(np.abs(steering_diff))),
#             float(np.std(steering)),
#             float(np.mean(yaw_rate)),
#             float(bool(crashed)),
#             float(bool(out_of_road)),
#         ],
#         dtype=np.float32,
#     )
#     return features

from collections import deque

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from metadrive.component.sensors.rgb_camera import RGBCamera
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "config"))

from config import Config

PHASE2_MODEL_NAME       = "ppo_metadrive_vision"
PHASE2_VECNORM_FILENAME = "vecnormalize.pkl"


def get_env_config(render: bool = False, start_seed: int = 0) -> dict:
    return {
        "use_render": render,
        "num_scenarios": 100,
        "start_seed": start_seed,
        "map": 7,
        "random_lane_width": True,
        "random_agent_model": False,
        "traffic_density": 0.1,
        "accident_prob": 0.1,
        "stack_size": 1,
        "image_observation": True,
        "norm_pixel": True,
        "use_AI_protector": False,
        "sensors": {
            "rgb_camera": (RGBCamera, Config.IMAGE_WIDTH, Config.IMAGE_HEIGHT),
        },
        "vehicle_config": {
            "image_source": "rgb_camera",
            "lidar": {
                "num_lasers": 240,
                "distance": 50,
                "num_others": 0,
            },
            "side_detector": {
                "num_lasers": 4,
                "distance": 50,
            },
            "lane_line_detector": {
                "num_lasers": 4,
                "distance": 20,
            },
        },
        "use_lateral_reward": True,
        "driving_reward": 1.0,
        "speed_reward": 0.25,
        "out_of_road_penalty": 5.0,
        "crash_vehicle_penalty": 5.0,
        "crash_object_penalty": 5.0,
        "horizon": 10000,
        "interface_panel": [],
    }

# Phase 1 Model


class MultiHeadExtractor(BaseFeaturesExtractor):
    """
    Separate processing heads for each modality in the flat observation vector.

    Why separate heads instead of one big MLP:
      - Each head only receives gradients relevant to its modality.
        The lidar head learns obstacle geometry without nav signals
        interfering, and vice versa.
      - Embedding sizes match signal complexity: lidar gets 128-dim,
        tiny 4-value detectors get 16-dim.
      - The lidar head can be transplanted directly into Phase 2's
        state branch — clean transfer of learned driving knowledge.

    Slices are defined by the module-level constants at the top of
    this file. Run inspect() to verify them for your setup.
    """

    def __init__(self, observation_space, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        # ── Ego state head (9-dim → 32-dim) ─────────────────────────────────
        # Processes: lateral offset, heading error, speed (long+lat),
        #            steering angle, yaw rate, and additional ego values.
        # Small head — 9 values, but high-signal (the agent's own motion).
        self.ego_head = nn.Sequential(
            nn.Linear(Config.EGO_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # ── Lidar head (240-dim → 128-dim) ──────────────────────────────────
        # Processes: 240 distance readings in a ring around the vehicle.
        # Largest head — most data, most spatial complexity.
        # This is the head that gets transferred to Phase 2.
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

        # ── Side detector head (4-dim → 16-dim) ─────────────────────────────
        # Processes: proximity to left/right road edges.
        # Tiny head — 4 values encoding binary-ish road boundary info.
        self.side_head = nn.Sequential(
            nn.Linear(Config.SIDE_DIM, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # ── Lane line detector head (4-dim → 16-dim) ────────────────────────
        # Processes: distance to nearest lane lines (left and right).
        # Similar to side head but for lane marking positions specifically.
        self.lane_head = nn.Sequential(
            nn.Linear(Config.LANE_DIM, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ReLU(),
        )

        # ── Navigation head (5-dim → 32-dim) ────────────────────────────────
        # Processes: heading to next waypoint, distance to destination,
        #            route completion progress, etc.
        # Needs enough capacity to learn "where should I go next."
        self.nav_head = nn.Sequential(
            nn.Linear(Config.NAV_DIM, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
        )

        # ── Fusion layer ─────────────────────────────────────────────────────
        # Combines all head outputs: 32 + 128 + 16 + 16 + 32 = 224-dim
        combined_dim = 32 + 128 + 16 + 16 + 32  # = 224

        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Slice the flat observation into each modality
        ego   = obs[:, Config.EGO_START   : Config.EGO_START   + Config.EGO_DIM  ]
        lidar = obs[:, Config.LIDAR_START : Config.LIDAR_START + Config.LIDAR_DIM]
        side  = obs[:, Config.SIDE_START  : Config.SIDE_START  + Config.SIDE_DIM ]
        lane  = obs[:, Config.LANE_START  : Config.LANE_START  + Config.LANE_DIM ]
        nav   = obs[:, Config.NAV_START   : Config.NAV_START   + Config.NAV_DIM  ]

        # Each head processes its own modality independently
        ego_feat  = self.ego_head(ego)
        lidar_feat = self.lidar_head(lidar)
        side_feat = self.side_head(side)
        lane_feat = self.lane_head(lane)
        nav_feat  = self.nav_head(nav)

        # Concatenate all embeddings and fuse
        combined = torch.cat([ego_feat, lidar_feat, side_feat, lane_feat, nav_feat], dim=1)
        return self.fusion(combined)




class FullStateWrapper(gym.Wrapper):
    """
    Reconstructs full 264-dim state alongside the RGB image.
    perceive() returns a tuple — cloud points are at index [0].
    Layout: ego(9) + lidar(240) + side(4) + lane(4) + nav(7) = 264
    """

    _FULL_DIM = (
        Config.EGO_DIM + Config.LIDAR_DIM +
        Config.SIDE_DIM + Config.LANE_DIM + Config.NAV_DIM
    )

    def __init__(self, env: gym.Env):
        super().__init__(env)
        orig = env.observation_space
        self.observation_space = gym.spaces.Dict({
            **{k: v for k, v in orig.spaces.items() if k != "state"},
            "state": gym.spaces.Box(
                low   = -np.ones(self._FULL_DIM, dtype=np.float32),
                high  =  np.ones(self._FULL_DIM, dtype=np.float32),
                dtype = np.float32,
            ),
        })

    def _get_inner_env(self):
        e = self.env
        while hasattr(e, "env"):
            e = e.env
        return e

    def _build_full_state(self, obs_state: np.ndarray) -> np.ndarray:
        inner  = self._get_inner_env()
        engine = inner.engine
        agent  = inner.agent
        pw     = engine.physics_world

        ego = np.array(obs_state[:Config.EGO_DIM],  dtype=np.float32)
        nav = np.array(obs_state[-Config.NAV_DIM:], dtype=np.float32)

        # perceive() returns a tuple — cloud points are at [0]
        lidar = np.array(
            engine.get_sensor("lidar").perceive(
                agent,
                physics_world = pw.dynamic_world,
                num_lasers    = agent.config["lidar"]["num_lasers"],
                distance      = agent.config["lidar"]["distance"],
                show          = False,
            )[0],
            dtype=np.float32,
        )

        side = np.array(
            engine.get_sensor("side_detector").perceive(
                agent,
                physics_world = pw.static_world,
                num_lasers    = agent.config["side_detector"]["num_lasers"],
                distance      = agent.config["side_detector"]["distance"],
                show          = False,
            )[0],
            dtype=np.float32,
        )

        lane = np.array(
            engine.get_sensor("lane_line_detector").perceive(
                agent,
                physics_world = pw.static_world,
                num_lasers    = agent.config["lane_line_detector"]["num_lasers"],
                distance      = agent.config["lane_line_detector"]["distance"],
                show          = False,
            )[0],
            dtype=np.float32,
        )

        full = np.concatenate([ego, lidar, side, lane, nav])

        assert full.shape[0] == self._FULL_DIM, (
            f"FullStateWrapper dim mismatch: got {full.shape[0]}, "
            f"expected {self._FULL_DIM}. "
            f"ego={ego.shape} lidar={lidar.shape} "
            f"side={side.shape} lane={lane.shape} nav={nav.shape}"
        )
        return full

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = dict(obs)
        obs["state"] = self._build_full_state(obs["state"])
        return obs, info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        obs = dict(obs)
        obs["state"] = self._build_full_state(obs["state"])
        return obs, reward, term, trunc, info


class SmoothnessWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, max_steer_delta=0.08, jerk_coef=0.5, steer_mag_coef=0.02):
        super().__init__(env)
        self.max_steer_delta = float(max_steer_delta)
        self.jerk_coef = float(jerk_coef)
        self.steer_mag_coef = float(steer_mag_coef)
        self.prev_steer = 0.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev_steer = 0.0
        return obs, info

    def step(self, action):
        a = np.array(action, dtype=np.float32).copy()

        steer = float(a[0])
        steer = np.clip(
            self.prev_steer + np.clip(steer - self.prev_steer, -self.max_steer_delta, self.max_steer_delta),
            -1.0,
            1.0,
        )
        a[0] = steer

        obs, reward, term, trunc, info = self.env.step(a)

        jerk = steer - self.prev_steer
        reward -= self.jerk_coef * (jerk ** 2) + self.steer_mag_coef * (steer ** 2)

        self.prev_steer = steer
        return obs, reward, term, trunc, info


class StateStackWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, n_stack: int = Config.N_STACK):
        super().__init__(env)
        self.n_stack = n_stack

        orig_state_space = env.observation_space["state"]
        self._buffer = deque(maxlen=n_stack)

        new_state_space = gym.spaces.Box(
            low=np.tile(orig_state_space.low, n_stack),
            high=np.tile(orig_state_space.high, n_stack),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Dict({
            **{k: v for k, v in env.observation_space.spaces.items() if k != "state"},
            "state": new_state_space,
        })

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._buffer.clear()
        for _ in range(self.n_stack):
            self._buffer.append(obs["state"].astype(np.float32))
        stacked = np.concatenate(list(self._buffer), axis=0)
        return {**obs, "state": stacked}, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._buffer.append(obs["state"].astype(np.float32))
        stacked = np.concatenate(list(self._buffer), axis=0)
        return {**obs, "state": stacked}, reward, terminated, truncated, info


# ── ResNet18 feature extractor (frozen backbone) ──────────────────────────────
# Pretrained ResNet18 is used as the image encoder.
# The backbone is FROZEN — no gradients flow through it — so only the
# projection head + MLP policy head train.  This halves VRAM usage vs
# fine-tuning and is safe on a 6 GB GPU at 224×224.
class ResNet18Extractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        cnn_features:   int = 256,
        mlp_features:   int = 128,
        fusion_hidden:  int = 256,
        fused_features: int = 256,
    ):
        super().__init__(observation_space, features_dim=fused_features)

        state_dim = observation_space["state"].shape[0]

        # ── frozen ResNet18 backbone ──────────────────────────────────────
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Strip the final FC classifier — output is (B, 512, 1, 1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False   # frozen = no gradients = saves VRAM

        # Projection: 512 → cnn_features
        self.cnn_proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, cnn_features),
            nn.LayerNorm(cnn_features),
            nn.ReLU(),
        )

        # ── state MLP (same as before) ────────────────────────────────────
        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, mlp_features),
            nn.LayerNorm(mlp_features),
            nn.ReLU(),
        )

        # ── gated fusion (same as before) ─────────────────────────────────
        combined_dim = cnn_features + mlp_features
        self.image_gate = nn.Sequential(
            nn.Linear(combined_dim, cnn_features),
            nn.Sigmoid(),
        )
        self.state_gate = nn.Sequential(
            nn.Linear(combined_dim, mlp_features),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fusion_hidden, fused_features),
            nn.LayerNorm(fused_features),
            nn.ReLU(),
        )

        # ImageNet normalisation constants (ResNet expects these)
        self.register_buffer(
            "img_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "img_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        # ── image preprocessing ───────────────────────────────────────────
        img = observations["image"]         # (B, H, W, C, N_STACK) or (B, H, W, C)
        # Collapse any trailing stack dimension
        while img.dim() > 4:
            img = img[..., -1]              # (B, H, W, C)
        img = img.permute(0, 3, 1, 2).float()   # (B, C, H, W)  values in [0,1]

        # Apply ImageNet normalisation
        img = (img - self.img_mean) / self.img_std

        # ── forward through frozen backbone ──────────────────────────────
        with torch.no_grad():
            feat = self.backbone(img)       # (B, 512, 1, 1)
        img_feat = self.cnn_proj(feat)      # (B, cnn_features)

        # ── state branch ─────────────────────────────────────────────────
        state    = observations["state"].float()
        st_feat  = self.mlp(state)          # (B, mlp_features)

        # ── gated fusion ─────────────────────────────────────────────────
        combined   = torch.cat([img_feat, st_feat], dim=1)
        gated_img  = img_feat * self.image_gate(combined)
        gated_st   = st_feat  * self.state_gate(combined)
        fused_input = torch.cat([gated_img, gated_st], dim=1)

        return self.fusion(fused_input)


# ── kept for reference / fallback ────────────────────────────────────────────
class DrivingCNNMLP(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space,
        cnn_features: int = 256,
        mlp_features: int = 128,
        fusion_hidden: int = 256,
        fused_features: int = 256,
    ):
        super().__init__(observation_space, features_dim=fused_features)

        state_dim = observation_space["state"].shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, cnn_features),
            nn.LayerNorm(cnn_features),
            nn.ReLU(),
        )

        self.mlp = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, mlp_features),
            nn.LayerNorm(mlp_features),
            nn.ReLU(),
        )

        combined_dim = cnn_features + mlp_features
        self.image_gate = nn.Sequential(
            nn.Linear(combined_dim, cnn_features),
            nn.Sigmoid(),
        )
        self.state_gate = nn.Sequential(
            nn.Linear(combined_dim, mlp_features),
            nn.Sigmoid(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(combined_dim, fusion_hidden),
            nn.LayerNorm(fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(fusion_hidden, fused_features),
            nn.LayerNorm(fused_features),
            nn.ReLU(),
        )

    def forward(self, observations: dict) -> torch.Tensor:
        img = observations["image"]
        img = img[..., 0]
        img = img.permute(0, 3, 1, 2).float()

        state = observations["state"].float()

        img_feat = self.cnn(img)
        st_feat  = self.mlp(state)

        combined   = torch.cat([img_feat, st_feat], dim=1)
        gated_img  = img_feat * self.image_gate(combined)
        gated_st   = st_feat  * self.state_gate(combined)
        fused_input = torch.cat([gated_img, gated_st], dim=1)

        return self.fusion(fused_input)


def compute_reward_features(states, actions, crashed: bool, out_of_road: bool):
    """Compute numeric segment features from trajectory tensors."""
    states  = np.asarray(states,  dtype=np.float32)
    actions = np.asarray(actions, dtype=np.float32)

    if states.ndim != 2 or states.shape[1] < 6:
        raise ValueError(f"Expected states shape (T, >=6), got {states.shape}")
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    if actions.ndim != 2 or actions.shape[0] != states.shape[0]:
        raise ValueError(f"Expected actions shape (T, A) aligned with states, got {actions.shape}")

    speed    = states[:, 2]
    lateral  = states[:, 0]
    yaw_rate = states[:, 5]
    steering = actions[:, 0]

    steering_diff = np.diff(steering) if steering.shape[0] > 1 else np.array([0.0], dtype=np.float32)

    features = np.array(
        [
            float(np.mean(speed)),
            float(np.std(speed)),
            float(np.mean(np.abs(lateral))),
            float(np.max(np.abs(lateral))),
            float(np.mean(np.abs(steering_diff))),
            float(np.std(steering)),
            float(np.mean(yaw_rate)),
            float(bool(crashed)),
            float(bool(out_of_road)),
        ],
        dtype=np.float32,
    )
    return features