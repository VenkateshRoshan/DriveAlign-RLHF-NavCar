import os

import cv2
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from helpers import N_STACK

VIDEO_DIR = "./outputs/videos/"
os.makedirs(VIDEO_DIR, exist_ok=True)


class VideoRecordingCallback(BaseCallback):
    def __init__(self, record_freq=100_000, video_dir=VIDEO_DIR, include_step_zero=True, verbose=0):
        super().__init__(verbose)
        self.record_freq       = record_freq
        self.video_dir         = video_dir
        self.include_step_zero = include_step_zero
        self.last_recorded     = -1

    def _on_training_start(self) -> None:
        if self.include_step_zero and self.num_timesteps == 0:
            self.last_recorded = 0
            self._record_video()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_recorded >= self.record_freq:
            self.last_recorded = self.num_timesteps
            self._record_video()
        return True

    def _record_video(self):
        step = self.num_timesteps
        path = os.path.join(self.video_dir, f"eval_step_{step:07d}.mp4")
        print(f"\n  📹 Recording eval video at step {step:,}...")

        env = self.training_env
        obs = env.reset()

        # obs["image"][0] shape is (84, 84, 3, N_STACK)
        # squeeze the stack axis → (84, 84, 3) RGB
        img_sample = obs["image"][0]
        print(f"     raw image obs shape: {img_sample.shape}")

        src_h, src_w = img_sample.shape[0], img_sample.shape[1]
        scale  = 5
        W, H   = src_w * scale, src_h * scale
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(path, fourcc, 10.0, (W, H))

        total_reward = 0.0
        MAX_STEPS    = 2_000

        for _ in range(MAX_STEPS):
            action, _ = self.model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)

            reward        = float(rewards[0])
            total_reward += reward

            # ── extract frame ─────────────────────────────────────────────
            img = obs["image"][0]          # (84, 84, 3, N_STACK) or (84, 84, 3)

            # Collapse any trailing stack/singleton dimensions → (84, 84, 3)
            while img.ndim > 3:
                img = img[..., -1]         # take the newest (last) stack frame

            frame = (img * 255).clip(0, 255).astype(np.uint8)   # (84, 84, 3) RGB
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_NEAREST)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # ── overlay stats ─────────────────────────────────────────────
            steer = float(action[0, 0])
            info  = infos[0]
            speed = float(info.get("velocity", obs["state"][0, 2]))

            cv2.putText(frame, f"Train step : {step:,}",           (10, 28),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(frame, f"Speed      : {speed:.2f} m/s",    (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   255,   0), 2)
            cv2.putText(frame, f"Steer      : {steer:+.3f}",       (10, 92),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,   200, 255), 2)
            cv2.putText(frame, f"Reward     : {reward:.3f}",       (10, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 200,   0), 2)
            cv2.putText(frame, f"Total      : {total_reward:.2f}", (10, 156), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 100, 100), 2)

            writer.write(frame)

            if dones[0]:
                break

        writer.release()
        env.reset()
        print(f"  ✅ Saved: {path}  (total reward: {total_reward:.2f})\n")