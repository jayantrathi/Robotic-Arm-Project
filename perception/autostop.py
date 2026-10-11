from __future__ import annotations

import os
import signal
import time

import numpy as np

from lerobot.rollout.robot_wrapper import ThreadSafeRobot


class AutoStopRobot(ThreadSafeRobot):
    def __init__(
        self,
        robot,
        *,
        autostop: bool = True,
        leave: float = 45.0,
        ret: float = 18.0,
        peak_min: float = 80.0,
        settle_s: float = 0.75,
        **_ignored,
    ):
        super().__init__(robot)
        self._as_on = autostop
        self._leave, self._ret = leave, ret
        self._peak_min, self._settle_s = peak_min, settle_s
        self._home: np.ndarray | None = None
        self._left = False
        self._peak = 0.0
        self._below_since: float | None = None
        self._fired = False

    @staticmethod
    def _arm_vec(obs) -> np.ndarray:
        keys = sorted(k for k in obs if k.endswith(".pos") and not k.startswith("gripper"))
        return np.array([float(obs[k]) for k in keys], dtype=float)

    def get_observation(self):
        obs = super().get_observation()
        if self._as_on and not self._fired:
            try:
                self._check_complete(obs)
            except Exception as e:
                print(f"[autostop] monitor error: {e}")
        return obs

    def _check_complete(self, obs) -> None:
        vec = self._arm_vec(obs)
        if vec.size == 0:
            return
        if self._home is None:
            self._home = vec
            print(f"[autostop] home pose set ({vec.size} joints); watching for return.")
            return

        d = float(np.linalg.norm(vec - self._home))
        self._peak = max(self._peak, d)
        if d > self._leave and not self._left:
            self._left = True
            print(f"[autostop] arm left home (d={d:.0f} deg); will stop when it returns.")

        settled = self._left and self._peak >= self._peak_min and d < self._ret
        if settled:
            now = time.time()
            if self._below_since is None:
                self._below_since = now
            elif now - self._below_since >= self._settle_s:
                self._fired = True
                print(f"[autostop] task complete: arm back home (d={d:.0f} deg). stopping.")
                self._on_complete()
        else:
            self._below_since = None

    def _on_complete(self) -> None:
        os.kill(os.getpid(), signal.SIGINT)


def make_autostop_wrapper(**cfg):
    def factory(robot):
        return AutoStopRobot(robot, **cfg)
    return factory
