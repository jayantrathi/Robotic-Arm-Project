#!/usr/bin/env python3
"""
Stop a rollout the moment the arm finishes one pick-and-place and returns home.

Why return-to-home
------------------
A fixed --duration is wrong: one grasp can take 12s or 22s, and whatever cap you
pick either cuts a slow grasp off or leaves time for the arm to wander into a
SECOND grab of the remaining object. The clean signal is the arm's own joints:
every episode starts at a home pose, swings far out to do the task (joint
distance 150-200 deg from home in the real data), then comes back to within
~10 deg of home. That return is unmistakable and -- crucially -- it happens
after BOTH successful and failed grasps, so unlike a gripper-release signal it
never mistakes a failed grab for success; it just ends the attempt and you
re-issue the command.

Validated offline on the recorded dataset: fires once, at the return, in ~12/13
episodes with LEAVE=45 / RETURN=18 / PEAK_MIN=80.

How it stops
-----------
On completion it sends itself SIGINT -- exactly what Ctrl-C does -- so LeRobot's
rollout catches it, runs its normal teardown, and returns the arm to a safe pose.
"""

from __future__ import annotations

import os
import signal
import time

import numpy as np

from lerobot.rollout.robot_wrapper import ThreadSafeRobot


class AutoStopRobot(ThreadSafeRobot):
    """ThreadSafeRobot that ends the episode when the arm returns home."""

    def __init__(
        self,
        robot,
        *,
        autostop: bool = True,
        leave: float = 45.0,     # deg from home that counts as "left home"
        ret: float = 18.0,       # deg from home that counts as "back home"
        peak_min: float = 80.0,  # must have swung out at least this far to count a real task
        settle_s: float = 0.75,  # must stay home this long (debounce) before stopping
        **_ignored,              # tolerate masking kwargs when subclassed
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

    # -- joint monitoring ----------------------------------------------------

    @staticmethod
    def _arm_vec(obs) -> np.ndarray:
        """The arm joints (everything '*.pos' except the gripper), order-stable."""
        keys = sorted(k for k in obs if k.endswith(".pos") and not k.startswith("gripper"))
        return np.array([float(obs[k]) for k in keys], dtype=float)

    def get_observation(self):
        obs = super().get_observation()
        if self._as_on and not self._fired:
            try:
                self._check_complete(obs)
            except Exception as e:      # never let monitoring crash the run
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
        """End the rollout cleanly (same path as Ctrl-C -> safe teardown)."""
        os.kill(os.getpid(), signal.SIGINT)


def make_autostop_wrapper(**cfg):
    """Factory for monkeypatching ThreadSafeRobot with auto-stop (no masking)."""
    def factory(robot):
        return AutoStopRobot(robot, **cfg)
    return factory
