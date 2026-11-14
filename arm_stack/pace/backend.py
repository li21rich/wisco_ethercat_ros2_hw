"""PACE replay backends: simulate logged commands with candidate actuator params.

The fit (spec stage 9) is over armature, friction (Coulomb + viscous),
encoder bias and delay per joint, with link inertials held at their stage 5
values. Drive CSP gains are fitted too when the joint config does not give
them (spec risk: CSP gains may not be readable over SDO).

Two backends:

  JointReplayBackend  built in. Each joint is replayed on its own (others held
                      during the chirp), with the link inertia about the joint
                      axis and gravity along the sweep taken from the
                      Pinocchio model. Vectorized over every candidate of every
                      joint, so one CMA-ES generation is one batched replay.
  IsaacLabBackend     replays the same arrays in Isaac Lab (full arm, one env
                      per candidate), isaac_backend.py. Needs isaaclab;
                      selected with backend='isaac'.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

PARAMS = ('armature', 'coulomb', 'viscous', 'encoder_bias', 'delay_s', 'drive_kp', 'drive_kd')


@dataclass
class JointLog:
    """One joint's replay data: command and measured position at fixed rate."""
    joint: int
    dt: float
    q_cmd: np.ndarray          # (T,)
    qd_cmd: np.ndarray
    q_meas: np.ndarray
    q_hold: np.ndarray         # full-arm pose the other joints held
    sweep: tuple = (0.0, 0.0)  # (center, amplitude) for the PACE recorder
    freq: tuple = (0.0, 0.0)


class JointReplayBackend:

    def __init__(self, model, logs: List[JointLog], gravity_ff: bool = True, substeps: int = 1):
        self.model = model
        self.logs = logs
        self.gravity_ff = gravity_ff
        self.substeps = substeps
        self.inertia = []
        self.g_grid = []
        for log in logs:
            j = log.joint
            M = model.mass_matrix(log.q_hold)
            self.inertia.append(float(M[j, j]))
            lo = min(log.q_meas.min(), log.q_cmd.min()) - 0.2
            hi = max(log.q_meas.max(), log.q_cmd.max()) + 0.2
            grid = np.linspace(lo, hi, 121)
            g = []
            for v in grid:
                q = log.q_hold.copy()
                q[j] = v
                g.append(model.gravity(q)[j])
            self.g_grid.append((grid, np.array(g)))
        self.friction_ff = [(model.coulomb[l.joint], model.viscous[l.joint]) for l in logs]

    def simulate(self, k: int, P: np.ndarray) -> np.ndarray:
        """Replay log k for a population P (pop x 7, PARAMS order); returns q (pop x T)."""
        log = self.logs[k]
        arm, fc, fv, bias, delay, kp, kd = [P[:, i] for i in range(7)]
        pop = len(P)
        T = len(log.q_cmd)
        dt = log.dt
        grid, g = self.g_grid[k]
        I = self.inertia[k] + arm
        fc0, fv0 = self.friction_ff[k]
        # fractional delay: interpolate the command history
        tgrid = np.arange(T) * dt
        q = np.full(pop, log.q_meas[0])          # true link angle
        qd = np.zeros(pop)
        out = np.empty((pop, T))
        h = dt / self.substeps
        for t in range(T):
            ts = tgrid[t] - delay
            qc = np.interp(ts, tgrid, log.q_cmd)
            qdc = np.interp(ts, tgrid, log.qd_cmd)
            g_true = np.interp(q, grid, g)
            # the host computed the feedforward from what it measured when it sent the
            # frame, and the frame arrives `delay` later
            q_sent = np.interp(ts, tgrid, log.q_meas)
            ff = np.interp(q_sent, grid, g) if self.gravity_ff else 0.0
            ff = ff + fc0 * np.tanh(qdc / 0.02) + fv0 * qdc
            for _ in range(self.substeps):
                tau = kp * (qc - (q + bias)) - kd * qd + ff
                fr = fc * np.tanh(qd / 0.01) + fv * qd
                qdd = (tau - g_true - fr) / I
                qd = qd + h * qdd
                q = q + h * qd
            out[:, t] = q + bias
        return out


class IsaacLabBackend:
    """The Isaac Lab replay (full arm, same parameters): arm_stack/pace/isaac_backend.py.

    Imported lazily so the stack runs without Isaac Lab; constructing it without
    an importable `isaaclab` raises an ImportError saying how to get it.
    """

    def __new__(cls, *args, **kwargs):
        from .isaac_backend import IsaacLabBackend as _Impl
        return _Impl(*args, **kwargs)


def default_bounds(spec, drive_known: Optional[Dict] = None):
    """Search box per joint, PARAMS order."""
    peak = spec.peak_torque_nm
    kp0 = (drive_known or {}).get('kp')
    kd0 = (drive_known or {}).get('kd')
    lo = [0.0, 0.0, 0.0, -0.02, 0.0, kp0 or 5.0, kd0 or 0.05]
    hi = [0.2, 0.08 * peak, 0.05 * peak, 0.02, 0.012, kp0 or 2000.0, kd0 or 40.0]
    return np.array(lo), np.array(hi)
