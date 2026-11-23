from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import pytest

from arm_stack.calib.stop_zero import verify_stop


@pytest.mark.parametrize('second_hit, expected', [
    (35.06, None),
    (36.30, 'did not repeat'),
    (-15.0, 'wrong side'),
])
def test_passive_stop_check_uses_independent_angles(monkeypatch, capsys, second_hit, expected):
    from arm_stack.config import load_arm
    joint = next(j for j in load_arm().joints if j.name == 'wrist_pitch')
    first_hit = -15.0 if expected == 'wrong side' else 35.05
    poses = iter([0.0, 10.0, first_hit, 0.0, second_hit])
    answers = iter(['upper', '0', '10', '', '', ''])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    reader = SimpleNamespace(disable_for_verification=lambda arm: None,
                             capture=lambda arm, j: (next(poses) + 13) * joint.counts_per_rev / 360)
    reader.positions = lambda arm: nullcontext(reader)
    args = SimpleNamespace(arm='arm01', sim=False, verify_stop='wrist_pitch')
    if expected:
        with pytest.raises(ValueError, match=expected):
            verify_stop(args, reader)
    else:
        assert verify_stop(args, reader) == 0
        assert 'conditional CAD angle 35.04 deg' in capsys.readouterr().out


@pytest.mark.parametrize('side, endpoint', [('lower', -90), ('upper', 180)])
def test_j1_endpoints_relative_to_start(monkeypatch, capsys, side, endpoint):
    poses = iter([0, endpoint, 0, endpoint])
    answers = iter([side, '', '', '', ''])
    monkeypatch.setattr('builtins.input', lambda prompt: next(answers))
    reader = SimpleNamespace(disable_for_verification=lambda arm: None,
                             capture=lambda arm, j: 48000 + next(poses) * j.counts_per_rev / 360)
    reader.positions = lambda arm: nullcontext(reader)
    args = SimpleNamespace(arm='arm01', sim=False, verify_stop='shoulder_pitch')
    assert verify_stop(args, reader) == 0
    report = capsys.readouterr().out
    assert f'configured endpoint {endpoint:.2f} deg' in report
    assert f'hand-seated readings {endpoint:.2f}, {endpoint:.2f}' in report


def test_missing_torque_is_not_a_stop(monkeypatch):
    from arm_stack.calib import limits
    state = SimpleNamespace(q=np.zeros(1), qd=np.zeros(1), tau=np.zeros(1), t=0.0)
    client = SimpleNamespace(state=lambda: state, cmd_dt=0.01)
    ctx = SimpleNamespace(client=client, quick=False,
                          arm=SimpleNamespace(names=['axis'], joints=[SimpleNamespace(peak_torque_nm=5)]),
                          model=SimpleNamespace(n=1, lower=np.array([-1.0]), upper=np.array([1.0]),
                                                gravity=lambda q: np.zeros(1)))
    monkeypatch.setattr(limits, 'move_to', lambda *args, **kwargs: state)
    def stalled(*args, **kwargs):
        state.t += client.cmd_dt
        return state
    monkeypatch.setattr(limits, 'stream', stalled)
    assert limits.find_stop(ctx, 0, 1, np.zeros(1), np.ones(1)) is None


def test_stiff_stop_is_found_before_the_lag_bound(monkeypatch):
    """A rigid stop: the joint halts at once and pushes against it. The command runs
    ahead at 0.3 rad/s, so the lag passes the 3 deg abort bound in about 0.17 s; with a
    slow torque rise the 40 ms confirm has not finished by then. That is still a stop
    (the 2026-11-15 gate failure: J1 halted at -92.5 deg, lag 3.01 deg, -10.3 Nm)."""
    from arm_stack.calib import limits
    wall = 0.5
    state = SimpleNamespace(q=np.array([0.4]), qd=np.zeros(1), tau=np.zeros(1), t=0.0)
    client = SimpleNamespace(state=lambda: state, cmd_dt=0.01)
    ctx = SimpleNamespace(client=client, quick=True,
                          arm=SimpleNamespace(names=['axis'], joints=[SimpleNamespace(peak_torque_nm=60)]),
                          model=SimpleNamespace(n=1, lower=np.array([-1.0]), upper=np.array([0.45]),
                                                gravity=lambda q: np.zeros(1)))
    monkeypatch.setattr(limits, 'move_to', lambda *args, **kwargs: state)
    def against_the_wall(client, q_cmd, qd, **kwargs):
        state.t += client.cmd_dt
        state.q = np.minimum(q_cmd[0], wall)
        lag = q_cmd[0][0] - state.q[0]
        # torque builds slowly with the lag: under 2.4 Nm (4% of peak) until 2.6 deg
        state.tau = np.array([max(0.0, lag - 2.6 * limits.DEG) * 400.0])
        return state
    monkeypatch.setattr(limits, 'stream', against_the_wall)
    assert limits.find_stop(ctx, 0, 1, np.zeros(1), np.ones(1)) == pytest.approx(wall)


def test_verify_disables_only_enabled_drives_and_checks_status(monkeypatch):
    from arm_stack.calib.commission import EncoderReader
    from arm_stack.config import load_arm
    reader = EncoderReader()
    arm = load_arm()
    enabled = {1, 2}
    writes = []
    def command(*args):
        if args == ('master',):
            return 'Active: no\nSlaves: 7'
        _, _, pos, _, kind, index, sub, value = args
        assert (kind, index, sub, value) == ('uint16', '0x6040', '0', '0')
        writes.append(int(pos))
        enabled.remove(int(pos))
    monkeypatch.setattr(reader, 'command', command)
    monkeypatch.setattr(reader, 'upload', lambda pos, kind, index: 4 if pos in enabled else 0)
    reader.disable_for_verification(arm)
    assert writes == [1, 2]


# --------------------------------------------------------------------------- hardware stop-zero

def test_measured_baseline_ignores_a_large_steady_gravity_load(monkeypatch):
    """Uncalibrated coordinates: the joint carries 4 Nm of gravity the model does not
    know about. Contact is torque above what the joint drew while moving freely."""
    from arm_stack.calib import limits
    wall = 0.30
    state = SimpleNamespace(q=np.array([0.0]), qd=np.zeros(1), tau=np.array([4.0]), t=0.0)
    client = SimpleNamespace(state=lambda: state, cmd_dt=0.01)
    ctx = SimpleNamespace(client=client, quick=False,
                          arm=SimpleNamespace(names=['axis'], joints=[SimpleNamespace(peak_torque_nm=34)]),
                          model=SimpleNamespace(n=1, lower=np.array([-1.0]), upper=np.array([1.0]),
                                                gravity=lambda q: np.zeros(1)))
    monkeypatch.setattr(limits, 'move_to', lambda *args, **kwargs: state)

    def joint(client, q_cmd, qd, **kwargs):
        state.t += client.cmd_dt
        target = q_cmd[0][0]
        state.qd = np.array([0.0 if target >= wall else qd[0][0]])
        state.q = np.array([min(target, wall)])
        state.tau = np.array([4.0 + max(0.0, target - wall) * 200.0])
        return state
    monkeypatch.setattr(limits, 'stream', joint)
    # with the model baseline the 4 Nm load alone reads as a press long before the wall
    hit = limits.find_stop(ctx, 0, 1, np.zeros(1), np.full(1, 8.0), reference=0.25,
                           approach=0.25, beyond=0.2, speed=0.1, baseline='measured')
    assert hit == pytest.approx(wall)


def test_zero_from_hits_corrects_the_rough_reference():
    from arm_stack.calib.stop_zero import zero_from_hits
    deg = np.pi / 180
    # the rough coordinates read the stop (really at -35 deg) as -31 deg
    assert zero_from_hits(1.0, [-31 * deg, -31.1 * deg], -35 * deg, 10 * deg, 'j') == \
        pytest.approx(1.0 + 3.95 * deg)
    with pytest.raises(RuntimeError, match='did not repeat'):
        zero_from_hits(0.0, [-31 * deg, -32 * deg], -35 * deg, 10 * deg, 'j')
    # far from CAD: the measured stop wins (put at the CAD angle), with a warning
    assert zero_from_hits(0.0, [-20 * deg, -20 * deg], -35 * deg, 10 * deg, 'j') == \
        pytest.approx(15 * deg)


def test_plan_skips_joints_without_hard_stops_and_picks_the_nearer_stop():
    from arm_stack.calib.stop_zero import plan_stops, stop_soft_limits
    deg = np.pi / 180
    arm = SimpleNamespace(names=['a', 'b', 'c'], joints=[SimpleNamespace(hard_stops=False),
                                                         SimpleNamespace(hard_stops=True),
                                                         SimpleNamespace(hard_stops=True)])
    stops = np.array([[-90, 180], [-0.1, 195], [-35, 35.2]]) * deg
    plan = plan_stops(arm, np.array([90, 3, 10]) * deg, stops)
    assert plan[0] is None
    assert plan[1]['side'] == 'lower' and plan[2]['side'] == 'upper'
    limits = stop_soft_limits(arm, np.array([-90, 0, -35]) * deg, np.array([180, 190, 35]) * deg, plan)
    assert np.allclose(np.array(limits['a']) / deg, [-90, 180])
    assert np.allclose(np.array(limits['b']) / deg, [2.9, 190])
    assert np.allclose(np.array(limits['c']) / deg, [-35, 32.2])


class StopArm:
    """Seven joints held in Mode P that stop at walls; the reported angles carry the
    rough-reference error `bias` (reported = true + bias)."""

    def __init__(self, true_q, lower, upper, bias, gravity):
        self.true_q = np.array(true_q, float)
        self.lower, self.upper, self.bias = lower, upper, np.asarray(bias, float)
        self.g = np.asarray(gravity, float)
        self.target = self.true_q.copy()
        self.cmd_dt = 0.01
        self.t = 0.0
        self.commands = []

    def state(self):
        q = self.true_q + self.bias
        press = np.maximum(self.target - self.bias - self.upper, 0) - \
            np.maximum(self.lower - (self.target - self.bias), 0)
        tau = self.g + 150.0 * press
        clamp = getattr(self, 'clamp', None)
        if clamp is not None:              # the drive's 0x6072 cap
            tau = np.clip(tau, -clamp, clamp)
        moving = np.abs(self.target - self.bias - self.true_q) > 1e-9
        return SimpleNamespace(q=q, qd=np.where(moving, 0.0, 0.0), tau=tau, t=self.t,
                               status=np.full(7, 0x27), raw_counts=np.zeros(7))

    def send(self, command):
        self.commands.append(command)
        self.target = np.asarray(command.q, float).copy()
        if command.torque_clamp is not None:
            assert np.all(np.asarray(command.torque_clamp) > 0)
        self.clamp = None if command.torque_clamp is None else np.asarray(command.torque_clamp, float)

    def step(self, seconds):
        self.t += seconds
        self.true_q = np.clip(self.target - self.bias, self.lower, self.upper)
        return self.state()

    def close(self):
        pass


@pytest.mark.parametrize('stop_pose', [None, 'hanging'])
def test_hardware_stop_zero_recovers_zeros_from_a_rough_pose(monkeypatch, tmp_path, stop_pose):
    from arm_stack import cli
    from arm_stack.calib import stop_zero
    from arm_stack.config import CalibrationStore, load_arm
    arm = load_arm()
    deg = np.pi / 180
    stops = stop_zero.stop_references(arm)
    # bigger than the 2 deg the simulator uses: the operator's pose is up to 6 deg off
    bias = np.array([4.0, -5.0, 6.0, -3.0, 5.5, -2.0, 3.5]) * deg
    model = SimpleNamespace(n=7, home=arm.home.copy(),
                            lower=np.array([-90, 0, -90, 0, -90, -35, -90]) * deg,
                            upper=np.array([180, 190, 90, 130, 90, 35, 90]) * deg,
                            gravity=lambda q: np.zeros(7))
    model.copy = lambda: SimpleNamespace(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                                            for k, v in vars(model).items()})
    monkeypatch.setattr(cli, 'build_model', lambda arm, cal: model)
    gravity = np.array([0.5, 2.0, 0.8, 1.5, 0.3, 0.2, 0.1])
    # the operator thinks the arm is at home; it really is at home - bias
    start = stop_zero.rough_pose(arm, stop_pose or 'home')
    hw = StopArm(start - bias, stops[:, 0], stops[:, 1], bias, gravity)
    # J1 has no hard stop: the sim arm lets it move freely
    hw.lower[0], hw.upper[0] = -10, 10
    monkeypatch.setattr('builtins.input', lambda prompt: pytest.fail('stop-zero must not prompt'))
    counts_per_rad = np.array([j.counts_per_rev for j in arm.joints]) / (2 * np.pi)
    direction = np.array([j.direction for j in arm.joints])
    # encoders whose true zero is 0: they count the true angle
    counts = direction * (start - bias) * counts_per_rad
    reader = SimpleNamespace(disable_for_verification=lambda arm: None,
                             capture_all=lambda arm: counts)
    reader.positions = lambda arm: nullcontext(reader)
    launch = SimpleNamespace(connect=lambda arm, seed: hw, close=lambda: None,
                             process=SimpleNamespace(wait=lambda: 0, poll=lambda: None))
    args = SimpleNamespace(arm='arm01', sim=False, force=False, travel_deg=None,
                           stop_tolerance_deg=10.0, stop_pose=stop_pose, calib_root=str(tmp_path))
    def factory(arm_, calibration_file, log_path):
        _assert_launch_accepts(arm_, calibration_file)
        return launch
    assert stop_zero.run_hardware(args, reader=reader, launch_factory=factory) == 0
    saved = CalibrationStore(arm, str(tmp_path)).latest()
    assert saved['stage'] == 'stop_zero' and saved['coordinate_frame'] == 'urdf'
    references = saved['commissioning']['references']
    for i, name in enumerate(arm.names):
        # no CAD angles: every zero stays on the rough reference (off by its bias)
        assert saved['zero_offsets_rad'][name] == pytest.approx(-bias[i], abs=1e-9), name
        if name == 'shoulder_pitch':
            assert name not in references
            continue
        # both measured hard stops, read in the rough coordinates (true wall + bias)
        ref = references[name]
        assert ref['lower']['stop_rad'] == pytest.approx(stops[i, 0] + bias[i], abs=0.3 * deg), name
        assert ref['upper']['stop_rad'] == pytest.approx(stops[i, 1] + bias[i], abs=0.3 * deg), name
        assert ref['travel_rad'] == pytest.approx(stops[i, 1] - stops[i, 0], abs=0.5 * deg), name
    assert saved['commissioning']['range_kind'] == 'both_stops_per_joint'
    assert saved['commissioning']['rough_reference']['joints_left_on_it'] == ['shoulder_pitch']
    # every stop search ran past the soft limits with a torque clamp and no gravity feedforward
    searches = [c for c in hw.commands if c.ignore_soft_limits]
    assert searches and all(c.torque_clamp is not None and not c.gravity_ff for c in searches)


@pytest.mark.parametrize('wall_deg, found', [(-120.0, True), (-1000.0, False)])
def test_hardware_stop_zero_searches_until_contact(monkeypatch, tmp_path, wall_deg, found):
    """J2's real stop 120 deg from the search pose (nowhere near CAD) is found and its
    limit saved; no wall within the 300 deg runaway cap stops the run, nothing saved."""
    from arm_stack import cli
    from arm_stack.calib import stop_zero
    from arm_stack.config import CalibrationStore, load_arm
    arm = load_arm()
    deg = np.pi / 180
    stops = stop_zero.stop_references(arm)
    model = SimpleNamespace(n=7, home=arm.home.copy(),
                            lower=np.array([-90, 0, -90, 0, -90, -35, -90]) * deg,
                            upper=np.array([180, 190, 90, 130, 90, 35, 90]) * deg,
                            gravity=lambda q: np.zeros(7))
    model.copy = lambda: SimpleNamespace(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                                            for k, v in vars(model).items()})
    monkeypatch.setattr(cli, 'build_model', lambda arm, cal: model)
    start = stop_zero.rough_pose(arm, 'hanging')
    lower, upper = stops[:, 0].copy(), stops[:, 1].copy()
    lower[0], upper[0] = -10, 10
    lower[1] = wall_deg * deg
    hw = StopArm(start, lower, upper, np.zeros(7), np.zeros(7))
    monkeypatch.setattr('builtins.input', lambda prompt: pytest.fail('stop-zero must not prompt'))
    counts_per_rad = np.array([j.counts_per_rev for j in arm.joints]) / (2 * np.pi)
    direction = np.array([j.direction for j in arm.joints])
    counts = direction * start * counts_per_rad
    reader = SimpleNamespace(disable_for_verification=lambda arm: None, capture_all=lambda arm: counts)
    reader.positions = lambda arm: nullcontext(reader)
    launch = SimpleNamespace(connect=lambda arm, seed: hw, close=lambda: None,
                             process=SimpleNamespace(wait=lambda: 0, poll=lambda: None))
    args = SimpleNamespace(arm='arm01', sim=False, force=False, travel_deg=None,
                           stop_tolerance_deg=10.0, stop_pose='hanging', calib_root=str(tmp_path))
    if not found:
        with pytest.raises(RuntimeError, match='shoulder_roll: no lower stop accepted'):
            stop_zero.run_hardware(args, reader=reader, launch_factory=lambda *a: launch)
        assert CalibrationStore(arm, str(tmp_path)).latest_path() is None
        # a failed run brings the arm back to rest before the drives shut off
        assert np.asarray(hw.commands[-1].q) == pytest.approx(start, abs=1e-9)
        return
    assert stop_zero.run_hardware(args, reader=reader, launch_factory=lambda *a: launch) == 0
    assert np.asarray(hw.commands[-1].q) == pytest.approx(start, abs=1e-9)   # PASS ends at rest
    saved = CalibrationStore(arm, str(tmp_path)).latest()
    assert saved['zero_offsets_rad']['shoulder_roll'] == pytest.approx(0.0, abs=1e-9)
    j2 = saved['commissioning']['references']['shoulder_roll']
    assert j2['lower']['stop_rad'] == pytest.approx(wall_deg * deg, abs=0.3 * deg)
    assert j2['upper']['stop_rad'] == pytest.approx(stops[1, 1], abs=0.3 * deg)


def test_hanging_rough_pose_straightens_only_the_elbow():
    from arm_stack.calib.stop_zero import rough_pose
    from arm_stack.config import load_arm
    arm = load_arm()
    hanging = rough_pose(arm, 'hanging')
    assert hanging[arm.names.index('elbow')] == 0.0
    assert hanging[arm.names.index('shoulder_pitch')] == pytest.approx(np.pi / 2, abs=1e-4)
    others = [i for i, n in enumerate(arm.names) if n != 'elbow']
    assert np.allclose(hanging[others], np.asarray(arm.home)[others])
    with pytest.raises(ValueError):
        rough_pose(arm, 'sideways')


def _assert_launch_accepts(arm, calibration_file):
    """The motion gate in test_bench.launch.py, applied to the temporary calibration."""
    import hashlib
    import math
    import yaml
    cal = yaml.safe_load(open(calibration_file))
    assert cal.get('version')
    limits = cal.get('soft_limits_rad') or {}
    for n in arm.names:
        assert math.isfinite(float(cal['zero_offsets_rad'][n])), n
        lo, hi = limits[n]
        assert lo is not None and hi is not None and math.isfinite(lo) and math.isfinite(hi) and lo < hi, n
    assert cal['joint_config_hash'] == hashlib.sha256(open(arm.path, 'rb').read()).hexdigest()[:16]


def test_hardware_stop_zero_returns_to_rest_on_ctrl_c(monkeypatch, tmp_path):
    from arm_stack import cli
    from arm_stack.calib import stop_zero
    from arm_stack.config import CalibrationStore, load_arm
    arm = load_arm()
    deg = np.pi / 180
    stops = stop_zero.stop_references(arm)
    model = SimpleNamespace(n=7, home=arm.home.copy(),
                            lower=np.array([-90, 0, -90, 0, -90, -35, -90]) * deg,
                            upper=np.array([180, 190, 90, 130, 90, 35, 90]) * deg,
                            gravity=lambda q: np.zeros(7))
    model.copy = lambda: SimpleNamespace(**{k: (v.copy() if isinstance(v, np.ndarray) else v)
                                            for k, v in vars(model).items()})
    monkeypatch.setattr(cli, 'build_model', lambda arm, cal: model)
    start = stop_zero.rough_pose(arm, 'hanging')
    hw = StopArm(start + 5 * deg, stops[:, 0], stops[:, 1], np.zeros(7), np.zeros(7))

    def interrupted(*a, **k):
        raise KeyboardInterrupt     # Ctrl-C during the searches
    monkeypatch.setattr(stop_zero, 'search_all', interrupted)
    counts_per_rad = np.array([j.counts_per_rev for j in arm.joints]) / (2 * np.pi)
    counts = np.array([j.direction for j in arm.joints]) * start * counts_per_rad
    reader = SimpleNamespace(disable_for_verification=lambda arm: None, capture_all=lambda arm: counts)
    reader.positions = lambda arm: nullcontext(reader)
    launch = SimpleNamespace(connect=lambda arm, seed: hw, close=lambda: None,
                             process=SimpleNamespace(wait=lambda: 0, poll=lambda: None))
    args = SimpleNamespace(arm='arm01', sim=False, force=False, travel_deg=None,
                           stop_tolerance_deg=10.0, stop_pose='hanging', calib_root=str(tmp_path))
    with pytest.raises(KeyboardInterrupt):
        stop_zero.run_hardware(args, reader=reader, launch_factory=lambda *a: launch)
    assert CalibrationStore(arm, str(tmp_path)).latest_path() is None
    assert np.asarray(hw.commands[-1].q) == pytest.approx(start, abs=1e-9)


class FlatTorqueStopArm(StopArm):
    """A stop that shows no torque rise and whose position jitters (a fast, springy stop):
    only the lag says it is there."""

    def state(self):
        s = super().state()
        s.tau = self.g.copy()
        # the stop gives: within ~1 deg of a held command (the 2026-11-22 J2 abort at lag 0.88)
        self._jitter = -getattr(self, '_jitter', 0.6 * np.pi / 180)
        at_wall = (self.true_q <= self.lower + 1e-9) | (self.true_q >= self.upper - 1e-9)
        s.q = s.q + np.where(at_wall, self._jitter, 0.0)
        return s


def test_fast_search_finds_a_stop_without_torque_rise():
    from arm_stack.calib.limits import find_stop
    from arm_stack.calib.common import StageContext
    from arm_stack.config import load_arm
    arm = load_arm()
    deg = np.pi / 180
    lower = np.full(7, -400 * deg)
    upper = np.full(7, 400 * deg)
    lower[1] = -150 * deg
    hw = FlatTorqueStopArm(np.zeros(7), lower, upper, np.zeros(7), np.zeros(7))
    hw.cmd_dt = 0.002
    model = SimpleNamespace(n=7, lower=np.full(7, -np.pi), upper=np.full(7, np.pi),
                            gravity=lambda q: np.zeros(7))
    ctx = StageContext(arm=arm, client=hw, calibration={}, model=model, sim=False, quick=False,
                       motion_speed=1.0)
    hit = find_stop(ctx, 1, -1, np.zeros(7), np.full(7, 5.0), reference=0.0, approach=0.0,
                    beyond=300 * deg, breakaway=3 * deg, speed=0.7, approach_speed=1.0,
                    baseline='measured', clip_near=False, gravity_ff=False)
    assert hit == pytest.approx(-150 * deg, abs=1 * deg)


class StickSlipArm(StopArm):
    """Sticks for 0.2 s every 20 deg (torque rising while it sticks), then slips: the
    2026-11-22 elbow trace, where a stick at -251.9 deg passed as a stop."""

    def step(self, seconds):
        self.t += seconds
        free = np.clip(self.target - self.bias, self.lower, self.upper)
        deg = np.pi / 180
        stuck_until = getattr(self, 'stuck_until', -1.0)
        if self.t < stuck_until:
            return self.state()
        crossed = np.floor(np.abs(free[1]) / (20 * deg)) > np.floor(np.abs(self.true_q[1]) / (20 * deg))
        self.true_q = free
        if crossed and free[1] > self.lower[1] + deg:
            self.stuck_until = self.t + 0.2
        return self.state()

    def state(self):
        s = super().state()
        if self.t < getattr(self, 'stuck_until', -1.0):
            lag = abs(self.target[1] - self.true_q[1])
            s.tau = s.tau.copy()
            s.tau[1] = -min(150.0 * lag, 5.0)       # pushing harder while it sticks
        return s


def test_stick_slip_is_not_a_stop():
    from arm_stack.calib.limits import find_stop
    from arm_stack.calib.common import StageContext
    from arm_stack.config import load_arm
    arm = load_arm()
    deg = np.pi / 180
    lower = np.full(7, -400 * deg)
    upper = np.full(7, 400 * deg)
    lower[1] = -150 * deg
    hw = StickSlipArm(np.zeros(7), lower, upper, np.zeros(7), np.zeros(7))
    hw.cmd_dt = 0.002
    model = SimpleNamespace(n=7, lower=np.full(7, -np.pi), upper=np.full(7, np.pi),
                            gravity=lambda q: np.zeros(7))
    ctx = StageContext(arm=arm, client=hw, calibration={}, model=model, sim=False, quick=False,
                       motion_speed=1.0)
    hit = find_stop(ctx, 1, -1, np.zeros(7), np.full(7, 8.0), reference=0.0, approach=0.0,
                    beyond=300 * deg, breakaway=3 * deg, speed=0.7, approach_speed=1.0,
                    baseline='measured', clip_near=False, gravity_ff=False, confirm_s=0.3)
    assert hit == pytest.approx(-150 * deg, abs=1 * deg)
