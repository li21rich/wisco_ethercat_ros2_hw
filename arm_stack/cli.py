"""`arm <stage> [--sim]`: one command per calibration stage (spec: Calibration procedures).

  arm bringup | zero | limits | kt | gravity | friction | latency | pivot | pace
  arm mocap-calib | mocap-repeat | validate
  arm all --sim                 every stage in order on one sim arm (Phase 0 gate run)
  arm status                    latest calibration version and what it holds
  arm export pace SESSION.mcap OUT
  arm export lerobot OUT SESSION.mcap [...]

Each stage prints its report and, if it passed, writes a new calibration
version (config/calib/<arm>/<date>_v<N>.yaml; --sim writes to a separate sim
store so sim runs never mix with the arm's files). --force writes a failing
stage too, for debugging. Every run is recorded to one MCAP session in
--log-dir with the calibration version in its meta.

Hardware runs need arm01.launch.py up first; --sim needs nothing but
Pinocchio and numpy.
"""

from __future__ import annotations

import argparse
import datetime as dt
import getpass
import importlib
import os
import re
import sys
import tempfile
import time
from typing import Dict, List

import numpy as np
import yaml

from .config import CalibrationStore, calibration_hash, git_sha, load_arm
from .model.robot import ArmModel

STAGES = [
    ('bringup', 'bringup'), ('zero', 'zero'), ('limits', 'limits'), ('kt', 'torque_constant'),
    ('gravity', 'gravity'), ('friction', 'friction'), ('latency', 'latency'), ('pivot', 'pivot'),
    ('pace', 'pace'), ('mocap-calib', 'mocap_calib'), ('mocap-repeat', 'mocap_repeat'),
    ('validate', 'validate'),
]
MODULES = dict(STAGES)
SIM_ROOT = os.path.join(os.path.expanduser('~'), '.cache', 'arm_stack', 'sim_calib')


def build_model(arm, calibration) -> ArmModel:
    model = ArmModel(arm.urdf, arm.names)
    model.apply_calibration(calibration, arm.link_masses_kg)
    soft = calibration.get('soft_limits_rad') or {}
    for i, n in enumerate(arm.names):
        lo, hi = (soft.get(n) or [None, None])
        if lo is not None:
            model.lower[i] = lo
        if hi is not None:
            model.upper[i] = hi
    return model


def make_client(args, arm, calibration):
    if args.sim:
        from .sim.arm_sim import SimArm, SimErrors
        errors = SimErrors.random(arm.n, seed=args.seed, scale=args.error_scale)
        return SimArm(arm, calibration, errors, seed=args.seed)
    from .hw.ros_client import RosArmClient
    power_on = None
    if args.powered_at:
        # seconds ago the drives were powered, e.g. --powered-at 1200
        power_on = time.monotonic() - float(args.powered_at)
    return RosArmClient(arm, calibration, power_on_monotonic=power_on)


def session_meta(arm, store, calibration, client, operator):
    return {
        'arm': arm.arm, 'joint_names': arm.names,
        'calibration_version': calibration.get('version'),
        'calibration_hash': calibration_hash(store.latest_path()),
        'joint_config_hash': arm.hash(), 'git_sha': git_sha(),
        'time_since_power_on_s': client.time_since_power_on(), 'operator': operator,
        'started': dt.datetime.now().isoformat(timespec='seconds'),
    }


def update_joint_config(path: str, updates: Dict[str, Dict]) -> List[str]:
    """Targeted edit of `direction:` lines in the joint YAML, keeping comments."""
    done = []
    directions = updates.get('direction') or {}
    if not directions:
        return done
    text = open(path).read()
    for joint, value in directions.items():
        pat = re.compile(r'(-\s*name:\s*%s\b(?:(?!\n\s*-\s*name:).)*?\n\s*direction:\s*)(-?1)' % re.escape(joint),
                         re.S)
        text, k = pat.subn(lambda m: m.group(1) + str(int(value)), text, count=1)
        if k:
            done.append('%s direction -> %+d' % (joint, value))
    with open(path, 'w') as f:
        f.write(text)
    return done


def run_stage(name, args, arm, store, client, recorder=None, extra=None):
    from .calib.common import StageContext
    calibration = store.latest()
    if args.sim and hasattr(client, 'reload_calibration'):
        client.reload_calibration(calibration)
    ctx = StageContext(arm=arm, client=client, calibration=calibration, model=build_model(arm, calibration),
                       sim=args.sim, quick=args.quick, log=recorder,
                       rng=np.random.default_rng(args.seed))
    mod = importlib.import_module('arm_stack.calib.' + MODULES[name])
    if recorder is not None:
        recorder.event('stage_start', stage=name, calibration_version=calibration.get('version'))
    t0 = time.monotonic()
    result = mod.run(ctx, **(extra or {}))
    print(result.report())
    print('  (%.1f s)' % (time.monotonic() - t0))
    if recorder is not None:
        recorder.event('stage_end', stage=name, passed=result.passed)
    path = None
    if result.writes and (result.passed or args.force):
        path = store.write(name, result.writes)
        print('  wrote %s' % path)
    elif result.writes:
        print('  not written (failed; --force to write anyway)')
    if result.joint_config_updates:
        if args.write_config and not args.sim:
            for line in update_joint_config(arm.path, result.joint_config_updates):
                print('  joint config: ' + line)
        elif any(result.joint_config_updates.values()):
            print('  joint config changes (apply with --write-config): %s' % result.joint_config_updates)
    if name == 'validate':
        target = path or store.latest_path()
        if target:
            report = os.path.splitext(target)[0] + '_validate_report.yaml'
            with open(report, 'w') as f:
                yaml.safe_dump({'passed': result.passed, 'lines': result.lines, 'flags': result.flags,
                                'metrics': _plain(result.metrics)}, f, sort_keys=False)
            print('  report %s' % report)
    return result


def _plain(o):
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return _plain(o.tolist())
    if isinstance(o, np.generic):
        return o.item()
    return o


def cmd_status(args):
    arm = load_arm(args.arm)
    store = CalibrationStore(arm, args.calib_root or (SIM_ROOT if args.sim else None))
    path = store.latest_path()
    if not path:
        print('%s: no calibration yet in %s' % (arm.arm, store.dir))
        return 0
    cal = store.latest()
    print('%s: v%s (%s, stage %s) %s' % (arm.arm, cal.get('version'), cal.get('date'), cal.get('stage'), path))
    for key in ('zero_offsets_rad', 'soft_limits_rad', 'kt_output_nm_per_a', 'link_inertials', 'friction',
                'latency_ms', 'joint_stiffness_nm_per_rad', 'kinematic_overlay', 'pace', 'validation'):
        v = cal.get(key)
        filled = bool(v) and (not isinstance(v, dict) or any(x not in (None, [None, None], {}) for x in v.values()))
        print('  %-28s %s' % (key, 'set' if filled else '-'))
    return 0


def cmd_export(args):
    from .log import export
    if args.kind == 'pace':
        print(export.export_pace(args.paths[0], args.paths[1]))
    else:
        print(export.export_lerobot(args.paths[1:], args.paths[0], repo_id=args.repo_id, fps=args.fps))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog='arm', description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=[s for s, _ in STAGES] + ['all', 'status', 'export'])
    p.add_argument('paths', nargs='*', help='export: kind and files')
    p.add_argument('--sim', action='store_true', help='run on the sim backend')
    p.add_argument('--quick', action='store_true', help='fewer poses and cycles (smoke runs)')
    p.add_argument('--arm', default='config/arms/arm01.yaml', help='joint config')
    p.add_argument('--calib-root', default=None, help='calibration store root (default config/calib)')
    p.add_argument('--log-dir', default=None, help='MCAP session directory (default ~/arm_logs)')
    p.add_argument('--no-log', action='store_true')
    p.add_argument('--seed', type=int, default=0, help='sim error seed and pose sampling')
    p.add_argument('--error-scale', type=float, default=1.0, help='sim: scale of the injected errors')
    p.add_argument('--force', action='store_true', help='write results even if the stage failed')
    p.add_argument('--write-config', action='store_true', help='apply joint config changes (bringup directions)')
    p.add_argument('--powered-at', default=None, help='seconds since the drives were powered on')
    p.add_argument('--operator', default=None)
    p.add_argument('--replay', nargs='*', default=None, help='validate: recorded teleop sessions to replay')
    p.add_argument('--repo-id', default='WiscoHumanoidsLab/arm01-teleop')
    p.add_argument('--fps', type=int, default=50)
    p.add_argument('--fresh', action='store_true', help='sim: start from an empty sim calibration store')
    args = p.parse_args(argv)

    if args.stage == 'status':
        return cmd_status(args)
    if args.stage == 'export':
        if len(args.paths) < 3 or args.paths[0] not in ('pace', 'lerobot'):
            p.error('arm export pace SESSION.mcap OUT | arm export lerobot OUT SESSION.mcap [...]')
        args.kind = args.paths[0]
        args.paths = args.paths[1:]
        return cmd_export(args)

    arm = load_arm(args.arm)
    root = args.calib_root or (SIM_ROOT if args.sim else None)
    if args.sim and args.fresh:
        root = tempfile.mkdtemp(prefix='arm_sim_calib_')
    store = CalibrationStore(arm, root)
    calibration = store.latest()
    if calibration.get('version') in (None, 0) and not args.sim and args.stage not in ('bringup', 'zero', 'all'):
        print('warning: no calibration file yet for %s; run arm bringup and arm zero first' % arm.arm)
    client = make_client(args, arm, calibration)
    recorder = None
    if not args.no_log:
        from .log.recorder import Recorder, RecordingClient, SessionRejected
        log_dir = args.log_dir or os.path.join(os.path.expanduser('~'), 'arm_logs')
        name = '%s_%s_%s%s.mcap' % (arm.arm, dt.datetime.now().strftime('%Y%m%d_%H%M%S'), args.stage,
                                    '_sim' if args.sim else '')
        meta = session_meta(arm, store, calibration, client, args.operator or getpass.getuser())
        if meta['calibration_version'] is None:
            meta['calibration_version'] = 0
        try:
            recorder = Recorder(os.path.join(log_dir, name), meta)
            client = RecordingClient(client, recorder, every=10 if args.sim else 1)
        except (ImportError, SessionRejected) as e:
            print('warning: not recording (%s)' % e)
            recorder = None

    extra = {}
    if args.stage == 'validate' and args.replay:
        from .log.recorder import command_array, load_session
        trajs = [command_array(load_session(path), 'q')['q'] for path in args.replay]
        extra = {'replay_trajectories': trajs, 'replay_dt': client.cmd_dt}
    try:
        if args.stage == 'all':
            if not args.sim:
                p.error('arm all runs on the sim only; run hardware stages one at a time')
            ok = True
            for name, _ in STAGES:
                r = run_stage(name, args, arm, store, client, recorder)
                ok &= r.passed
            print('ALL STAGES %s' % ('PASS' if ok else 'had failures'))
            return 0 if ok else 1
        r = run_stage(args.stage, args, arm, store, client, recorder, extra)
        return 0 if r.passed else 1
    finally:
        if recorder is not None:
            print('  session %s' % recorder.path)
        client.close()


if __name__ == '__main__':
    sys.exit(main())
