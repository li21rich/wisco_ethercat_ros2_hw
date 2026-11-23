"""Exercise mechanical-stop encoder registration on the simulated EtherCAT arm."""

import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from .common import DEG, StageContext, safe_home
from .limits import find_stop, path_gravity
from ..config import CalibrationStore, empty_calibration, load_arm, resolve
from ..control.motion import move_to
from ..units import Units


def stop_references(arm):
    if arm.arm != 'arm01':
        raise ValueError('The CAD contact capture is for arm01.')
    capture = json.loads(Path(resolve('tools/cad/arm01_capture.json')).read_text())
    stops = np.deg2rad([capture['contact_onset_deg'][n] for n in arm.names])
    limit = ET.parse(arm.urdf).find("./joint[@name='shoulder_pitch']/limit")
    stops[arm.names.index('shoulder_pitch')] = [float(limit.attrib[k]) for k in ('lower', 'upper')]
    return stops


def check_encoder(args, reader=None):
    """Compare all drive position readings across a small hand movement."""
    from .commission import EncoderReader

    arm = load_arm(args.arm)
    if args.sim or args.check_encoder not in arm.names:
        raise ValueError('Choose a physical joint from ' + ', '.join(arm.names))
    reader = reader or EncoderReader()
    reader.disable_for_verification(arm)
    with reader.positions(arm) as positions:
        input('With the supported arm stationary, press Enter for the baseline: ')
        before = positions.capture_all(arm)
        input(f'Move only {args.check_encoder} a few degrees by hand, hold it there, then press Enter: ')
        after = positions.capture_all(arm)
    for joint, first, last in zip(arm.joints, before, after):
        print(f'bus {joint.bus_position} {joint.name}: 0x6064 {first:.1f} -> {last:.1f}, '
              f'delta {last-first:+.1f} counts')
    return 0


def verify_stop(args, reader=None):
    """Measure a hand-seated stop independently of the saved encoder zero."""
    from .commission import EncoderReader, angle, fit_encoder

    if args.sim:
        raise ValueError('Stop verification measures the physical arm. For simulation use --stop-zero.')
    arm = load_arm(args.arm)
    stops = stop_references(arm)
    if args.verify_stop not in arm.names:
        raise ValueError('Choose a joint from ' + ', '.join(arm.names))
    i = arm.names.index(args.verify_stop)
    joint = arm.joints[i]
    reader = reader or EncoderReader()
    reader.disable_for_verification(arm)
    with reader.positions(arm) as positions:
        start_relative = joint.name == 'shoulder_pitch'
        print(f'{joint.name}: support the arm with drives disabled and hold the other joints '
              'in the intended homing posture throughout the check.')
        side = input('Which physical stop are you verifying (lower/upper)? ').strip().lower()
        if side not in ('lower', 'upper'):
            raise ValueError('Choose lower or upper.')
        if start_relative:
            input('Set J1 at the starting position and press Enter to record it as 0 deg: ')
            ca = positions.capture(arm, joint)
            direction, cpr = joint.direction, joint.counts_per_rev
            zero = direction * ca * 2 * np.pi / cpr
            print(f'J1 start recorded. Configured range {stops[i, 0]/DEG:+.0f} to {stops[i, 1]/DEG:+.0f} deg.')
            a = b = 0.0
            return_pose = 'the starting position (0 deg)'
        else:
            print('Use an independent angle reference aligned to the URDF axis.')
            a = angle('Place at a known interior angle A. Enter measured URDF degrees: ')
            ca = positions.capture(arm, joint)
            b = angle('Move to a second interior angle B at least 5 degrees away. Enter measured degrees: ')
            cb = positions.capture(arm, joint)
            direction, cpr, zero = fit_encoder(ca, a, cb, b)
            if direction != joint.direction:
                raise ValueError('Measured encoder direction disagrees with the joint configuration.')
            return_pose = 'interior angle A'
        hits = []
        for attempt in range(2):
            input(f'Approach {side} stop by hand ({attempt+1}/2). Confirm the intended stop faces '
                  'touch, with cables and stand clear. Press Enter when stationary: ')
            counts = positions.capture(arm, joint)
            if start_relative and abs(counts - ca) <= 2:
                raise RuntimeError(f'{joint.name} bus {joint.bus_position} encoder shows no measurable movement '
                                   f'(start {ca:.1f}, stop {counts:.1f} counts). '
                                   f'Run arm autocal --check-encoder {joint.name} to compare all drives.')
            hits.append(direction * counts * 2 * np.pi / cpr - zero)
            if attempt == 0:
                input(f'Move back to {return_pose} and press Enter: ')
                repeat = positions.capture(arm, joint)
                if abs(repeat - ca) * 360 / cpr > 0.2:
                    raise ValueError('Reference position did not repeat within 0.2 deg.')
    reference = stops[i, int(side == 'upper')]
    label = 'configured endpoint' if start_relative else 'conditional CAD angle'
    print(f'{joint.name} {side}: hand-seated readings {hits[0]/DEG:.2f}, {hits[1]/DEG:.2f} deg; '
          f'{label} {reference/DEG:.2f} deg; '
          f'mean difference {(np.mean(hits)-reference)/DEG:+.2f} deg.')
    if abs(hits[1] - hits[0]) > 0.2 * DEG:
        raise ValueError('Contact measurements did not repeat within 0.2 deg.')
    if (side == 'lower' and max(hits) >= min(a, b)) or (side == 'upper' and min(hits) <= max(a, b)):
        raise ValueError('Selected stop is on the wrong side of the measured interior angles.')
    print('Report only. No calibration or motor settings written.')
    return 0


# --------------------------------------------------------------------------- hardware
# Powered stop-zero on the real arm. The operator places the arm roughly at
# home_pose_rad; those rough coordinates only plan the searches. Each joint with
# hard stops then creeps into its nearer CAD contact stop twice, and its encoder
# zero is set from where it actually touched. Joints with `hard_stops: false`
# (J1) keep the rough home reference and are reported as such.
DEFAULT_TOLERANCE_DEG = 10.0     # how far off the rough home pose may be, per joint
HW_SEARCH_SPEED = 0.70           # rad/s into the stop (about 40 deg/s), both directions
HW_CONTACT_CONFIRM_S = 0.30     # a contact must stay put this long (the elbow sticks and slips)
HW_MOVE_SPEED = 1.00             # rad/s for the moves between searches
HW_SETTLE_SPEED = 0.25           # rad/s for the first move to the search pose
STOP_REPEAT = 0.25 * DEG
SOFT_INSET = 3 * DEG
# Hardware searches creep from the search pose until the joint stalls on its hard stop;
# no CAD stop angle is used. This is only the runaway cap on the travel.
SEARCH_TRAVEL = 300 * DEG
# each search starts from rest: contacts count once the joint has moved this far
BREAKAWAY = 3 * DEG
# Push margin over modeled gravity, as a fraction of peak torque. The sim's 8% is
# 0.8 Nm on an X4-10, under the 0.87 Nm stiction measured on the bench; 12% still
# stays well below the bring-up clamps (capped at 90% of them below).
HW_SEARCH_CLAMP_FRACTION = 0.9  # stop searches push with up to this fraction of max_torque_nm
HW_PUSH_MARGIN = 0.12


def plan_stops(arm, pose, stops):
    """Per joint: None (no hard stop) or the nearer stop to `pose`."""
    plan = []
    for i, joint in enumerate(arm.joints):
        if not joint.hard_stops:
            plan.append(None)
            continue
        lower, upper = stops[i]
        direction = -1 if pose[i] - lower <= upper - pose[i] else 1
        plan.append({'direction': direction, 'side': 'lower' if direction < 0 else 'upper',
                     'reference': float(lower if direction < 0 else upper)})
    return plan


def search_clamp(model, pose, j, direction, reference, tolerance, peak, max_torque, names,
                 near=None, end=None):
    """0x6072 clamp for the search on joint j: modeled gravity (with 30% room for the
    rough coordinates) plus HW_PUSH_MARGIN of peak, never above 90% of the bring-up clamp.
    near / end: where the search starts and its runaway cap (default: around the CAD stop)."""
    if near is None:
        near = reference - direction * (tolerance + 5 * DEG)
    if end is None:
        end = reference + direction * (tolerance + 3 * DEG)
    g = np.maximum(path_gravity(model, pose, j, near), path_gravity(model, pose, j, end))
    clamp = np.minimum(1.3 * g + HW_PUSH_MARGIN * peak, 0.9 * max_torque)
    if clamp[j] - g[j] < 0.06 * peak[j] or np.any(clamp < 1.1 * g):
        worst = int(np.argmax(g / max_torque))
        raise RuntimeError('%s: modeled gravity on its stop search (%.2f Nm on %s) leaves too little '
                           'torque under the bring-up clamp. Nothing was moved toward the stop.'
                           % (names[j], g[worst], names[worst]))
    return clamp


def zero_from_hits(seed_zero, hits, reference, tolerance, name):
    """New encoder zero from two contact angles measured in the rough coordinates."""
    if abs(hits[0] - hits[1]) > STOP_REPEAT:
        raise RuntimeError('%s: stop did not repeat within %.2f deg (%.2f, %.2f). Nothing saved.'
                           % (name, STOP_REPEAT / DEG, hits[0] / DEG, hits[1] / DEG))
    correction = float(np.mean(hits)) - reference
    if abs(correction) > tolerance:
        # the measured hard stop wins over CAD; flag it so a wrong CAD angle is easy to spot
        print('  WARNING %s: contact %.2f deg from the CAD stop %.2f deg (more than %.0f deg). The zero '
              'puts the measured stop at the CAD angle; if the starting pose was close, the CAD '
              'stop angle is what is off.' % (name, correction / DEG, reference / DEG, tolerance / DEG),
              flush=True)
    # q = direction * counts / scale - zero: a hit read at `hit` that is really at
    # `reference` means the zero grows by hit - reference.
    return float(seed_zero) + correction


def rough_pose(arm, which='home'):
    """The pose the operator sets by hand: home_pose_rad, or 'hanging' (elbow straight)."""
    pose = np.asarray(arm.home, float).copy()
    if which == 'hanging':
        pose[arm.names.index('elbow')] = 0.0
    elif which != 'home':
        raise ValueError('--stop-pose must be home or hanging.')
    return pose


def rough_seed(arm, counts, previous, start=None):
    """URDF-frame start-up calibration anchored on the operator's rough pose."""
    start = np.asarray(arm.home if start is None else start, float)
    zeros = Units(arm).counts_to_rad(counts) - start
    seed = empty_calibration(arm)
    seed.update(version=int(previous.get('version') or 0) + 1, arm_serial=arm.arm,
                joint_config_hash=arm.hash(), stage='stop_zero_pending', coordinate_frame='urdf',
                zero_offsets_rad=dict(zip(arm.names, zeros.tolist())),
                link_inertials=None, gravity_validation=None, local_gravity=None,
                commissioning={'method': 'rough_home_pending', 'verified': False,
                               'reference_counts': np.asarray(counts, float).tolist()})
    return seed


TRACE_DIR = None
TRACE_ENABLED = False     # set by run_hardware on the real launch only


def _write_trace(trace, label):
    """Per-step search trace to ~/arm_logs/stop_zero_<time>/<label>.csv (for debugging)."""
    global TRACE_DIR
    rows = trace.get('rows')
    if not rows or not TRACE_ENABLED:
        return
    try:
        import time
        if TRACE_DIR is None:
            TRACE_DIR = Path.home() / 'arm_logs' / time.strftime('stop_zero_%Y%m%d_%H%M%S')
            TRACE_DIR.mkdir(parents=True, exist_ok=True)
            print(f'  search traces: {TRACE_DIR}', flush=True)
        with open(TRACE_DIR / f'{label}.csv', 'w') as f:
            f.write('t,q_cmd_deg,q_deg,qd_rad_s,tau_nm,tau_cmd_nm,lag_deg,phase\n')
            for t, qc, q, qd, tau, tauc, lag, phase in rows:
                f.write(f'{t:.4f},{qc/DEG:.3f},{q/DEG:.3f},{qd:.4f},{tau:.3f},{tauc:.3f},{lag/DEG:.3f},{phase}\n')
            if trace.get('reason'):
                f.write(f'# {trace["reason"]}\n')
    except OSError as error:
        print(f'  (trace not written: {error})', flush=True)


def search_all(ctx, plan, pose, tolerance, seed_zeros):
    """Both hard stops per planned joint, two contacts each, searched from the pose until
    contact (no CAD angles): the planned side first, then the other.

    The zeros stay on the rough reference; contacts hold each joint's measured lower and
    upper stop and the travel between them, in those coordinates."""
    arm, model, client = ctx.arm, ctx.model, ctx.client
    peak = np.array([j.peak_torque_nm for j in arm.joints])
    max_torque = np.array([j.max_torque_nm for j in arm.joints])
    zeros = dict(seed_zeros)
    contacts = {}
    for i, (name, step) in enumerate(zip(arm.names, plan)):
        if step is None:
            print(f'{name}: no hard stops (hard_stops: false); keeps the rough home reference.', flush=True)
            continue
        start_q = float(pose[i])
        found = {}
        for direction in (step['direction'], -step['direction']):
            side = 'lower' if direction < 0 else 'upper'
            clamp = search_clamp(model, pose, i, direction, start_q, tolerance, peak, max_torque,
                                 arm.names, near=start_q, end=start_q + direction * SEARCH_TRAVEL)
            # the placeholder link masses badly underestimate gravity: search (and hold the
            # other joints) at HW_SEARCH_CLAMP_FRACTION of each joint's clamp instead
            clamp = np.maximum(clamp, HW_SEARCH_CLAMP_FRACTION * max_torque)
            print(f'{name}: creeping {"down" if direction < 0 else "up"} from {start_q/DEG:+.1f} deg to '
                  f'the {side} stop at {HW_SEARCH_SPEED/DEG:.0f} deg/s, clamp {clamp[i]:.2f} Nm',
                  flush=True)
            hits = []
            for attempt in range(2):
                trace = {}
                hit = find_stop(ctx, i, direction, pose, clamp, reference=start_q,
                                approach=0.0, beyond=SEARCH_TRAVEL, breakaway=BREAKAWAY,
                                speed=HW_SEARCH_SPEED, approach_speed=HW_MOVE_SPEED,
                                baseline='measured', clip_near=False, gravity_ff=False, trace=trace,
                                confirm_s=HW_CONTACT_CONFIRM_S)
                _write_trace(trace, f'{name}_{side}_{attempt + 1}')
                if trace.get('reason'):
                    print(f'    why: {trace["reason"]}', flush=True)
                if hit is None:
                    move_to(client, pose, max_speed=HW_MOVE_SPEED, torque_clamp=clamp, gravity_ff=False)
                    raise RuntimeError(f'{name}: no {side} stop accepted (the search runs up to '
                                       f'{SEARCH_TRAVEL/DEG:.0f} deg; a "search stopped" line above means it '
                                       f'quit early). '
                                       f'Nothing saved.')
                print(f'  contact {attempt + 1}: {hit/DEG:+.2f} deg', flush=True)
                hits.append(hit)
            if abs(hits[0] - hits[1]) > STOP_REPEAT:
                raise RuntimeError('%s: %s stop did not repeat within %.2f deg (%.2f, %.2f). Nothing saved.'
                                   % (name, side, STOP_REPEAT / DEG, hits[0] / DEG, hits[1] / DEG))
            found[side] = {'stop_rad': float(np.mean(hits)), 'rough_angles_rad': [float(h) for h in hits]}
            print(f'  {side} hard stop at {found[side]["stop_rad"]/DEG:+.2f} deg', flush=True)
            move_to(client, pose, max_speed=HW_MOVE_SPEED, settle=0.5, torque_clamp=clamp, gravity_ff=False)
        lo, hi = found['lower']['stop_rad'], found['upper']['stop_rad']
        contacts[name] = {'lower': found['lower'], 'upper': found['upper'], 'travel_rad': hi - lo,
                          'center_rad': 0.5 * (lo + hi)}
        urdf_lo, urdf_hi = float(model.lower[i]), float(model.upper[i])
        print(f'  {name}: stops {lo/DEG:+.2f} .. {hi/DEG:+.2f} deg, travel {(hi - lo)/DEG:.2f} deg '
              f'(URDF range {urdf_lo/DEG:+.1f} .. {urdf_hi/DEG:+.1f}, {(urdf_hi - urdf_lo)/DEG:.1f} deg)',
              flush=True)
        if lo > urdf_lo + 10 * DEG and hi < urdf_hi - 10 * DEG or hi < urdf_lo or lo > urdf_hi:
            print(f'  WARNING {name}: the measured stops do not cover the URDF range; check the '
                  f'direction sign, encoder scale and the rough reference.', flush=True)
    return zeros, contacts


def measured_soft_limits(arm, lower, upper, contacts):
    """3 deg inside both measured stops, kept inside the URDF range; URDF range elsewhere."""
    limits = {}
    for i, name in enumerate(arm.names):
        lo, hi = float(lower[i]), float(upper[i])
        if name in contacts:
            m_lo = contacts[name]['lower']['stop_rad'] + SOFT_INSET
            m_hi = contacts[name]['upper']['stop_rad'] - SOFT_INSET
            lo, hi = max(lo, m_lo), min(hi, m_hi)
            if lo >= hi:
                # the measured range and the URDF range do not overlap: trust the stops
                print('  WARNING %s: measured stops and the URDF range do not overlap; soft limits '
                      'are the measured stops (check the direction sign).' % name, flush=True)
                lo, hi = m_lo, m_hi
        limits[name] = [lo, hi]
    return limits


def stop_soft_limits(arm, lower, upper, plan):
    """3 deg inside each measured stop, the URDF range elsewhere (inside it either way)."""
    limits = {}
    for i, (name, step) in enumerate(zip(arm.names, plan)):
        lo, hi = float(lower[i]), float(upper[i])
        if step is not None and step['side'] == 'lower':
            lo = max(lo, step['reference'] + SOFT_INSET)
        elif step is not None:
            hi = min(hi, step['reference'] - SOFT_INSET)
        limits[name] = [lo, hi]
    return limits


def run_hardware(args, reader=None, launch_factory=None):
    from ..cli import build_model
    from ..hw.client import Command
    from .autocal import MotionMonitor
    from .commission import EncoderReader

    if args.force or args.travel_deg is not None:
        raise ValueError('Stop-zero accepts neither --force nor --travel-deg; '
                         'use --stop-tolerance-deg for the starting-pose tolerance.')
    tolerance_deg = args.stop_tolerance_deg if getattr(args, 'stop_tolerance_deg', None) is not None \
        else DEFAULT_TOLERANCE_DEG
    if not np.isfinite(tolerance_deg) or not 2 <= tolerance_deg <= 20:
        raise ValueError('--stop-tolerance-deg must be between 2 and 20 degrees.')
    tolerance = tolerance_deg * DEG
    arm = load_arm(args.arm)
    if arm.home is None:
        raise ValueError('Configure home_pose_rad: the rough starting pose for stop-zero.')
    start = rough_pose(arm, getattr(args, 'stop_pose', None) or 'home')
    joint_hash = arm.hash()
    store = CalibrationStore(arm, args.calib_root)
    previous_path = store.latest_path()
    previous = store.latest()
    stops = stop_references(arm)
    nominal = build_model(arm, {})
    if (getattr(args, 'stop_pose', None) or 'home') == 'hanging':
        # search from the hanging pose itself (3 deg inside the limits): the straight elbow
        # carries almost no gravity, which the placeholder link masses badly underestimate
        hanging = nominal.copy()
        hanging.home = start
        pose = safe_home(hanging)
    else:
        pose = safe_home(nominal)
    plan = plan_stops(arm, pose, stops)
    # debugging: STOP_ZERO_ONLY=elbow,wrist_yaw searches only those joints and never saves
    import os
    only = [n.strip() for n in os.environ.get('STOP_ZERO_ONLY', '').split(',') if n.strip()]
    if only:
        unknown = sorted(set(only) - set(arm.names))
        if unknown:
            raise ValueError('STOP_ZERO_ONLY: unknown joints %s' % ', '.join(unknown))
        plan = [step if name in only else None for name, step in zip(arm.names, plan)]
        print('DEBUG STOP_ZERO_ONLY=%s: searching only those joints; nothing will be saved.'
              % ','.join(only), flush=True)

    print('Stop-zero: place the whole arm ROUGHLY (within %.0f deg per joint) at this pose, supported, '
          'drives disabled:' % tolerance_deg)
    for name, value in zip(arm.names, start):
        print('  %-15s %+7.2f deg' % (name, value / DEG))
    print('Then each joint creeps into both mechanical stops, twice each, with the other joints held:')
    for name, step in zip(arm.names, plan):
        if step is None:
            print('  %-15s no stop search (hard_stops: false): keeps the rough reference' % name)
        else:
            print('  %-15s %s stop first, then the other side, each until contact'
                  % (name, step['side']))
    print('Starting now: no prompts (the arm rests on its stand). Ctrl-C returns it to rest.', flush=True)
    reader = reader or EncoderReader()
    reader.disable_for_verification(arm)
    with reader.positions(arm) as positions:
        counts = positions.capture_all(arm)
    if arm.hash() != joint_hash:
        raise RuntimeError('Joint configuration changed while capturing the starting pose.')
    seed = rough_seed(arm, counts, previous, start)
    model = build_model(arm, seed)
    envelope = model.copy()
    reach = SEARCH_TRAVEL + 6 * DEG
    envelope.lower = np.minimum(model.lower, pose) - reach
    envelope.upper = np.maximum(model.upper, pose) + reach
    # the launch refuses a URDF-frame calibration without finite soft limits on every joint:
    # hand it the stop-search envelope (the MotionMonitor stops the run outside it anyway)
    seed['soft_limits_rad'] = {name: [float(lo), float(hi)] for name, lo, hi
                               in zip(arm.names, envelope.lower, envelope.upper)}

    global TRACE_ENABLED
    if launch_factory is None:
        from ..hw.launch import ArmLaunch
        launch_factory = ArmLaunch
        TRACE_ENABLED = True
    with tempfile.TemporaryDirectory(prefix='arm_stop_zero_') as directory:
        calibration_file = Path(directory) / 'pending.yaml'
        import yaml
        calibration_file.write_text(yaml.safe_dump(seed, sort_keys=False))
        # compare against exactly what the controller loads (tuples/numpy floats differ after YAML)
        seed = yaml.safe_load(calibration_file.read_text())
        launch, client = None, None
        try:
            print('Starting the position controller.', flush=True)
            launch = launch_factory(arm, calibration_file, Path(directory) / 'launch.log')
            client = launch.connect(arm, seed)
            state = client.state()
            client.send(Command(mode='P', q=state.q.copy(), gravity_ff=False))
            print('Powered position holding is active; starting the stop searches.', flush=True)
            monitored = MotionMonitor(client, arm, envelope, check_tracking=False)
            ctx = StageContext(arm=arm, client=monitored, calibration=seed, model=model, sim=False,
                               quick=False, motion_speed=HW_MOVE_SPEED)
            move_to(monitored, pose, max_speed=HW_SETTLE_SPEED, settle=0.5, gravity_ff=False)
            zeros, contacts = search_all(ctx, plan, pose, tolerance, seed['zero_offsets_rad'])
            if only:
                for name, c in contacts.items():
                    print('DEBUG %s: stops %+.2f .. %+.2f deg, travel %.2f deg' % (
                        name, c['lower']['stop_rad'] / DEG, c['upper']['stop_rad'] / DEG,
                        c['travel_rad'] / DEG), flush=True)
                return_to_rest(launch, client, start)
                print('DEBUG run: nothing saved.', flush=True)
                return 0
            if arm.hash() != joint_hash:
                raise RuntimeError('Joint configuration changed during stop-zero; nothing saved.')
            if store.latest_path() != previous_path:
                raise RuntimeError('Calibration store changed during stop-zero; nothing saved.')
            unreferenced = [n for n, step in zip(arm.names, plan) if step is None]
            path = store.write('stop_zero', {
                'coordinate_frame': 'urdf', 'zero_offsets_rad': zeros,
                'soft_limits_rad': measured_soft_limits(arm, nominal.lower, nominal.upper, contacts),
                'commissioning': {'method': 'powered_stop_zero', 'verified': False,
                                  'encoder_reference': 'mechanical_stops',
                                  'rough_reference': {'pose_rad': start.tolist(),
                                                      'counts': np.asarray(counts, float).tolist(),
                                                      'tolerance_deg': float(tolerance_deg),
                                                      'joints_left_on_it': unreferenced},
                                  'references': contacts, 'direction_and_scale': 'configured',
                                  'mechanical_stops_measured': True, 'range_kind': 'both_stops_per_joint',
                                  'relative_motion': False, 'local_checks_passed': False},
                'local_gravity': None, 'link_inertials': None, 'gravity_validation': None,
                'kinematic_overlay': None, 'friction': None, 'joint_stiffness_nm_per_rad': None,
                'pace': None, 'latency_ms': None, 'validation': None,
            })
            print('Stop-zero PASS. Saved %s' % path, flush=True)
            if unreferenced:
                print('Still on the rough reference: %s. Check them with arm indicator.'
                      % ', '.join(unreferenced), flush=True)
            return_to_rest(launch, client, start)
            print('Relaunch to load the new calibration.', flush=True)
            return 0
        except KeyboardInterrupt:
            print('\nStop-zero interrupted. Nothing saved.', flush=True)
            return_to_rest(launch, client, start)
            raise
        except Exception as error:
            print('Stop-zero failed: %s' % error, flush=True)
            return_to_rest(launch, client, start)
            raise
        finally:
            try:
                if client is not None:
                    client.close()
            finally:
                if launch is not None:
                    launch.close()


def return_to_rest(launch, client, rest):
    """Bring the arm back to the rest pose before the drives shut off (after PASS, a failure or
    Ctrl-C). A second Ctrl-C skips it; an E-stop that cut motor power leaves nothing to move."""
    from ..hw.client import Command
    if launch is None or client is None or launch.process.poll() is not None:
        return False
    print('Returning to rest (Ctrl-C again to skip)...', flush=True)
    try:
        client.send(Command(mode='P', q=client.state().q, gravity_ff=False))
        move_to(client, rest, max_speed=HW_MOVE_SPEED, settle=0.5, gravity_ff=False)
    except KeyboardInterrupt:
        print('Rest move skipped.', flush=True)
        return False
    except Exception as error:
        print('Rest move failed: %s' % error, flush=True)
        return False
    print('At rest.', flush=True)
    return True


def run(args):
    if not args.sim:
        return run_hardware(args)
    if getattr(args, 'stop_tolerance_deg', None) is not None or getattr(args, 'stop_pose', None):
        raise ValueError('--stop-tolerance-deg and --stop-pose apply to the hardware stop-zero.')
    if args.force or args.travel_deg is not None:
        raise ValueError('Stop-zero uses physical references and accepts neither --force nor --travel-deg.')
    from ..cli import build_model
    from ..sim.arm_sim import SimArm, SimErrors

    arm = load_arm(args.arm)
    model = build_model(arm, {})
    home = safe_home(model)
    stops = stop_references(arm)
    start = home.copy()
    start[arm.names.index('shoulder_pitch')] = 0.0
    print('J1 starts at 0 deg with its configured operating range.')
    # Encoder zeros are recovered from the stops, independently of the starting pose.
    bias = np.random.default_rng(args.seed).uniform(-2 * DEG, 2 * DEG, arm.n)
    errors = SimErrors(zero_offsets_rad=bias, hard_stop_margin_rad=np.zeros(arm.n))
    client = SimArm(arm, {}, errors, q0=start, seed=args.seed)
    client.stop_lower, client.stop_upper = stops.T.copy()
    start_encoders = client.state().q.copy()
    ctx = StageContext(arm, client, {}, model, sim=True, quick=args.quick)
    root = args.calib_root
    if root and Path(root).resolve() == Path(resolve('config/calib')).resolve():
        raise ValueError('Use a simulation calibration directory.')
    if args.fresh or not root:
        root = tempfile.mkdtemp(prefix='arm_stop_zero_')
    store = CalibrationStore(arm, root)
    previous = store.latest()
    zeros, contacts = {}, {}
    peak = np.array([joint.peak_torque_nm for joint in arm.joints])
    for i, name in enumerate(arm.names):
        # The loaded shoulder needs the normal approach speed for the contact dwell.
        approach_pose = start if name == 'shoulder_pitch' else home
        ctx.quick = args.quick and name != 'shoulder_pitch'
        lower, upper = stops[i]
        direction = -1 if approach_pose[i] - lower <= upper - approach_pose[i] else 1
        reference = float(lower if direction == -1 else upper)
        clamp = path_gravity(model, approach_pose, i, reference) + 0.08 * peak
        print(f'{name}: approaching simulated stop at {reference/DEG:+.2f} deg', flush=True)
        hits = []
        for _ in range(2):
            hit = find_stop(ctx, i, direction, approach_pose, clamp, reference=reference)
            if hit is None or abs(hit - reference) > 4 * DEG:
                raise RuntimeError(f'{name}: no contact near the expected reference. Nothing saved.')
            hits.append(hit)
        if abs(hits[0] - hits[1]) > 0.25 * DEG:
            raise RuntimeError(f'{name}: stop did not repeat within 0.25 deg. Nothing saved.')
        zeros[name] = float(np.mean(hits) - reference)
        contacts[name] = {'reference_rad': reference, 'encoder_angles_rad': hits}
        move_to(client, home, max_speed=0.3)
    start_error = start_encoders - np.array([zeros[n] for n in arm.names]) - start
    worst = int(np.argmax(np.abs(start_error)))
    print(f'Independent simulated start-pose check: maximum error '
          f'{abs(start_error[worst])/DEG:.3f} deg ({arm.names[worst]})')
    if abs(start_error[worst]) > 0.5 * DEG:
        raise RuntimeError('Simulated zero error exceeds 0.5 deg. Nothing saved.')
    if store.latest() != previous:
        raise RuntimeError('Calibration changed during stop-zero. Nothing saved.')
    path = store.write('stop_zero', {
        'coordinate_frame': 'urdf', 'zero_offsets_rad': zeros,
        'soft_limits_rad': {name: [None, None] for name in arm.names},
        'commissioning': {'method': 'simulated_stop_zero', 'references': contacts,
                          'relative_motion': False, 'local_checks_passed': False,
                          'range_kind': 'unmeasured', 'verified': False},
        'local_gravity': None, 'link_inertials': None, 'gravity_validation': None,
        'kinematic_overlay': None, 'friction': None, 'joint_stiffness_nm_per_rad': None,
        'pace': None, 'latency_ms': None, 'validation': None,
    })
    print(f'Saved simulated stop-referenced encoder offsets to {path}')
    return 0
