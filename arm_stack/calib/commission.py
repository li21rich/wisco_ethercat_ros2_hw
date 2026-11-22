"""Passive commissioning of a supported arm from measured joint angles."""

from contextlib import contextmanager
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

from ..config import CalibrationStore, load_arm


def require_live_counts(names, samples):
    """Refuse a joint whose 0x6064 reads exactly 0 in every sample. A settled 18-bit
    absolute encoder essentially never does; 0 means the drive is not reporting its
    position (an unmapped or legacy PDO profile, the controller's q not received) or
    reset it at power on, and a reference or fit from it would be wrong on every
    restart. samples: (k, n) or (n,)."""
    samples = np.atleast_2d(np.asarray(samples, float))
    dead = [names[i] for i in range(samples.shape[1]) if np.all(samples[:, i] == 0.0)]
    if dead:
        raise RuntimeError('%s: 0x6064 reads exactly 0 counts. The drive is not reporting its position '
                           '(PDO profile, encoder) or reset it at power on; fix that before capturing.'
                           % ', '.join(dead))


class EncoderReader:
    method = 'passive_sdo'

    @contextmanager
    def positions(self, arm):
        """Use cyclic encoder feedback on arm01. Other rigs use their SDO reader."""
        if arm.arm == 'arm01' and arm.control_mode == 'arm':
            from ..hw.launch import passive_feedback
            with passive_feedback(arm, {}) as client:
                yield CyclicPositions(client)
        else:
            yield self

    def command(self, *args):
        result = subprocess.run(['ethercat', *args], capture_output=True, text=True, timeout=5)
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or 'EtherCAT command failed')
        return result.stdout

    def require_idle_master(self, arm):
        master = self.command('master')
        if not re.search(r'Active:\s+no\b', master):
            raise RuntimeError('Stop ros2 launch before commissioning; the master is active.')
        count = re.search(r'Slaves:\s+(\d+)', master)
        if not count or int(count.group(1)) != arm.n:
            raise RuntimeError('Expected exactly %d responding drives.' % arm.n)

    def disable_for_verification(self, arm):
        self.require_idle_master(arm)
        for joint in arm.joints:
            if self.upload(joint.bus_position, 'uint16', '0x6041') & 4:
                self.require_idle_master(arm)
                print(f'Disabling {joint.name} for passive encoder capture.', flush=True)
                self.command('download', '-p', str(joint.bus_position), '-t', 'uint16', '0x6040', '0', '0')
                deadline = time.monotonic() + 2
                while self.upload(joint.bus_position, 'uint16', '0x6041') & 4:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f'{joint.name} still reports operation enabled after disable.')
                    time.sleep(0.05)
        self.require_idle(arm)

    def require_idle(self, arm):
        self.require_idle_master(arm)
        for joint in arm.joints:
            if self.upload(joint.bus_position, 'uint16', '0x6041') & 4:
                raise RuntimeError('%s is enabled. Disable the drives before commissioning.' % joint.name)

    def upload(self, position, kind, index):
        output = self.command('upload', '-p', str(position), '-t', kind, index, '0')
        return int(output.split()[-1])

    def capture(self, arm, joint):
        self.require_idle(arm)
        values = []
        for _ in range(20):
            values.append(self.upload(joint.bus_position, 'int32', '0x6064'))
            time.sleep(0.025)
        self.require_idle(arm)
        if np.ptp(values) > joint.counts_per_rev * 0.2 / 360:
            raise RuntimeError('%s moved during measurement; hold it stationary and retry.' % joint.name)
        require_live_counts([joint.name], np.asarray(values, float)[:, None])
        return float(np.mean(values))

    def capture_all(self, arm):
        """Sample every joint together so a settling arm cannot mix poses."""
        self.require_idle(arm)
        samples = []
        for _ in range(20):
            samples.append([self.upload(j.bus_position, 'int32', '0x6064') for j in arm.joints])
            time.sleep(0.025)
        self.require_idle(arm)
        samples = np.asarray(samples, float)
        for i, joint in enumerate(arm.joints):
            if np.ptp(samples[:, i]) > joint.counts_per_rev * 0.2 / 360:
                raise RuntimeError('%s moved during measurement; let the whole arm settle and retry.' % joint.name)
        require_live_counts(arm.names, samples)
        return np.mean(samples, axis=0)


class CyclicPositions:
    method = 'passive_cyclic'

    def __init__(self, client):
        self.client = client

    def capture_all(self, arm):
        samples = []
        for sample in range(12):
            if sample:
                time.sleep(0.05)
            state = self.client.state()
            if np.any(state.status & 4):
                raise RuntimeError('A drive is enabled during passive encoder capture.')
            counts = np.asarray(state.raw_counts, float)
            if counts.shape != (arm.n,) or not np.isfinite(counts).all():
                raise RuntimeError('Invalid cyclic encoder readings.')
            samples.append(counts.copy())
        scale = np.array([j.counts_per_rev for j in arm.joints])
        moving = np.flatnonzero(np.ptp(samples, axis=0) * 360 / scale > 0.2)
        if moving.size:
            names = ', '.join(arm.names[i] for i in moving)
            raise RuntimeError(f'{names} moved during capture. Let the supported arm settle.')
        require_live_counts(arm.names, samples)
        return np.mean(samples, axis=0)

    def capture(self, arm, joint):
        return self.capture_all(arm)[arm.names.index(joint.name)]


def angle(prompt):
    while True:
        try:
            value = float(input(prompt))
            if math.isfinite(value):
                return math.radians(value)
        except ValueError:
            pass
        print('Enter a finite angle in degrees relative to the robot model zero pose.')


MIN_SPAN = math.radians(5)
SCALE_TOLERANCE = 0.04


def fit_encoder(count_a, angle_a, count_b, angle_b, configured_cpr=None, gear_ratio=None):
    """Direction, counts per output revolution and zero from two measured angles.

    Two hand-read angles cannot set the encoder scale to better than a few percent
    (0.25 deg of reading error over a 10 deg span is 2.5%), and a scale that is a few
    percent off skews every position and velocity. So the measured scale only picks
    between the plausible encoder scales, the configured counts_per_rev or that
    times / over the gear ratio (an output- vs motor-side encoder), and must be
    within 4% of one of them; anything else is refused as a mapping or reading
    error. Without configured_cpr (no candidates) the measured value is returned.
    """
    if abs(angle_b - angle_a) < MIN_SPAN:
        raise ValueError('The two reference angles must differ by at least %.0f degrees.'
                         % math.degrees(MIN_SPAN))
    slope = (count_b - count_a) / (angle_b - angle_a)
    if not math.isfinite(slope) or abs(count_b - count_a) < 10:
        raise ValueError('Insufficient encoder movement. Check the bus position and joint mapping.')
    measured = abs(slope) * 2 * math.pi
    counts_per_rev = round(measured)
    if configured_cpr:
        candidates = [float(configured_cpr)]
        if gear_ratio and gear_ratio > 1:
            candidates += [configured_cpr * gear_ratio, configured_cpr / gear_ratio]
        best = min(candidates, key=lambda c: abs(measured / c - 1))
        if abs(measured / best - 1) > SCALE_TOLERANCE:
            raise ValueError('Measured %.0f counts per output revolution, more than %.0f%% from every '
                             'plausible scale (%s). Recheck the two angles and the joint mapping.'
                             % (measured, SCALE_TOLERANCE * 100,
                                ', '.join('%.0f' % c for c in candidates)))
        counts_per_rev = int(round(best))
    direction = 1 if slope > 0 else -1
    zero = direction * count_a * 2 * math.pi / counts_per_rev - angle_a
    return direction, counts_per_rev, zero


def atomic_text(path, text):
    path = Path(path)
    with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.chmod(temporary, path.stat().st_mode & 0o777)
    temporary.replace(path)


def run(args, reader=None):
    if not args.write_config:
        raise RuntimeError('Use arm commission --write-config to save measured encoder scales and directions.')
    if args.sim:
        raise RuntimeError('arm commission reads physical encoders. Use arm all --sim for simulation.')
    arm = load_arm(args.arm)
    reader = reader or EncoderReader()
    reader.require_idle(arm)
    print('Support the arm and keep the drives disabled. Keep the electronics powered for encoder feedback.')
    print('Measure physical joint angles relative to the URDF zero pose with an angle reference.')
    print('For each joint, support the other joints and move only the named joint.')
    original = Path(arm.path).read_text()
    updated = original
    zeros, limits, references = {}, {}, {}
    with reader.positions(arm) as positions:
        for joint in arm.joints:
            print('\n%s, EtherCAT bus position %d' % (joint.name, joint.bus_position))
            a = angle('Place at reference angle A, then enter its measured degrees: ')
            count_a = positions.capture(arm, joint)
            b = angle('Place at reference angle B (at least 5 deg away; 30+ reads the scale far better), '
                      'enter measured degrees: ')
            count_b = positions.capture(arm, joint)
            row = next(r for r in arm.raw['joints'] if r['name'] == joint.name)
            direction, cpr, zero = fit_encoder(count_a, a, count_b, b, joint.counts_per_rev,
                                               float(row.get('gear_ratio') or 0) or None)
            input('Return to reference angle A and press Enter to verify repeatability: ')
            repeat = positions.capture(arm, joint)
            if abs(repeat - count_a) * 360 / cpr > 0.2:
                raise RuntimeError('Reference A repeatability exceeds 0.2 degrees; calibration was not saved.')
            print('Measured direction %+d, counts/output revolution %d, zero %.4f deg' %
                  (direction, cpr, math.degrees(zero)))
            input('Manually place at the LOWER usable endpoint, allowing mechanical/cable clearance. Enter: ')
            low = direction * positions.capture(arm, joint) * 2 * math.pi / cpr - zero
            input('Manually place at the UPPER usable endpoint, allowing mechanical/cable clearance. Enter: ')
            high = direction * positions.capture(arm, joint) * 2 * math.pi / cpr - zero
            inset = math.radians(3)
            if high - low <= 2 * inset:
                raise RuntimeError('Endpoints are reversed or less than 6 degrees apart; calibration was not saved.')
            limits[joint.name] = [low + inset, high - inset]
            zeros[joint.name] = zero
            references[joint.name] = {'angles_rad': [a, b], 'counts': [count_a, count_b],
                                      'direction': direction, 'counts_per_rev': cpr}
            print('Software limits %.2f to %.2f degrees (3 deg inside the measured endpoints).' %
                  tuple(math.degrees(v) for v in limits[joint.name]))
            input('Return this joint to a supported pose INSIDE those software limits, then press Enter: ')
            rest = direction * positions.capture(arm, joint) * 2 * math.pi / cpr - zero
            if not limits[joint.name][0] <= rest <= limits[joint.name][1]:
                raise RuntimeError('%s is outside its selected software limits; calibration was not saved.' % joint.name)
            pattern = re.compile(r'(^[ \t]*-[ \t]+name:[ \t]*' + re.escape(joint.name) +
                                 r'\b.*?)(?=^[ \t]*-[ \t]+name:|^\S|\Z)',
                                 re.M | re.S)
            def replace(match):
                block = match.group(0)
                for key, value in [('direction', direction), ('counts_per_rev', cpr)]:
                    block, count = re.subn(r'(^[ \t]+' + key + r':[ \t]*)[^\s#]+',
                                           lambda m: m.group(1) + str(value), block, flags=re.M)
                    if count != 1:
                        raise RuntimeError('Expected one %s field for %s' % (key, joint.name))
                return block
            updated, count = pattern.subn(replace, updated)
            if count != 1:
                raise RuntimeError('Cannot locate joint %s in the configuration.' % joint.name)
    reader.require_idle(arm)
    if Path(arm.path).read_text() != original:
        raise RuntimeError('Joint configuration changed during measurement. Calibration was not saved.')
    atomic_text(arm.path, updated)
    try:
        calibrated_arm = load_arm(args.arm)
        path = CalibrationStore(calibrated_arm, args.calib_root).write('commission', {
            'coordinate_frame': 'urdf',
            'zero_offsets_rad': zeros, 'soft_limits_rad': limits,
            'commissioning': {'method': positions.method, 'references': references},
            # Recommissioning invalidates coordinate-dependent dynamics.
            'link_inertials': None, 'friction': None, 'kt_output_nm_per_a': None,
            'kinematic_overlay': None, 'latency_ms': None,
            'gravity_validation': None, 'local_gravity': None, 'pace': None, 'validation': None,
            'joint_stiffness_nm_per_rad': None,
        })
    except BaseException:
        atomic_text(arm.path, original)
        raise
    print('\nSaved measured calibration to %s' % path)
    print('Start arm01.launch.py enable_motion:=true for powered position holding and checks.')
    return 0
