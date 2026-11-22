# Rejected arm01 calibrations

Kept for the record; `CalibrationStore` and the launch only read
`config/calib/arm01/*_vN.yaml`, so nothing in this folder is ever loaded.

## 2026-11-15_v1.yaml (`arm reference`, relative frame)

- `reference_counts` for shoulder_roll (J2) and shoulder_yaw (J3) are exactly 0.
  Those drives were not reporting 0x6064 (they were on the `legacy` PDO profile
  at 6a27ac0, the commit the file names), or reset their position at power on.
  A relative zero built on them is wrong after the next power cycle.
  `arm reference` / `arm commission` now refuse a joint that reads exactly 0.
- `joint_config_hash` (214d9b610c1d8bde) matches neither the arm01.yaml of
  6a27ac0 nor the current one (J5 changed to an x4_10 since), so
  `enable_motion:=true` refused to launch with it anyway.
- As the active (latest) file it set `coordinate_frame: relative`, which blocks
  every normal stage (`arm zero`, `arm bringup`, `arm limits`, ...).

Once J2 and J3 report their absolute position, rerun `arm reference` (or
`arm commission --write-config`) to make a new one.
