# Companion (TMR Jetson, host `ubuntu`, user `tmr-user`) site scripts — snapshot

Copied verbatim from `tmr-user@companion:~` on 2026-09-03 (via ebimHP) for
inspection; the rig's copies are the source of truth. `MD5SUMS.txt` holds the
checksums as read on the companion at copy time.

| file | role |
|---|---|
| `start_upper.bash` | upper-body bring-up (`--restart`): controller manager, `joint_impedance_controller` per arm, gripper clients, spine; log `~/start_upper.log` |
| `start_base.bash` | base bring-up; log `~/start_base.log` |
| `base_nudge.py` | base nudge helper |
| `home_arms.py` | PTP homing of both arms from a yaml (`python3 ~/home_arms.py --file ~/t5_ep163_home_pose.yaml`, see 16c) |

Not copied: `~/olix_domain_bridge.yaml`, `~/zed_override.yaml`, `~/teleop_home_pose.yaml`,
the `t5_*_home_pose.yaml` files (already tracked under `outputs/rig/t5/`), and the logs.
