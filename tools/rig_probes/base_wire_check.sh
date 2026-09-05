#!/usr/bin/env bash
# Base wire facts the approach depends on — run in the station's pixi shell
# (camelo DDS profile, domain 0) with start_base running on the companion.
# Answers the submission README §8 wire questions without moving anything
# (never yet run on the rig — the team's access ended before it was written):
#   * message TYPE on /swerve_drive_controller/cmd_vel (contracts assume
#     geometry_msgs/msg/TwistStamped, per the site's base_nudge.py) and on
#     cmd_vel_out; a mismatch means camelo's publisher never matches and the
#     base silently never moves — flip TopicMap.base_cmd_stamped in that case
#   * odom type/rate and frame ids (the filter's relative-odometry channel)
#   * whether anything publishes camera_info for the head camera (the
#     perception profile carries a self-calibrated model because nothing did)
set -u
CMD=/swerve_drive_controller/cmd_vel
OUT=/swerve_drive_controller/cmd_vel_out
ODOM=/swerve_drive_controller/odom
echo "== topic types =="
for t in "$CMD" "$OUT" "$ODOM"; do
  printf '%-40s ' "$t"; ros2 topic type "$t" 2>/dev/null || echo "(absent — is start_base running?)"
done
echo "== subscribers on $CMD (must be >= 1 before the approach starts) =="
ros2 topic info "$CMD" 2>/dev/null | grep -E 'Publisher|Subscription' || echo "(no info)"
echo "== odom frame ids + one sample =="
timeout 5 ros2 topic echo --once "$ODOM" 2>/dev/null | grep -E 'frame_id|^ *x:|^ *y:|^ *z:' | head -8 || echo "(no odom sample in 5 s)"
echo "== odom rate (wire; ~50 Hz expected) =="
timeout 6 ros2 topic hz "$ODOM" 2>/dev/null | tail -n 2 || echo "(no rate)"
echo "== camera_info topics (expected: none on this rig) =="
ros2 topic list 2>/dev/null | grep -i camera_info || echo "(none)"
echo "== head image type/rate =="
ros2 topic type /head_camera/zed_node/rgb/color/rect/image 2>/dev/null || echo "(head image absent)"
