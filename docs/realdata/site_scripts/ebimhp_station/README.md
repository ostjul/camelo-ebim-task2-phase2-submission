# ebimHP station (`~/teleoperation/station`, user `ebim`) — snapshot 2026-09-03

Copied verbatim via ssh for inspection; the rig's copies are the source of
truth. `~/teleoperation` is the site's own git checkout (see
`STATION_GIT_STATE.txt` for its HEAD, status and diff stat).

| file | role |
|---|---|
| `start_cameras.bash` | head (ZED-M) + both wrist (D405) cameras; **site-edited today**: wrists via `config_file:=configs/d405_color_640x480.yml` (U-21) — see `station_site_changes.diff` |
| `configs/d405_color_640x480.yml` | new today: `depth_module.color_profile: "640x480x30"`, `depth_module.enable_auto_exposure: false`, `depth_module.exposure: 5000` (U-21, U-37) |
| `start_robot_with_cameras.bash` | untracked in the site checkout (`station/robot/`): combined robot + cameras launcher, copied for completeness |
| `record_bag.bash` | raw rosbag of the LABS topic manifest (`--check` first) |
| `start_teleop.bash`, `start_teleop_docker.bash` | native / docker teleop bring-up (the docker path is how the corpus was recorded) |
| `start_gello.bash`, `start_pedal.bash`, `start_camera_viewer*.bash` | leader arms, foot pedals, viewer |
| `configs/teleop_common.sh`, `configs/tmr_laptop_env.sh` | shared launcher functions (`teleop_start_launch` = `ros2 launch …`), env |
| `configs/sync_robot_clock.sh` | companion clock sync (`--check`), the every-30-min routine in 16c |
| `configs/config_realsense_camera.yml`, `configs/config_zed_camera.yml` | docker-path camera configs (the realsense one already used `depth_module.color_profile`) |
| `configs/fastdds_laptop_*.xml` | the site's DDS profiles (camelo runs use `make dds-profile` → `outputs/rig/fastdds_camelo.xml` instead) |
| `configs/teleop_home_pose.yaml` | the teleop home pose (not the T6 start pose) |
| `station_site_changes.diff` | the two site edits made for camelo (cameras launcher; the station copy of `controllers.yaml`) |
| `station_other_local_changes.diff` | other uncommitted local changes found in the site checkout (controller sources, camera viewer) — not ours, recorded for context |

`MD5SUMS.txt` = checksums as read on ebimHP at copy time.
