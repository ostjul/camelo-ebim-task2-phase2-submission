#!/usr/bin/env python3
"""Export a recorded bag's camera topics to MP4 files.

    python3 bag_to_video.py <BAG_DIR>                 # all three cameras
    python3 bag_to_video.py <BAG_DIR> --out ~/videos
    python3 bag_to_video.py <BAG_DIR> --topics /head_camera/zed_node/rgb/color/rect/image

WHY
A 40 s episode with three cameras is ~3 GB of MCAP, and viewing it needs a ROS 2 Humble
environment with the MCAP storage plugin - which the teleop laptop does not have. The same
footage as MP4 is a few tens of MB and plays in anything. Copy the MP4s, not the bag.

Runs inside the gello-humble container, which has rosbag2_py, cv_bridge and OpenCV:

    docker exec -u $(id -u):20 -e HOME=/tmp gello-humble bash -lc \\
      'source /opt/ros/humble/setup.bash && python3 /workspace/teleoperation/bag_to_video.py \\
       /workspace/teleop_bags/<BAG>'

Frame rate is measured from the message timestamps rather than assumed, so the MP4 plays
back at the speed it was actually captured - the wrist cameras and the ZED do not run at
the same rate, and a fixed 30 fps would silently stretch or compress one of them.
"""

import argparse
import os
import sys

DEFAULT_TOPICS = [
    "/wrist_camera_left/camera/color/image_raw",
    "/wrist_camera_right/camera/color/image_raw",
    "/head_camera/zed_node/rgb/color/rect/image",
]


def slug(topic):
    return topic.strip("/").replace("/", "__")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("bag", help="bag directory (the one holding metadata.yaml)")
    ap.add_argument("--out", default=None, help="output dir (default: <bag>/video)")
    ap.add_argument("--topics", nargs="*", default=DEFAULT_TOPICS)
    ap.add_argument("--fps", type=float, default=None, help="override measured fps")
    args = ap.parse_args()

    import cv2
    import rosbag2_py
    from cv_bridge import CvBridge
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import Image

    out_dir = args.out or os.path.join(args.bag, "video")
    os.makedirs(out_dir, exist_ok=True)
    bridge = CvBridge()

    # Pass 1: measure each topic's real rate from its timestamps. Cheap - we only read
    # the index, not the payloads.
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    wanted = set(args.topics)
    stamps = {t: [] for t in wanted}
    while reader.has_next():
        topic, _data, t = reader.read_next()
        if topic in wanted:
            stamps[topic].append(t)

    rates = {}
    for t, ts in stamps.items():
        if len(ts) < 2:
            continue
        span = (ts[-1] - ts[0]) / 1e9
        rates[t] = args.fps or (len(ts) - 1) / span if span > 0 else 30.0
        print(f"{t}: {len(ts)} frames over {span:.1f}s -> {rates[t]:.1f} fps")
    missing = [t for t in args.topics if t not in rates]
    for t in missing:
        print(f"{t}: not in this bag (or <2 frames) - skipped")
    if not rates:
        print("No camera topics found in the bag.", file=sys.stderr)
        return 1

    # Pass 2: decode and write. Writers are created lazily, on the first frame, because
    # the frame size is not known until then.
    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=args.bag, storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    writers, counts = {}, {t: 0 for t in rates}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    while reader.has_next():
        topic, data, _t = reader.read_next()
        if topic not in rates:
            continue
        img = bridge.imgmsg_to_cv2(deserialize_message(data, Image), desired_encoding="bgr8")
        if topic not in writers:
            h, w = img.shape[:2]
            path = os.path.join(out_dir, f"{slug(topic)}.mp4")
            writers[topic] = (cv2.VideoWriter(path, fourcc, rates[topic], (w, h)), path)
            print(f"writing {path}  ({w}x{h} @ {rates[topic]:.1f} fps)")
        writers[topic][0].write(img)
        counts[topic] += 1

    for topic, (w, path) in writers.items():
        w.release()
        size = os.path.getsize(path) / 1e6
        print(f"done: {path}  {counts[topic]} frames, {size:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
