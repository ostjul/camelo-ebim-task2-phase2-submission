#!/usr/bin/env python3
"""Read-only camera, wrist-proximity, and base-obstacle operator view.

The node only subscribes. It never publishes, remaps, or changes a camera/lidar driver.
Launch it in its dedicated Humble container with ``./start_camera_viewer.bash``. Press
``q`` or Escape to exit.
"""

import argparse
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, Image, LaserScan


ROTATIONS = {
    0: None,
    90: cv2.ROTATE_90_CLOCKWISE,
    180: cv2.ROTATE_180,
    270: cv2.ROTATE_90_COUNTERCLOCKWISE,
}


def stop_ros():
    try:
        rclpy.try_shutdown()
    except Exception:
        pass


class OperatorViewer(Node):
    def __init__(self, cli_height, cli_left_rotation, cli_right_rotation):
        super().__init__("operator_camera_viewer")
        self.bridge = CvBridge()
        self.window = "TMR operator view - q/Esc to close"
        self.last_warning = {}

        # Every input topic is a ROS parameter. These are subscriptions only.
        # image_rect_raw, not image_raw: the D405's color sensor is one of its stereo
        # pair, so realsense2_camera (v4.55.1, native pixi env) treats it as inherently
        # rectified and names the topic accordingly - confirmed 2026-08-30, differs from
        # the retired Docker setup's older realsense2_camera which used image_raw.
        self.left_color_topic = self.param(
            "left_color_topic", "/wrist_camera_left/camera/color/image_rect_raw"
        )
        self.head_color_topic = self.param(
            "head_color_topic", "/head_camera/zed_node/rgb/color/rect/image"
        )
        self.right_color_topic = self.param(
            "right_color_topic", "/wrist_camera_right/camera/color/image_rect_raw"
        )
        self.left_depth_topic = self.param(
            "left_depth_topic",
            "/wrist_camera_left/camera/depth/image_rect_raw",
        )
        self.right_depth_topic = self.param(
            "right_depth_topic",
            "/wrist_camera_right/camera/depth/image_rect_raw",
        )
        self.front_lidar_topic = self.param("front_lidar_topic", "/lidar_front/scan")
        self.rear_lidar_topic = self.param("rear_lidar_topic", "/lidar_rear/scan")
        self.battery_topic = self.param("battery_topic", "/battery_state")

        # Proximity parameters. Depth values are 16UC1 millimeters; only zero is invalid.
        self.depth_roi_fraction = float(self.param("depth_roi_fraction", 0.12))
        self.depth_min_valid_pixels = int(self.param("depth_min_valid_pixels", 20))
        self.depth_percentile = float(self.param("depth_percentile", 10.0))
        self.depth_danger_cm = float(self.param("depth_danger_cm", 8.0))
        self.depth_safe_cm = float(self.param("depth_safe_cm", 30.0))

        # Planar lidar positions come from the recorded base_link static transforms.
        # Heading and angular handedness include each driver's scan ordering; together
        # the two 275-degree fields cover 360 degrees.
        #
        # CORRECTED 2026-08-30: front/rear were swapped - the operator confirmed the
        # radar's FRONT/REAR sectors reacted backwards against the physical robot (and
        # matches a persistent, low-noise close-range self-return found on live scan data
        # that fell outside the self-filter rectangle - consistent with the self-filter
        # geometry being applied to the wrong physical unit). /lidar_front/scan and
        # /lidar_rear/scan themselves were verified correct (frame_id matches topic name);
        # only the assumed mounting pose below was backwards. If this is ever wrong again,
        # re-verify by standing at a known side of the robot and checking which sector
        # reacts, not by re-deriving the transform.
        self.front_lidar_x_m = float(self.param("front_lidar_x_m", -0.3275))
        self.front_lidar_y_m = float(self.param("front_lidar_y_m", -0.2175))
        self.front_lidar_heading_rad = float(
            self.param("front_lidar_heading_rad", -3.0 * math.pi / 4.0)
        )
        # The prior front mapping was heading=-45deg/sign=-1. Mirroring left/right in
        # the ROBOT frame requires negating the complete bearing, which changes BOTH
        # values to heading=+45deg/sign=+1. Changing only the sign rotated the scan 90deg
        # and scattered front returns into side sectors.
        self.front_lidar_angle_sign = float(self.param("front_lidar_angle_sign", 1.0))
        self.rear_lidar_x_m = float(self.param("rear_lidar_x_m", 0.3275))
        self.rear_lidar_y_m = float(self.param("rear_lidar_y_m", 0.2175))
        self.rear_lidar_heading_rad = float(
            self.param("rear_lidar_heading_rad", math.pi / 4.0)
        )
        # The rear driver's LaserScan ordering is mirrored relative to the mounting-point
        # quaternion: positive native angles turn CCW in the base plane. This mapping
        # keeps native -45 degrees aimed straight rear while preserving left/right.
        self.rear_lidar_angle_sign = float(self.param("rear_lidar_angle_sign", 1.0))
        # Six 60-degree zones: three across the front half and three across the rear.
        self.radar_sector_count = int(self.param("radar_sector_count", 6))
        self.lidar_percentile = float(self.param("lidar_percentile", 5.0))
        self.lidar_red_threshold_m = float(self.param("lidar_red_threshold_m", 0.55))
        self.lidar_amber_threshold_m = float(
            self.param("lidar_amber_threshold_m", 0.90)
        )
        self.chassis_half_length_m = float(self.param("chassis_half_length_m", 0.381))
        self.chassis_half_width_m = float(self.param("chassis_half_width_m", 0.273))
        # The nominal rectangle ends inside the rounded bumpers/body panels. Returns
        # only a few centimetres outside that rectangle were dominating the 5th
        # percentile and making sectors appear permanently blocked by the robot itself.
        self.lidar_self_filter_margin_m = float(
            self.param("lidar_self_filter_margin_m", 0.08)
        )
        self.radar_max_range_m = float(self.param("radar_max_range_m", 2.0))
        self.battery_critical_percent = float(
            self.param("battery_critical_percent", 10.0)
        )
        self.battery_low_percent = float(self.param("battery_low_percent", 25.0))
        self.battery_stale_seconds = float(self.param("battery_stale_seconds", 5.0))

        self.height = int(self.param("display_height", cli_height))
        self.wrist_width = int(round(self.height * 3.0 / 4.0))
        self.head_width = int(round(self.height * 16.0 / 9.0))
        self.radar_height = int(self.param("radar_height", 260))
        self.display_fps = float(self.param("display_fps", 30.0))
        self.rotations = (
            int(self.param("left_rotation_deg", cli_left_rotation)),
            0,
            int(self.param("right_rotation_deg", cli_right_rotation)),
        )
        self.mirrors = (
            bool(self.param("left_mirror_view", True)),
            bool(self.param("head_mirror_view", True)),
            bool(self.param("right_mirror_view", True)),
        )
        self.validate_parameters()

        self.panel_specs = (
            (self.left_color_topic, "left wrist"),
            (self.head_color_topic, "head"),
            (self.right_color_topic, "right wrist"),
        )

        # Callbacks cache message references only. Conversion, filtering, and drawing are
        # all performed by render() on this single-threaded executor's timer callback.
        self.image_messages = [None, None, None]
        self.processed_image_messages = [None, None, None]
        self.frames = [None, None, None]
        self.depth_messages = [None, None]
        self.processed_depth_messages = [None, None]
        self.depth_cm = [None, None]
        self.scan_messages = [None, None]
        self.processed_scan_messages = [None, None]
        self.radar_sectors_m = [None] * self.radar_sector_count
        self.battery_message = None
        self.battery_received_at = 0.0

        for index, (topic, label) in enumerate(self.panel_specs):
            self.create_subscription(
                Image,
                topic,
                lambda message, index=index: self.cache_image(message, index),
                qos_profile_sensor_data,
            )
            self.get_logger().info(
                f"{label}: {topic} (display rotation {self.rotations[index]} deg, "
                f"mirror={'on' if self.mirrors[index] else 'off'})"
            )

        for index, topic in enumerate((self.left_depth_topic, self.right_depth_topic)):
            self.create_subscription(
                Image,
                topic,
                lambda message, index=index: self.cache_depth(message, index),
                qos_profile_sensor_data,
            )
            self.get_logger().info(f"wrist depth {index + 1}: {topic}")

        for index, topic in enumerate((self.front_lidar_topic, self.rear_lidar_topic)):
            self.create_subscription(
                LaserScan,
                topic,
                lambda message, index=index: self.cache_scan(message, index),
                qos_profile_sensor_data,
            )
            pose = (
                (self.front_lidar_x_m, self.front_lidar_y_m, self.front_lidar_heading_rad),
                (self.rear_lidar_x_m, self.rear_lidar_y_m, self.rear_lidar_heading_rad),
            )[index]
            self.get_logger().info(
                f"lidar {index + 1}: {topic} (base pose x={pose[0]:.4f}, "
                f"y={pose[1]:.4f}, heading={pose[2]:.3f} rad)"
            )

        self.create_subscription(
            BatteryState,
            self.battery_topic,
            self.cache_battery,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"battery: {self.battery_topic}")

        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        initial = self.compose_view()
        cv2.imshow(self.window, initial)
        cv2.resizeWindow(self.window, initial.shape[1], initial.shape[0])
        cv2.setWindowProperty(self.window, cv2.WND_PROP_TOPMOST, 1)
        cv2.waitKey(1)
        self.create_timer(1.0 / self.display_fps, self.render)

    def param(self, name, default):
        return self.declare_parameter(name, default).value

    def validate_parameters(self):
        if self.height < 120 or self.radar_height < 120:
            raise ValueError("display_height and radar_height must each be at least 120")
        if self.display_fps <= 0.0:
            raise ValueError("display_fps must be positive")
        if not 0.01 <= self.depth_roi_fraction <= 1.0:
            raise ValueError("depth_roi_fraction must be in [0.01, 1.0]")
        if self.depth_min_valid_pixels < 1:
            raise ValueError("depth_min_valid_pixels must be positive")
        if not 0.0 <= self.depth_percentile <= 100.0:
            raise ValueError("depth_percentile must be in [0, 100]")
        if self.depth_safe_cm <= self.depth_danger_cm:
            raise ValueError("depth_safe_cm must be greater than depth_danger_cm")
        if not 0.0 <= self.lidar_percentile <= 100.0:
            raise ValueError("lidar_percentile must be in [0, 100]")
        if self.lidar_amber_threshold_m <= self.lidar_red_threshold_m:
            raise ValueError("lidar amber threshold must be greater than red threshold")
        if self.radar_max_range_m <= 0.0:
            raise ValueError("radar_max_range_m must be positive")
        if self.chassis_half_length_m <= 0.0 or self.chassis_half_width_m <= 0.0:
            raise ValueError("chassis dimensions must be positive")
        if self.lidar_self_filter_margin_m < 0.0:
            raise ValueError("lidar_self_filter_margin_m must be nonnegative")
        if not 0.0 <= self.battery_critical_percent < self.battery_low_percent <= 100.0:
            raise ValueError("battery thresholds must satisfy 0 <= critical < low <= 100")
        if self.battery_stale_seconds <= 0.0:
            raise ValueError("battery_stale_seconds must be positive")
        if self.radar_sector_count < 4 or self.radar_sector_count > 36:
            raise ValueError("radar_sector_count must be in [4, 36]")
        if self.front_lidar_angle_sign == 0.0 or self.rear_lidar_angle_sign == 0.0:
            raise ValueError("lidar angle signs must be nonzero")
        if any(rotation not in ROTATIONS for rotation in self.rotations):
            raise ValueError("wrist rotation must be one of 0, 90, 180, 270")

    def cache_image(self, message, index):
        self.image_messages[index] = message

    def cache_depth(self, message, index):
        self.depth_messages[index] = message

    def cache_scan(self, message, index):
        self.scan_messages[index] = message

    def cache_battery(self, message):
        self.battery_message = message
        self.battery_received_at = time.monotonic()

    def warn_throttled(self, key, text):
        now = time.monotonic()
        if now - self.last_warning.get(key, 0.0) >= 5.0:
            self.get_logger().warning(text)
            self.last_warning[key] = now

    def update_cached_values(self):
        for index, message in enumerate(self.image_messages):
            if message is None or message is self.processed_image_messages[index]:
                continue
            try:
                image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                rotation = ROTATIONS[self.rotations[index]]
                # cv2.ROTATE_90_CLOCKWISE is integer 0, so truthiness would incorrectly
                # treat it as "no rotation". Only None means that rotation is disabled.
                image = (
                    cv2.rotate(image, rotation) if rotation is not None else image
                )
                self.frames[index] = cv2.flip(image, 1) if self.mirrors[index] else image
                self.processed_image_messages[index] = message
            except Exception as error:
                self.warn_throttled(f"image-{index}", f"{self.panel_specs[index][0]}: {error}")

        for index, message in enumerate(self.depth_messages):
            if message is None or message is self.processed_depth_messages[index]:
                continue
            try:
                depth = self.bridge.imgmsg_to_cv2(message, desired_encoding="passthrough")
                self.depth_cm[index] = self.measure_depth(depth)
                self.processed_depth_messages[index] = message
            except Exception as error:
                topic = (self.left_depth_topic, self.right_depth_topic)[index]
                self.warn_throttled(f"depth-{index}", f"{topic}: {error}")

        scans_changed = any(
            message is not None and message is not self.processed_scan_messages[index]
            for index, message in enumerate(self.scan_messages)
        )
        if scans_changed:
            self.radar_sectors_m = self.measure_radar_sectors()
            self.processed_scan_messages = list(self.scan_messages)

    def measure_depth(self, depth):
        if depth.ndim != 2:
            raise ValueError(f"expected a single-channel depth image, got shape {depth.shape}")
        height, width = depth.shape
        side = max(1, int(round(min(height, width) * self.depth_roi_fraction)))
        x0 = max(0, (width - side) // 2)
        y0 = max(0, (height - side) // 2)
        values = np.asarray(depth[y0 : y0 + side, x0 : x0 + side]).reshape(-1)
        values = values[values != 0]
        if values.size < self.depth_min_valid_pixels:
            return None
        return float(np.percentile(values, self.depth_percentile)) / 10.0

    def scan_points_in_base(self, scan, x_m, y_m, heading_rad, angle_sign):
        ranges = np.asarray(scan.ranges, dtype=np.float32)
        if ranges.size == 0:
            return np.empty(0), np.empty(0)
        angles = scan.angle_min + np.arange(ranges.size, dtype=np.float32) * scan.angle_increment
        valid = (
            np.isfinite(ranges)
            & (ranges >= scan.range_min)
            & (ranges <= scan.range_max)
        )
        ranges = ranges[valid]
        base_angles = heading_rad + angle_sign * angles[valid]
        point_x = x_m + ranges * np.cos(base_angles)
        point_y = y_m + ranges * np.sin(base_angles)
        # Reject returns on the robot itself before sector aggregation.
        half_length = self.chassis_half_length_m + self.lidar_self_filter_margin_m
        half_width = self.chassis_half_width_m + self.lidar_self_filter_margin_m
        outside_chassis = (
            (np.abs(point_x) > half_length)
            | (np.abs(point_y) > half_width)
        )
        point_x = point_x[outside_chassis]
        point_y = point_y[outside_chassis]
        return np.arctan2(point_y, point_x), np.hypot(point_x, point_y)

    def measure_radar_sectors(self):
        poses = (
            (
                self.front_lidar_x_m,
                self.front_lidar_y_m,
                self.front_lidar_heading_rad,
                self.front_lidar_angle_sign,
            ),
            (
                self.rear_lidar_x_m,
                self.rear_lidar_y_m,
                self.rear_lidar_heading_rad,
                self.rear_lidar_angle_sign,
            ),
        )
        sector_width = 2.0 * math.pi / self.radar_sector_count
        sector_values = [[] for _ in range(self.radar_sector_count)]
        have_scan = False
        for source_index, (scan, pose) in enumerate(zip(self.scan_messages, poses)):
            if scan is None:
                continue
            have_scan = True
            bearings, distances = self.scan_points_in_base(scan, *pose)
            # Each 275-degree lidar can see around the chassis into the opposite half.
            # Mixing both complete scans made self-returns and corner occlusions from the
            # rear lidar overwrite front sectors (and vice versa). The front unit owns
            # x>=0 / the three front sectors; the rear unit owns x<0 / the rear three.
            correct_half = (
                np.cos(bearings) >= 0.0
                if source_index == 0
                else np.cos(bearings) < 0.0
            )
            bearings = bearings[correct_half]
            distances = distances[correct_half]
            indices = np.floor(
                np.mod(bearings + sector_width / 2.0, 2.0 * math.pi) / sector_width
            ).astype(np.int32)
            for index in range(self.radar_sector_count):
                values = distances[indices == index]
                if values.size:
                    sector_values[index].append(values)
        if not have_scan:
            return [None] * self.radar_sector_count
        result = []
        for index in range(self.radar_sector_count):
            values = (
                np.concatenate(sector_values[index])
                if sector_values[index]
                else np.empty(0)
            )
            result.append(
                None
                if values.size == 0
                else float(np.percentile(values, self.lidar_percentile))
            )
        return result

    def waiting_tile(self, index):
        width = self.head_width if index == 1 else self.wrist_width
        tile = np.zeros((self.height, width, 3), np.uint8)
        cv2.putText(
            tile,
            f"{self.panel_specs[index][1]}: waiting",
            (20, self.height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return tile

    def camera_tile(self, index):
        image = self.frames[index]
        if image is None:
            tile = self.waiting_tile(index)
        else:
            source_height = image.shape[0]
            width = self.head_width if index == 1 else self.wrist_width
            interpolation = cv2.INTER_AREA if source_height > self.height else cv2.INTER_LINEAR
            tile = cv2.resize(image, (width, self.height), interpolation=interpolation)
            cv2.putText(
                tile,
                self.panel_specs[index][1],
                (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.85,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        if index in (0, 2):
            depth_index = 0 if index == 0 else 1
            self.draw_proximity(tile, self.depth_cm[depth_index])
        elif index == 1:
            self.draw_battery(tile)
        return tile

    def draw_battery(self, tile):
        message = self.battery_message
        fresh = (
            message is not None
            and time.monotonic() - self.battery_received_at <= self.battery_stale_seconds
        )
        percentage = None
        voltage = None
        if fresh:
            if math.isfinite(message.percentage) and 0.0 <= message.percentage <= 1.0:
                percentage = 100.0 * message.percentage
            elif (
                math.isfinite(message.charge)
                and math.isfinite(message.capacity)
                and message.charge >= 0.0
                and message.capacity > 0.0
            ):
                percentage = 100.0 * message.charge / message.capacity
            if math.isfinite(message.voltage) and message.voltage >= 0.0:
                voltage = message.voltage

        if percentage is None:
            color = (130, 130, 130)
            label = "BATTERY --"
            fraction = 0.0
        else:
            percentage = float(np.clip(percentage, 0.0, 100.0))
            fraction = percentage / 100.0
            if percentage < self.battery_critical_percent:
                color = (0, 0, 255)
            elif percentage < self.battery_low_percent:
                color = (0, 180, 255)
            else:
                color = (0, 220, 0)
            label = f"BATTERY {percentage:.0f}%"
            if voltage is not None:
                label += f"  {voltage:.1f}V"

        height, width = tile.shape[:2]
        icon_width = max(90, min(150, width // 5))
        icon_height = 22
        right = width - 14
        left = right - icon_width
        top = 12
        cv2.rectangle(tile, (left, top), (right, top + icon_height), (235, 235, 235), 2)
        cv2.rectangle(
            tile,
            (right + 2, top + 6),
            (right + 7, top + icon_height - 6),
            (235, 235, 235),
            -1,
        )
        fill_right = left + int(round(icon_width * fraction))
        if fill_right > left:
            cv2.rectangle(tile, (left, top), (fill_right, top + icon_height), color, -1)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0]
        cv2.putText(
            tile,
            label,
            (left - text_size[0] - 12, top + 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def draw_proximity(self, tile, distance_cm):
        height, width = tile.shape[:2]
        roi_side = max(2, int(round(min(height, width) * self.depth_roi_fraction)))
        x0 = (width - roi_side) // 2
        y0 = (height - roi_side) // 2
        cv2.rectangle(tile, (x0, y0), (x0 + roi_side, y0 + roi_side), (0, 255, 255), 2)

        bar_left, bar_right = 12, width - 12
        bar_top, bar_bottom = height - 28, height - 12
        cv2.rectangle(tile, (bar_left, bar_top), (bar_right, bar_bottom), (220, 220, 220), 2)
        if distance_cm is None:
            text, color, fraction = "depth: no reading", (160, 160, 160), 0.0
        else:
            fraction = float(
                np.clip(
                    (distance_cm - self.depth_danger_cm)
                    / (self.depth_safe_cm - self.depth_danger_cm),
                    0.0,
                    1.0,
                )
            )
            color = (0, int(round(255 * fraction)), int(round(255 * (1.0 - fraction))))
            text = f"{distance_cm:.1f} cm"
        fill_right = bar_left + int(round((bar_right - bar_left) * fraction))
        if fill_right > bar_left:
            cv2.rectangle(tile, (bar_left, bar_top), (fill_right, bar_bottom), color, -1)
        cv2.putText(
            tile,
            text,
            (12, height - 39),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
        if distance_cm is not None and distance_cm < self.depth_danger_cm:
            cv2.rectangle(tile, (3, 3), (width - 4, height - 4), (0, 0, 255), 7)

    def lidar_color(self, distance_m):
        if distance_m is None:
            return (130, 130, 130)
        if distance_m < self.lidar_red_threshold_m:
            return (0, 0, 255)
        if distance_m < self.lidar_amber_threshold_m:
            return (0, 180, 255)
        return (0, 220, 0)

    def draw_radar(self, width):
        radar = np.full((self.radar_height, width, 3), 22, np.uint8)
        origin = (width // 2, self.radar_height // 2)
        max_radius = max(35, min(width // 2 - 25, self.radar_height // 2 - 18))
        sector_width = 2.0 * math.pi / self.radar_sector_count

        for fraction in (0.25, 0.5, 0.75, 1.0):
            radius = max(1, int(round(max_radius * fraction)))
            cv2.circle(radar, origin, radius, (70, 70, 70), 1, cv2.LINE_AA)
            cv2.putText(
                radar,
                f"{self.radar_max_range_m * fraction:.1f}m",
                (origin[0] + 4, origin[1] - radius + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (130, 130, 130),
                1,
                cv2.LINE_AA,
            )

        for index in range(self.radar_sector_count):
            angle = (index + 0.5) * sector_width
            endpoint = (
                origin[0] - int(round(math.sin(angle) * max_radius)),
                origin[1] - int(round(math.cos(angle) * max_radius)),
            )
            cv2.line(radar, origin, endpoint, (42, 42, 42), 1, cv2.LINE_AA)

        scale = max_radius / self.radar_max_range_m
        chassis_half_width = max(10, int(round(self.chassis_half_width_m * scale)))
        chassis_half_height = max(14, int(round(self.chassis_half_length_m * scale)))
        cv2.rectangle(
            radar,
            (origin[0] - chassis_half_width, origin[1] - chassis_half_height),
            (origin[0] + chassis_half_width, origin[1] + chassis_half_height),
            (210, 210, 210),
            2,
        )
        cv2.putText(
            radar,
            "BASE",
            (origin[0] - 22, origin[1] + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )

        cv2.putText(radar, "FRONT", (origin[0] - 25, 15), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(radar, "REAR", (origin[0] - 20, self.radar_height - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(radar, "LEFT", (origin[0] - max_radius - 38, origin[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(radar, "RIGHT", (origin[0] + max_radius + 4, origin[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1, cv2.LINE_AA)

        for index, distance in enumerate(self.radar_sectors_m):
            bearing = index * sector_width
            arc_radius = max_radius if distance is None else min(
                max_radius, int(round(distance * scale))
            )
            arc_radius = max(8, arc_radius)
            color = self.lidar_color(distance)
            bearing_deg = math.degrees(bearing)
            half_sector_deg = math.degrees(sector_width) / 2.0 - 2.0
            # OpenCV ellipse angles increase clockwise on screen. Robot bearing zero is
            # screen-up, hence 270 degrees minus the base-frame bearing.
            arc_center_deg = 270.0 - bearing_deg
            arc_start_deg = arc_center_deg - half_sector_deg
            arc_end_deg = arc_center_deg + half_sector_deg
            axes = (arc_radius, arc_radius)
            cv2.ellipse(
                radar,
                origin,
                axes,
                0.0,
                arc_start_deg,
                arc_end_deg,
                (5, 5, 5),
                14,
                cv2.LINE_AA,
            )
            cv2.ellipse(
                radar,
                origin,
                axes,
                0.0,
                arc_start_deg,
                arc_end_deg,
                color,
                9,
                cv2.LINE_AA,
            )
            reading = "--" if distance is None else f"{distance:.1f}"
            label_radius = min(max_radius, arc_radius + 14)
            text_x = origin[0] - int(round(math.sin(bearing) * label_radius)) - 10
            text_y = origin[1] - int(round(math.cos(bearing) * label_radius)) + 5
            cv2.putText(
                radar,
                reading,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        return radar

    def compose_view(self):
        tiles = [self.camera_tile(index) for index in range(3)]
        radar = self.draw_radar(tiles[1].shape[1])
        left_footer = np.full((self.radar_height, tiles[0].shape[1], 3), 12, np.uint8)
        right_footer = np.full((self.radar_height, tiles[2].shape[1], 3), 12, np.uint8)
        return np.hstack(
            (
                np.vstack((tiles[0], left_footer)),
                np.vstack((tiles[1], radar)),
                np.vstack((tiles[2], right_footer)),
            )
        )

    def render(self):
        self.update_cached_values()
        cv2.imshow(self.window, self.compose_view())
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            stop_ros()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--height", type=int, default=540, help="camera panel height")
    # Viewer-only rotations; the right panel includes the requested extra 90-degree turn.
    parser.add_argument("--left-rotation", type=int, choices=ROTATIONS, default=270)
    parser.add_argument("--right-rotation", type=int, choices=ROTATIONS, default=90)
    args, ros_args = parser.parse_known_args()
    return args, ros_args


def main():
    args, ros_args = parse_args()
    rclpy.init(args=ros_args)
    node = OperatorViewer(args.height, args.left_rotation, args.right_rotation)
    try:
        # Single-threaded executor: subscriptions cache; the timer processes and draws.
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        stop_ros()
        node.destroy_node()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
