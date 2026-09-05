"""QoS helpers — the single place subscription QoS is decided.

Isaac Sim's ROS 2 bridge publishes images (and CameraInfo) BEST_EFFORT; a
RELIABLE subscriber silently receives nothing. The real Franka
``measured_joint_states`` broadcaster is the same trap (sensor-data /
BEST_EFFORT): a RELIABLE subscriber logs
``Last incompatible policy: RELIABILITY`` and never sees arm joints.
Every image subscription, every camera_info subscription, and every
real-robot state subscription must use SENSOR_QOS (BEST_EFFORT). Sim
non-image topics stay RELIABLE (``DEFAULT_DEPTH``) — that is the working
Isaac path.
"""

from rclpy.qos import qos_profile_sensor_data, qos_profile_system_default

SENSOR_QOS = qos_profile_sensor_data
SYSTEM_QOS = qos_profile_system_default
DEFAULT_DEPTH = 10
