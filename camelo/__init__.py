"""Camelo participation stack for the EBiM benchmark.

Layering (import direction only ever goes down):

    scripts/*            thin CLIs
    camelo.runner        episode loop + batch eval          (needs rclpy)
    camelo.ros           obs collector + command publisher  (needs rclpy)
    camelo.policy        adapters, backends, server         (no rclpy)
    camelo.control       chunk executor + base quantizer    (numpy only)
    camelo.contracts     mirrored benchmark contract        (numpy only)
    camelo.benchmark     benchmark checkout resolution      (stdlib only)
"""

__version__ = "0.1.0"
