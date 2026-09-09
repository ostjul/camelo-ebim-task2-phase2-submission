# Camelo — EBiM Task 2 real-robot submission: the ACT policy server.
# The runnable server ships as a prebuilt GPU container image (torch +
# lerobot + camelo); this Dockerfile pins the released version. The ROS-side
# executor runs natively on the rig from this repository's source — see
# README.md ("Install on the station") and compose.yaml.
#
#   docker build -t camelo-task2-server .
#
# release v0.1.0 (2026-09-09)
FROM ghcr.io/ostjul/camelo-ebim-task2-phase2-submission:v0.1.0
