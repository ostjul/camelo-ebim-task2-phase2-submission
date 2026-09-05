"""Offline evaluation instruments.

Numpy-only at import time. Anything that needs torch or lerobot imports it
*inside* a function, so `make test` keeps passing with neither installed
(AGENTS.md hard rule 3).
"""
