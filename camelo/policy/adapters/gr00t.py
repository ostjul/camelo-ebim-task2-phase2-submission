"""GR00T adapter on the NVIDIA Isaac-GR00T native stack (``gr00t-isaac``).

NOTE: the DEFAULT GR00T path is now the lerobot-native policy
(``--adapter gr00t`` -> lerobot_generic, GR00T N1.7 is a lerobot policy
type since lerobot 0.6). This adapter remains for the Isaac-GR00T native
runtime — mainly its TensorRT deployment path (NVIDIA quotes ~10 Hz on
DGX Spark / ~36 Hz on H100 with TensorRT vs ~8-12 Hz PyTorch eager).

Follows the benchmark's ai_data_contract.yaml gr00t mapping: canonical keys
consumed as-is (video head/wrist_left/wrist_right, 16-dim proprio split
into left_arm/left_gripper/right_arm/right_gripper), full 16-step ABSOLUTE
joint-space horizon out.

Isaac-GR00T is not on PyPI — install per docker/Dockerfile.policy. GR00T
inference needs an embodiment head: zero-shot on this custom dual-FR3
platform means picking the closest available embodiment tag (or, later, a
head fine-tuned on our recorded data). Both the checkpoint and the tag
(``--gr00t-embodiment``) are configurable rather than assumed.
"""

from __future__ import annotations

import logging

import numpy as np

from camelo import contracts as C
from camelo.policy.base import Obs, PolicyAdapter

log = logging.getLogger(__name__)


class Gr00tAdapter(PolicyAdapter):
    name = "gr00t"

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        embodiment_tag: str = "new_embodiment",
    ):
        try:
            from gr00t.model.policy import Gr00tPolicy
        except ImportError as exc:
            raise RuntimeError(
                "Isaac-GR00T is not installed (this is the gr00t-isaac native-stack "
                "adapter). Install per docker/Dockerfile.policy or "
                "https://github.com/NVIDIA/Isaac-GR00T — or just use the default "
                "lerobot-native GR00T path: --adapter gr00t."
            ) from exc

        self.task = ""
        self.embodiment_tag = embodiment_tag
        # Modality config/transforms are checkpoint-specific in the GR00T
        # API; resolve them from the checkpoint where supported.
        self.policy = Gr00tPolicy(
            model_path=checkpoint,
            embodiment_tag=embodiment_tag,
            device=device,
        )
        log.info("loaded GR00T %s (embodiment_tag=%s)", checkpoint, embodiment_tag)

    def infer(self, obs: Obs) -> np.ndarray:
        proprio = obs.model_state()
        gr00t_obs = {
            "video.head": obs.images["head"][None],
            "video.wrist_left": obs.images["wrist_left"][None],
            "video.wrist_right": obs.images["wrist_right"][None],
            "state.left_arm": proprio[C.M_LEFT_ARM][None],
            "state.left_gripper": proprio[C.M_LEFT_GRIP : C.M_LEFT_GRIP + 1][None],
            "state.right_arm": proprio[C.M_RIGHT_ARM][None],
            "state.right_gripper": proprio[C.M_RIGHT_GRIP : C.M_RIGHT_GRIP + 1][None],
            "annotation.human.action.task_description": [self.task],
        }
        result = self.policy.get_action(gr00t_obs)

        chunk = np.zeros((C.MODEL_HORIZON, C.MODEL_ACTION_DIM), dtype=np.float32)
        chunk[:, C.M_LEFT_ARM] = np.asarray(result["action.left_arm"]).reshape(-1, 7)[
            : C.MODEL_HORIZON
        ]
        chunk[:, C.M_LEFT_GRIP] = np.asarray(result["action.left_gripper"]).reshape(-1)[
            : C.MODEL_HORIZON
        ]
        chunk[:, C.M_RIGHT_ARM] = np.asarray(result["action.right_arm"]).reshape(-1, 7)[
            : C.MODEL_HORIZON
        ]
        chunk[:, C.M_RIGHT_GRIP] = np.asarray(result["action.right_gripper"]).reshape(-1)[
            : C.MODEL_HORIZON
        ]
        return C.model_to_canonical_actions(chunk)
