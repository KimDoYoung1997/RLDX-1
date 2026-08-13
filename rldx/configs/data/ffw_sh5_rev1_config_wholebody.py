import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig

def _abs_non_eef() -> ActionConfig:
    return ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
    )


ffw_sh5_rev1_wholebody = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["cam_head"]),
    # head (54:56) and lift (56:57) are present in the raw 57-dim vectors but
    # deliberately excluded — this embodiment only trains the two arms and hands.
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(16)),
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
        action_configs=[_abs_non_eef() for _ in range(4)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(ffw_sh5_rev1_wholebody, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(ffw_sh5_rev1_wholebody), EmbodimentTag.NEW_EMBODIMENT)
