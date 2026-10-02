import copy

from rldx.configs.data.embodiment_configs import register_modality_config
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import ActionConfig, ActionFormat, ActionRepresentation, ActionType, ModalityConfig


def _non_eef(rep: ActionRepresentation) -> ActionConfig:
    return ActionConfig(rep=rep, type=ActionType.NON_EEF, format=ActionFormat.DEFAULT)


# FFW-SH5 rev1 — 26-dim action, chunk 40, arms RELATIVE + fingers ABSOLUTE.
# Same groups and horizon as ffw_sh5_rev1_config_used26_h40.py; only the arm
# representation changes. Use with `--action-horizon 40`.
#
# Arms are trained as chunk-wise deltas: StateActionProcessor subtracts the
# current state of the group with the same name (state_key defaults to the
# action key, reference = state[key][-1]), so every step of the chunk is
# relative to the chunk's start, not to the previous step. This split — arms
# RELATIVE, hands/grippers ABSOLUTE — is the one this repo already uses
# wherever it uses RELATIVE (unitree_g1, behavior_r1_pro); RLWRLD's own
# ALLEX config keeps everything ABSOLUTE.
#
# Fingers stay ABSOLUTE: their commands are held set-points, and while
# grasping the joints stall on the box, so the measured state lags the
# command (2.3x the open-hand gap on the 1002 data). A delta built on that
# state would make grip strength depend on predicting the offset; an
# absolute target closes to the same level however far the box blocks it.
#
# The dataset meta/relative_stats.json is tied to (representation, horizon):
# train this on a dataset copy that an all-ABSOLUTE run has not touched, or
# the empty file that run writes is kept and normalisation fails.
ffw_sh5_rev1_used26_h40_relarm = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["cam_head"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["left_arm", "right_arm", "left_hand", "right_hand"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(40)),
        modality_keys=[
            "left_arm",
            "right_arm",
            "left_hand_j14_16",
            "left_hand_j18_20",
            "right_hand_j14_16",
            "right_hand_j18_20",
        ],
        action_configs=[
            _non_eef(ActionRepresentation.RELATIVE),  # left_arm
            _non_eef(ActionRepresentation.RELATIVE),  # right_arm
            _non_eef(ActionRepresentation.ABSOLUTE),  # left_hand_j14_16
            _non_eef(ActionRepresentation.ABSOLUTE),  # left_hand_j18_20
            _non_eef(ActionRepresentation.ABSOLUTE),  # right_hand_j14_16
            _non_eef(ActionRepresentation.ABSOLUTE),  # right_hand_j18_20
        ],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(ffw_sh5_rev1_used26_h40_relarm, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(ffw_sh5_rev1_used26_h40_relarm), EmbodimentTag.NEW_EMBODIMENT)
