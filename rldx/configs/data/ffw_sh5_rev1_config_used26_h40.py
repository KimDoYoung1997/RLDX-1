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


# FFW-SH5 rev1 — 26-dim action, chunk 40. Use with `--action-horizon 40`.
#
# chunk 40 = 4.0 s at 10 fps; executing 20 of them at deployment gives the
# intended 2.0 s of committed motion, the same 20/40 ratio the paper uses on
# the ALLEX humanoid (§4.2 + Appendix C). 40 is also a chunk length RLDX-1 was
# actually trained at, unlike 50.
#
# The raw vectors are 57-dim, but the teleop stack drives far less than that.
# Measured over all 33,018 frames of rlwrld_demo_0826_merge, `action` is
# identically 0.000 on 28 of the 54 arm+hand dims: every finger joint except
# 14/15/16 and 18/19/20 on each hand (joint 17 — the spread axis — is also
# constant). head/lift (54:57) never receive a command either.
#
# So the real control space is 26-dim:
#     left_arm 7 + right_arm 7 + 4 x 3 commanded finger-flexion joints.
#
# Training the action head on all 54 is not wrong — q99 normalisation collapses
# the constant dims and the flow-matching target for them is exactly
# recoverable from the noisy input — but it spends decoder width and gradient
# on dims that carry no signal. This config trains on the 26 that move.
#
# `state` deliberately keeps the full 20-dim hands. Proprioception is free: the
# uncommanded finger joints still deflect (±0.02–0.3 rad) and that reading may
# carry contact information the commands do not. Only the *output* side is
# trimmed. state and action modality_keys are consumed independently by
# StateActionProcessor, so they do not have to match — and with every action
# ABSOLUTE there is no state_key linkage that would require it.
#
# Group names encode the raw joint indices rather than a finger numbering,
# because the 4-joints-per-finger layout is inferred, not documented.
ffw_sh5_rev1_used26_h40 = {
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
        action_configs=[_abs_non_eef() for _ in range(6)],
    ),
    "language": ModalityConfig(delta_indices=[0], modality_keys=["annotation.human.task_description"]),
}

register_modality_config(ffw_sh5_rev1_used26_h40, EmbodimentTag.GENERAL_EMBODIMENT)
register_modality_config(copy.deepcopy(ffw_sh5_rev1_used26_h40), EmbodimentTag.NEW_EMBODIMENT)
