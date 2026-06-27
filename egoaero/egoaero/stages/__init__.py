"""Stage package: import all 8 stage modules so callers can do
``from .stages import stage0_ego_io, ...``."""
from . import (  # noqa: F401
    stage0_ego_io,
    stage1_semantic,
    stage2_track,
    stage3_mesh,
    stage4_hand,
    stage5_ego_comp,
    stage6_contact,
    stage7_eval,
    stage8_quality,
)
