"""
Export entry points pre-bound to a segment floor small enough for the test fixtures.

Fixtures across the export test modules are 30-120 steps because they test export *mechanics*,
not the length floor. Binding a small floor here keeps them from breaking every time the
production ``MIN_SEGMENT_STEPS`` moves. A test that is *about* the floor passes its own
``min_segment_steps=``, which overrides this, or imports the raw entry point directly.
"""

from functools import partial

from polyumi_ingest.export.dp import export_scene_to_dp as _export_scene_to_dp
from polyumi_ingest.export.dp import export_scenes_to_dp as _export_scenes_to_dp
from polyumi_ingest.export.dp import export_scenes_to_polyumi as _export_scenes_to_polyumi

TEST_MIN_SEGMENT_STEPS = 8

export_scene_to_dp = partial(_export_scene_to_dp, min_segment_steps=TEST_MIN_SEGMENT_STEPS)
export_scenes_to_dp = partial(_export_scenes_to_dp, min_segment_steps=TEST_MIN_SEGMENT_STEPS)
export_scenes_to_polyumi = partial(_export_scenes_to_polyumi, min_segment_steps=TEST_MIN_SEGMENT_STEPS)
