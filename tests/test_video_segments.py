"""Segment windowing tests.

These guard the one piece of Phase 1 that changed behaviour rather than just
moving code: feeding SAM2 one segment at a time instead of the whole video.

Runs without a GPU, without SAM2, and without a Roboflow key — only OpenCV is
needed — which is what makes it useful on a laptop.

    python -m pytest tests/test_video_segments.py
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.video import frame_step_for, iter_segments, probe  # noqa: E402


def write_video(path: Path, frame_count: int, width: int = 64, height: int = 48, fps: int = 30) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    for index in range(frame_count):
        writer.write(np.full((height, width, 3), index % 256, dtype=np.uint8))
    writer.release()


def collect(path: Path, stride: int, max_frames: int | None = None, frame_step: int = 1):
    """Replay runner.py's emit rule and return (emitted indices, detector frames)."""

    emitted: list[int] = []
    detector_frames: list[int] = []
    for segment in iter_segments(
        path, frame_stride=stride, max_frames=max_frames, frame_step=frame_step
    ):
        detector_frames.append(segment.start_index)
        last_offset = len(segment.frames) - 1 if segment.is_last else len(segment.frames) - 2
        emitted.extend(segment.frame_indices[offset] for offset in range(last_offset + 1))
    return emitted, detector_frames


@pytest.fixture
def video_factory():
    with tempfile.TemporaryDirectory() as directory:
        def make(frame_count: int) -> tuple[Path, int]:
            path = Path(directory) / f"clip_{frame_count}.mp4"
            write_video(path, frame_count)
            return path, probe(path).frame_count

        yield make


@pytest.mark.parametrize(
    "frame_count,stride",
    [(100, 15), (91, 15), (90, 15), (46, 15), (16, 15), (15, 15), (7, 15), (1, 15), (200, 30), (50, 1)],
)
def test_every_frame_emitted_exactly_once(video_factory, frame_count, stride):
    path, actual = video_factory(frame_count)
    emitted, _ = collect(path, stride)
    assert emitted == list(range(actual))


@pytest.mark.parametrize("max_frames", [1, 16, 45, 46])
def test_max_frames_is_respected(video_factory, max_frames):
    path, actual = video_factory(100)
    emitted, _ = collect(path, 15, max_frames=max_frames)
    assert emitted == list(range(min(actual, max_frames)))


@pytest.mark.parametrize("frame_count,stride", [(100, 15), (91, 15), (46, 15), (16, 15), (15, 15)])
def test_detector_call_count_matches_original_formula(video_factory, frame_count, stride):
    """The old script ran the detector every `stride` frames starting at 0.

    Detector spend is metered per call on Roboflow's free plan, so a regression
    here costs money as well as changing tracking behaviour.
    """

    path, actual = video_factory(frame_count)
    _, detector_frames = collect(path, stride)
    assert len(detector_frames) == math.ceil(actual / stride)
    assert detector_frames == list(range(0, actual, stride))


def test_segments_overlap_by_one_frame(video_factory):
    """Consecutive segments share a frame: the next segment's detector frame."""

    path, _ = video_factory(100)
    starts = [segment.start_index for segment in iter_segments(path, frame_stride=15)]
    assert starts == [0, 15, 30, 45, 60, 75, 90]


def test_segment_window_never_exceeds_stride_plus_one(video_factory):
    """This bound is what keeps SAM2 memory flat regardless of video length."""

    path, _ = video_factory(200)
    for segment in iter_segments(path, frame_stride=15):
        assert len(segment.frames) <= 16


# --- downsampling -------------------------------------------------------


@pytest.mark.parametrize(
    "source_fps,target_fps,expected",
    [(60, 15, 4), (60, 30, 2), (30, 15, 2), (30, 30, 1), (60, None, 1), (25, 15, 2), (60, 60, 1)],
)
def test_frame_step_for(source_fps, target_fps, expected):
    assert frame_step_for(source_fps, target_fps) == expected


def test_frame_step_never_zero_when_target_exceeds_source():
    """Asking for more fps than the source has must not divide by zero or skip all."""

    assert frame_step_for(30, 60) == 1
    assert frame_step_for(0, 15) == 1


@pytest.mark.parametrize("step", [2, 4])
def test_downsampling_emits_original_frame_indices(video_factory, step):
    """Track records must reference the source video, not the sampled sequence."""

    path, actual = video_factory(100)
    emitted, detector_frames = collect(path, 15, frame_step=step)

    assert emitted == list(range(0, actual, step))
    # Detector frames are a subset of emitted frames, spaced by stride*step.
    assert set(detector_frames) <= set(emitted)
    assert detector_frames[0] == 0


@pytest.mark.parametrize("step", [2, 4])
def test_downsampling_reduces_work_proportionally(video_factory, step):
    """The whole point: 1/step as many frames reach the GPU."""

    path, actual = video_factory(100)
    full, _ = collect(path, 15, frame_step=1)
    reduced, _ = collect(path, 15, frame_step=step)
    assert len(reduced) == pytest.approx(len(full) / step, abs=1)


def test_max_frames_counts_processed_frames_not_source(video_factory):
    """max_frames bounds spend, so it must count frames that reach the GPU."""

    path, _ = video_factory(100)
    emitted, _ = collect(path, 15, max_frames=16, frame_step=4)
    assert len(emitted) == 16
    # 16 processed frames at step 4 spans 64 source frames.
    assert emitted[-1] == 60
