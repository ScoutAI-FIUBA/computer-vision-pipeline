"""Video decoding.

The pipeline decodes the input video exactly once, with OpenCV, and hands the
decoded frames to both the detector and SAM2. The original local script decoded
twice (once with cv2 for annotation, once inside SAM2's `init_state`), which is
the thing this module exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass(frozen=True)
class VideoInfo:
    """Basic properties read from the container header."""

    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float | None:
        if self.fps <= 0 or self.frame_count <= 0:
            return None
        return self.frame_count / self.fps


def open_capture(video_path: Path):
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    return capture


def probe(video_path: Path) -> VideoInfo:
    """Read video properties without decoding any frames."""

    import cv2

    capture = open_capture(video_path)
    try:
        return VideoInfo(
            fps=float(capture.get(cv2.CAP_PROP_FPS) or 0.0),
            frame_count=int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def frame_step_for(source_fps: float, target_fps: float | None) -> int:
    """How many source frames to advance per processed frame.

    COST: this is the single biggest cost lever in the pipeline. SAM2's image
    encoder runs on every processed frame at roughly 4 frames/sec, so runtime is
    proportional to frames processed. Halving the rate halves the bill.

    60fps source at 15fps target gives step 4: three out of every four frames are
    decoded and thrown away, which is cheap CPU work, and never reach the GPU.
    """

    if not target_fps or target_fps <= 0 or source_fps <= 0:
        return 1
    return max(1, round(source_fps / target_fps))


@dataclass(frozen=True)
class Segment:
    """One tracking window: the detector frame plus the frames it propagates to.

    `frame_indices` are indices into the ORIGINAL video, so downsampling never
    changes what a track record refers to. `frames[i]` corresponds to
    `frame_indices[i]`.
    """

    frame_indices: list[int]
    frames: list
    is_last: bool

    @property
    def start_index(self) -> int:
        return self.frame_indices[0]


def _raw_segments(
    video_path: Path,
    *,
    frame_stride: int,
    max_frames: int | None,
    frame_step: int,
) -> Iterator[tuple[list[int], list]]:
    """Decode the video once, yielding `(frame_indices, frames)` windows."""

    capture = open_capture(video_path)
    try:
        window: list = []
        indices: list[int] = []
        source_index = 0
        processed = 0

        while max_frames is None or processed < max_frames:
            ok, frame = capture.read()
            if not ok:
                break

            # Decode everything, keep every `frame_step`-th. Sequential reads are
            # used rather than seeking because seeking is unreliable across codecs.
            if source_index % frame_step == 0:
                window.append(frame)
                indices.append(source_index)
                processed += 1

                # A full window is the detector frame plus `frame_stride` more.
                if len(window) == frame_stride + 1:
                    yield indices, window
                    # Carry the final frame over: it is the next detector frame.
                    window = [window[-1]]
                    indices = [indices[-1]]

            source_index += 1

        # Trailing window (video ended mid-segment). A single carried-over frame
        # still counts: the previous segment stopped one short of it precisely
        # because the detector was going to run on it here.
        if window:
            yield indices, window
    finally:
        capture.release()


def iter_segments(
    video_path: Path,
    *,
    frame_stride: int,
    max_frames: int | None = None,
    frame_step: int = 1,
) -> Iterator[Segment]:
    """Yield one tracking `Segment` at a time.

    A segment is the window SAM2 actually needs: the frame the detector runs on
    plus the `frame_stride` frames it propagates through. Consecutive segments
    overlap by one frame, because the last frame of a segment is the frame the
    next detector call runs on.

    Yielding segment-sized windows instead of the whole video is what keeps
    memory flat. SAM2 holds frames as float32 at 1024x1024x3 (~12 MB each), so
    loading a whole game would need hundreds of GB; a segment needs ~200 MB
    regardless of how long the video is.

    `frame_step` skips source frames (see `frame_step_for`). Segment frames are
    consecutive *processed* frames, but `frame_indices` always refers to the
    original video.

    `is_last` matters for output: a shared boundary frame belongs to the next
    segment (where the detector runs on it), so it is only emitted by the
    segment that has no successor.
    """

    previous: tuple[list[int], list] | None = None
    for indices, frames in _raw_segments(
        video_path,
        frame_stride=frame_stride,
        max_frames=max_frames,
        frame_step=frame_step,
    ):
        if previous is not None:
            yield Segment(frame_indices=previous[0], frames=previous[1], is_last=False)
        previous = (indices, frames)
    if previous is not None:
        yield Segment(frame_indices=previous[0], frames=previous[1], is_last=True)


class AnnotatedVideoWriter:
    """Optional MP4 writer. Only constructed when annotation is enabled."""

    def __init__(self, output_path: Path, info: VideoInfo, fps: float | None = None) -> None:
        import cv2

        output_path.parent.mkdir(parents=True, exist_ok=True)
        effective_fps = fps or info.fps or 30.0
        self._writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            effective_fps,
            (info.width, info.height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {output_path}")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "AnnotatedVideoWriter":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.release()
