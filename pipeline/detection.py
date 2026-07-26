"""Player detection.

`Detector` is the seam between the pipeline and whoever runs the object
detection model. Today the only implementation calls Roboflow's hosted API. If
we later move the model into the container with `inference-gpu`, that is a new
class here and nothing else in the pipeline changes.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Protocol


class Detector(Protocol):
    """Runs player detection on a single decoded frame."""

    def detect(self, frame) -> dict:
        """Return a raw Roboflow-shaped result dict with a `predictions` key."""
        ...


class HostedRoboflowDetector:
    """Calls Roboflow's hosted serverless API, one frame per request.

    The frame is passed as a numpy array. `inference_sdk` JPEG-encodes it in
    memory at the same quality the old script got by writing a temp file to disk
    and passing the path, so results match without the filesystem round-trip.
    """

    def __init__(self, *, api_key: str, api_url: str, model_id: str) -> None:
        from inference_sdk import InferenceHTTPClient

        self._client = InferenceHTTPClient(api_url=api_url, api_key=api_key)
        self._model_id = model_id
        self.call_count = 0

    def detect(self, frame) -> dict:
        self.call_count += 1
        return self._client.infer(frame, model_id=self._model_id)


class PrefetchingDetector:
    """Wraps a `Detector` so one call can be in flight while the GPU works.

    COST: a hosted detector call takes roughly 0.3s of pure network wait. Done
    inline that is 0.3s of GPU sitting idle while Modal bills per second, on
    every segment. Because the pipeline knows the next detector frame before it
    starts the current segment (it is the last frame of the current window), we
    can fire that request early and collect it later, hiding almost all of the
    latency behind SAM2 compute.

    One worker is deliberate: there is never more than one outstanding call, and
    a bigger pool would only add concurrent load against a rate-limited free-tier
    API for no gain.
    """

    def __init__(self, inner: Detector) -> None:
        self._inner = inner
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="detect")
        self._pending: Future | None = None

    @property
    def call_count(self) -> int:
        return getattr(self._inner, "call_count", 0)

    def submit(self, frame) -> None:
        """Start a detection in the background. Overwrites any unclaimed result."""

        if self._pending is not None:
            self._pending.cancel()
        self._pending = self._pool.submit(self._inner.detect, frame)

    def take(self, frame) -> dict:
        """Return the prefetched result, or detect now if nothing was prefetched."""

        pending, self._pending = self._pending, None
        if pending is None:
            return self._inner.detect(frame)
        return pending.result()

    def detect(self, frame) -> dict:
        return self.take(frame)

    def close(self) -> None:
        if self._pending is not None:
            self._pending.cancel()
            self._pending = None
        self._pool.shutdown(wait=False)

    def __enter__(self) -> "PrefetchingDetector":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def detections_from_result(
    result: dict,
    *,
    min_confidence: float = 0.0,
    allowed_classes: tuple[str, ...] | None = None,
):
    """Parse a Roboflow result into supervision Detections, filtered.

    Mirrors the filtering the original local script applied before seeding SAM2:
    drop anything below `min_confidence`, then keep only the allowed classes.
    """

    import supervision as sv

    detections = sv.Detections.from_inference(result)

    if detections.confidence is not None and min_confidence > 0.0:
        detections = detections[detections.confidence >= min_confidence]

    if allowed_classes:
        allowed = {str(name) for name in allowed_classes}
        class_names = detections.data.get("class_name", ["object"] * len(detections))
        detections = detections[[str(name) in allowed for name in class_names]]

    return detections


def build_detector(config) -> HostedRoboflowDetector:
    """Construct the detector described by `config`."""

    return HostedRoboflowDetector(
        api_key=config.roboflow_api_key,
        api_url=config.roboflow_api_url,
        model_id=config.player_model_id,
    )
