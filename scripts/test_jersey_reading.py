"""Direct test of jersey-number reading on a video.

Samples frames across a clip, runs the player detector to find `number` boxes,
crops each and asks the OCR model to read it. Reports how often a number was
actually produced — the metric that decides whether per-player stats are viable.

    python scripts/test_jersey_reading.py path/to/video.mp4
    python scripts/test_jersey_reading.py path/to/video.mp4 --step 20

Reads ROBOFLOW_API_KEY from .env. Never prints the key.
"""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

DETECTOR = "basketball-player-detection-3-ycjdo/16"
OCR = "basketball-jersey-numbers-ocr/6"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--step", type=int, default=40, help="Sample every Nth frame")
    ap.add_argument("--pad", type=int, default=4, help="Pixels of padding around each number crop")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ModuleNotFoundError:
        pass

    key = (os.getenv("ROBOFLOW_API_KEY") or "").strip()
    if not key:
        sys.exit("ROBOFLOW_API_KEY not set in .env")
    if not args.video.exists():
        sys.exit(f"Video not found: {args.video}")

    import cv2
    from inference_sdk import InferenceHTTPClient

    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=key)
    cap = cv2.VideoCapture(str(args.video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"video: {args.video.name}  {n} frames @ {fps:.0f}fps  ({n/fps:.1f}s)\n")
    print(f"{'frame':>6}  {'#boxes':>6}  {'sizes(px)':>20}  reads")

    total_boxes = total_reads = 0
    for f in range(0, n, args.step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok:
            break
        result = client.infer(frame, model_id=DETECTOR)
        nums = [p for p in result["predictions"] if p["class"] == "number"]
        total_boxes += len(nums)
        reads, sizes = [], []
        for p in nums:
            x1, y1 = int(p["x"] - p["width"] / 2), int(p["y"] - p["height"] / 2)
            x2, y2 = int(p["x"] + p["width"] / 2), int(p["y"] + p["height"] / 2)
            sizes.append(f"{x2-x1}x{y2-y1}")
            crop = frame[max(0, y1 - args.pad):y2 + args.pad, max(0, x1 - args.pad):x2 + args.pad]
            if crop.size == 0:
                continue
            try:
                resp = client.infer(crop, model_id=OCR).get("response", {})
                val = resp.get(">", "") if isinstance(resp, dict) else str(resp)
            except Exception:
                val = ""
            if val:
                reads.append(val)
                total_reads += 1
        print(f"{f:>6}  {len(nums):>6}  {','.join(sizes):>20}  {reads}")
    cap.release()

    pct = 100 * total_reads // max(1, total_boxes)
    print(f"\nTOTAL: {total_boxes} number-boxes, {total_reads} readable ({pct}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
