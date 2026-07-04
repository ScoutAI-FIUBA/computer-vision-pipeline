from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from scripts.detect_players import MODEL_ID, annotate_image, detect_players
from scripts.detect_players_video import inspect_video, process_video


load_dotenv()

app = FastAPI(title="Basketball Player Detection")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def save_upload(file: UploadFile, contents: bytes, default_name: str) -> Path:
    suffix = Path(file.filename or default_name).suffix or Path(default_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        temp_file.write(contents)
        return Path(temp_file.name)


@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)) -> dict:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="Missing ROBOFLOW_API_KEY")

    temp_path = None
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")

        temp_path = save_upload(file, contents, "upload.jpg")

        result = detect_players(temp_path, api_key)
        annotated = annotate_image(temp_path, result)

        import cv2

        ok, encoded = cv2.imencode(".jpg", annotated)
        if not ok:
            raise HTTPException(status_code=500, detail="Could not encode annotated image")

        return {
            "model_id": MODEL_ID,
            "predictions": result.get("predictions", []),
            "annotated_image_base64": base64.b64encode(encoded).decode("ascii"),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/detect/video")
async def detect_video(
    file: UploadFile = File(...),
    dry_run: bool = Form(True),
    frame_stride: int = Form(30, ge=1),
    max_frames: int | None = Form(None, ge=1),
    sampled_output_only: bool = Form(True),
    reuse_last_detections: bool = Form(False),
    confidence: float = Form(0.0),
    output_fps: float | None = Form(None, gt=0),
) -> dict:
    input_path = None
    output_path = None
    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty upload")

        input_path = save_upload(file, contents, "upload.mp4")
        info = inspect_video(input_path, frame_stride, max_frames)

        if dry_run:
            return {
                "model_id": MODEL_ID,
                "dry_run": True,
                **info,
            }

        api_key = os.getenv("ROBOFLOW_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="Missing ROBOFLOW_API_KEY")

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            output_path = Path(temp_file.name)

        stats = process_video(
            input_path,
            output_path,
            api_key,
            frame_stride=frame_stride,
            max_frames=max_frames,
            sampled_output_only=sampled_output_only,
            reuse_last_detections=reuse_last_detections,
            confidence=confidence,
            output_fps=output_fps,
        )

        return {
            **stats,
            "content_type": "video/mp4",
            "filename": f"{Path(file.filename or 'upload.mp4').stem}_detections.mp4",
            "annotated_video_base64": base64.b64encode(output_path.read_bytes()).decode("ascii"),
        }
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    finally:
        if input_path is not None:
            input_path.unlink(missing_ok=True)
        if output_path is not None:
            output_path.unlink(missing_ok=True)
