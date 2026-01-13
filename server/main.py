from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uuid import uuid4
import os
import json
from typing import List, Optional

from analysis import run_full_analysis, generate_mask_from_points  # see below

class PointIn(BaseModel):
  x: float
  y: float
  label: Optional[int] = 1  # 1 for positive, 0 for negative

class MeasurementIn(BaseModel):
    id:        str
    type:      str
    points:    List[PointIn]
    label:     Optional[str] = None
    value:     Optional[str] = None
    realValue: Optional[str] = None

class AnnotateRequest(BaseModel):
  video_time:   float
  video_id:     str
  known_length: float
  measurements: List[MeasurementIn]

class GenerateMaskRequest(BaseModel):
  video_id: str
  video_time: float
  points: List[PointIn]  # Each point has x, y, and label (1=positive, 0=negative)


app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] to be broader
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# serve annotated videos and metrics as static files
os.makedirs("files/uploads", exist_ok=True)
os.makedirs("files/results", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="files/uploads"), name="uploads")
app.mount("/results", StaticFiles(directory="files/results"), name="results")


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file, save it, and return a video_id."""
    video_id = uuid4().hex
    save_path = f"files/uploads/{video_id}.mp4"
    with open(save_path, "wb") as f:
        f.write(await file.read())
    return {"video_id": video_id}

@app.post("/api/annotate")
async def annotate(
    data: AnnotateRequest,
    background_tasks: BackgroundTasks
):
    vid_path = f"files/uploads/{data.video_id}.mp4"
    if not os.path.exists(vid_path):
        raise HTTPException(404, "Video not found")

    # pull out your different tools:
    try:
        angle_m = next(m for m in data.measurements if m.type == "angle")
        length_m = next(m for m in data.measurements if m.type == "length")
        line_m = next(m for m in data.measurements if m.type == "line")
        track_m = next(m for m in data.measurements if m.type == "tracking")
    except StopIteration:
        raise HTTPException(400, "Missing required measurement(s)")

    line_points        = [(p.x, p.y) for p in line_m.points]
    measurement_points = [(p.x, p.y) for p in length_m.points]
    click_point        = (track_m.points[0].x, track_m.points[0].y)

    out_video   = f"files/results/{data.video_id}.mp4"
    out_metrics = f"files/results/{data.video_id}.json"

    video_time = data.video_time

    background_tasks.add_task(
        run_full_analysis,
        input_video=vid_path,
        line_pts=line_points,
        meas_pts=measurement_points,
        click_pt=click_point,
        known_length=data.known_length,
        output_video=out_video,
        output_metrics=out_metrics,
        video_time=video_time,
    )

    return {
        "status": "processing",
        "video_url":   f"/results/{data.video_id}.mp4",
        "metrics_url": f"/results/{data.video_id}.json"
    }

@app.get("/api/status/{video_id}")
def status(video_id: str):
    """Check if the annotated video is ready."""
    out_video = f"files/results/{video_id}.mp4"
    return {"ready": os.path.exists(out_video)}


@app.get("/api/metrics/{video_id}")
def get_metrics(video_id: str):
    """Return measurement results if available."""
    path = f"files/results/{video_id}.json"
    if not os.path.exists(path):
        raise HTTPException(404, "Metrics not ready")
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(content=data)

@app.post("/api/generate_mask")
async def generate_mask(data: GenerateMaskRequest):
    """
    Generate a mask for an object based on positive and negative point prompts.
    Returns the mask as a base64-encoded PNG image.
    """
    vid_path = f"files/uploads/{data.video_id}.mp4"
    if not os.path.exists(vid_path):
        raise HTTPException(404, "Video not found")

    if not data.points:
        raise HTTPException(400, "At least one point is required")

    # Convert points to the format expected by the analysis function
    points_list = [(p.x, p.y) for p in data.points]
    labels_list = [p.label for p in data.points]

    try:
        mask_base64 = generate_mask_from_points(
            input_video=vid_path,
            video_time=data.video_time,
            points=points_list,
            labels=labels_list
        )

        return {
            "status": "success",
            "mask": mask_base64,
            "video_id": data.video_id,
            "frame_time": data.video_time
        }
    except Exception as e:
        raise HTTPException(500, f"Mask generation failed: {str(e)}")
