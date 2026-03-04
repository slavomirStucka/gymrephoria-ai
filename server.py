from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import shutil
import os
from squat_analyzer import SquatAnalyzer
from deadlift_analyzer import DeadliftAnalyzer
from pullup_analyzer import PullupAnalyzer
from fastapi.staticfiles import StaticFiles
import uuid


app = FastAPI()


UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
OUTPUT_FOLDER = "videos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
app.mount("/videos", StaticFiles(directory=OUTPUT_FOLDER), name="videos")


analyzer = SquatAnalyzer()
deadlift_analyzer = DeadliftAnalyzer()
pullup_analyzer = PullupAnalyzer()

@app.post("/analyze-squat")
async def analyze_squat(video: UploadFile = File(...)):

    unique_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.mp4")
    output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}.mp4")

    # Uloženie vstupného videa
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Spustenie analýzy (uloží output.mp4)
    result = analyzer.analyze(input_path)

    # Presun output.mp4 do videos/ s unikátnym názvom
    os.replace("output.mp4", output_path)

    return {
        "analysis": result,
        "video_url": f"http://127.0.0.1:8000/videos/{unique_id}.mp4"
    }

@app.post("/analyze-deadlift")
async def analyze_deadlift(video: UploadFile = File(...)):

    unique_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.mp4")
    output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}.mp4")

    # Uloženie vstupného videa
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    # Spustenie analýzy
    result = deadlift_analyzer.analyze(input_path)

    # Presun výstupného videa
    os.replace("output.mp4", output_path)

    return {
        "analysis": result,
        "video_url": f"http://127.0.0.1:8000/videos/{unique_id}.mp4"
    }

@app.post("/analyze-pullups")
async def analyze_pullups(video: UploadFile = File(...)):

    unique_id = str(uuid.uuid4())

    input_path = os.path.join(UPLOAD_FOLDER, f"{unique_id}.mp4")
    output_path = os.path.join(OUTPUT_FOLDER, f"{unique_id}.mp4")

    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    result = pullup_analyzer.analyze(input_path)

    os.replace("output.mp4", output_path)

    return {
        "analysis": result,
        "video_url": f"http://127.0.0.1:8000/videos/{unique_id}.mp4"
    }