import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
from contextlib import asynccontextmanager
import urllib.request

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Import the fix BEFORE importing basketball_referee
try:
    import yolo_loader_fix
except ImportError:
    print("Warning: yolo_loader_fix not found")

import cv2
from basketball_referee import ImprovedFreeThrowScorer, CVATDatasetConverter, FreeThrowModelTrainer

# Configuration from environment variables
MODEL_PATH = os.getenv('MODEL_PATH', '/app/models/best.pt')
MODEL_URL = os.getenv('MODEL_URL', '')  # URL to download model from
PORT = int(os.getenv('PORT', 8000))
ENVIRONMENT = os.getenv('ENVIRONMENT', 'development')

# For local development, fall back to old path if it exists
if ENVIRONMENT == 'development' and os.path.exists(
        r"C:/Users/carlc/Desktop/API  AI REFEREE MODEL/runs/detect/train3/weights/best.pt"):
    MODEL_PATH = r"C:/Users/carlc/Desktop/API  AI REFEREE MODEL/runs/detect/train3/weights/best.pt"

scorer_instance = None


def download_model():
    """Download model if not present and URL is provided"""
    if not os.path.exists(MODEL_PATH) and MODEL_URL:
        print(f"Downloading model from {MODEL_URL}...")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        try:
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
            print(f"Model downloaded successfully to {MODEL_PATH}")
            return True
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False
    return os.path.exists(MODEL_PATH)


import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
from contextlib import asynccontextmanager
import requests  # Add this import

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Import the fix BEFORE importing basketball_referee
try:
    import yolo_loader_fix
except ImportError:
    print("Warning: yolo_loader_fix not found")

import cv2
from basketball_referee import ImprovedFreeThrowScorer, CVATDatasetConverter, FreeThrowModelTrainer

# Global variables
MODEL_PATH = "best.pt"  # Relative path in the container
MODEL_URL = "https://drive.google.com/uc?export=download&id=1gFM7iLnI_ea330JrG6LlZPKLIFu4Luj6/best.pt"  # Replace with your model file URL
scorer_instance = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global scorer_instance
    print("\n" + "=" * 60)
    print("AI BASKETBALL REFEREE API STARTING")
    print("=" * 60)
    print(f"Python file: {__file__}")
    print(f"Model path: {MODEL_PATH}")

    # Download model if not exists
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        try:
            response = requests.get(MODEL_URL, timeout=30)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
            print("✅ Model downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download model: {e}")
            import traceback
            traceback.print_exc()

    print(f"Model exists: {os.path.exists(MODEL_PATH)}")

    if not os.path.exists(MODEL_PATH):
        print("❌ Model file not found!")
    else:
        try:
            print("Loading model...")
            scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
            print("✅ Model loaded successfully!")
            print(f"Scorer type: {type(scorer_instance)}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()

    print("=" * 60 + "\n")
    yield

    # Shutdown logic
    print("\n" + "=" * 60)
    print("AI BASKETBALL REFEREE API SHUTTING DOWN")
    print("=" * 60)
    if scorer_instance:
        print("Cleaning up resources...")
    print("Goodbye!")

# Create FastAPI app with lifespan
print("Creating FastAPI app...")
app = FastAPI(
    title="AI Basketball Referee API",
    version="1.0.0",
    description="Automated basketball free throw detection and scoring",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with status info."""
    return {
        "message": "AI Basketball Referee API",
        "version": "1.0.0",
        "status": "ready" if scorer_instance is not None else "model not loaded",
        "model_loaded": scorer_instance is not None,
        "environment": ENVIRONMENT,
        "endpoints": [
            "/",
            "/model_status",
            "/score_video/",
            "/train_model/",
            "/upload_model/",
            "/docs"
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "model_loaded": scorer_instance is not None
    }


@app.get("/model_status")
async def model_status():
    """Detailed model status."""
    return {
        "loaded": scorer_instance is not None,
        "path": MODEL_PATH,
        "exists": os.path.exists(MODEL_PATH),
        "size_mb": os.path.getsize(MODEL_PATH) / 1024 / 1024 if os.path.exists(MODEL_PATH) else 0,
        "scorer_type": str(type(scorer_instance)) if scorer_instance else None,
        "environment": ENVIRONMENT
    }


@app.post("/upload_model/")
async def upload_model(model_file: UploadFile = File(...)):
    """Upload a model file to use for inference"""
    global scorer_instance

    if not model_file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="Model file must be a .pt file")

    print(f"Uploading model: {model_file.filename}")

    # Save model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    content = await model_file.read()
    with open(MODEL_PATH, "wb") as f:
        f.write(content)

    print(f"Model saved to {MODEL_PATH} ({len(content) / 1024 / 1024:.2f} MB)")

    # Try to load the new model
    try:
        scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
        print("✅ New model loaded successfully!")
        return {
            "status": "success",
            "message": "Model uploaded and loaded successfully",
            "model_path": MODEL_PATH,
            "size_mb": len(content) / 1024 / 1024
        }
    except Exception as e:
        os.remove(MODEL_PATH)  # Remove invalid model
        print(f"❌ Failed to load uploaded model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@app.post("/score_video/")
async def score_video(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """Analyzes an uploaded video to detect and score free throws."""
    global scorer_instance

    print(f"\n=== Processing video: {video_file.filename} ===")

    if scorer_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please upload a model using /upload_model/ or train one using /train_model/"
        )

    # Save and process video
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / video_file.filename

        # Save video
        content = await video_file.read()
        with open(video_path, "wb") as f:
            f.write(content)
        print(f"Video saved: {len(content) / 1024 / 1024:.2f} MB")

        # Reset scorer
        scorer_instance.made_shots = 0
        scorer_instance.missed_shots = 0
        scorer_instance.shot_attempts = 0
        scorer_instance.shot_tracker.reset()

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Total frames: {total_frames}")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Run detection
            detections = scorer_instance.detect_objects(frame)
            hoop_info = scorer_instance.update_hoop_position(detections)
            ball_info = scorer_instance.find_ball(detections)
            player_bboxes = scorer_instance.find_players(detections)

            # Update shot tracking
            old_phase = scorer_instance.shot_tracker.shot_phase
            result = scorer_instance.shot_tracker.update(ball_info, hoop_info, player_bboxes, False)

            # Count attempts
            if old_phase == 'idle' and scorer_instance.shot_tracker.shot_phase == 'rising':
                scorer_instance.shot_attempts += 1
                print(f"Shot attempt #{scorer_instance.shot_attempts} at frame {frame_count}")

            # Count results
            if result == 'score':
                scorer_instance.made_shots += 1
                print(f"SCORE! Total: {scorer_instance.made_shots}")
                scorer_instance.shot_tracker.reset()
            elif result == 'miss':
                scorer_instance.missed_shots += 1
                print(f"MISS! Total: {scorer_instance.missed_shots}")
                scorer_instance.shot_tracker.reset()

            if frame_count % 100 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames")

        cap.release()

        print(f"Processing complete. Frames: {frame_count}")

        accuracy = (
                    scorer_instance.made_shots / scorer_instance.shot_attempts * 100) if scorer_instance.shot_attempts > 0 else 0

        return {
            "made_shots": scorer_instance.made_shots,
            "missed_shots": scorer_instance.missed_shots,
            "total_attempts": scorer_instance.shot_attempts,
            "accuracy_percentage": round(accuracy, 1),
            "frames_processed": frame_count
        }


@app.post("/train_model/")
async def train_model(
        cvat_zip_files: List[UploadFile] = File(..., description="CVAT YOLO 1.1 annotated datasets as ZIP files"),
        epochs: int = Form(150, description="Number of training epochs"),
        batch_size: int = Form(16, description="Training batch size"),
        model_size: str = Form("s", description="YOLO model size (n, s, m, l)"),
        device: str = Form("auto", description="Device to use for training (cpu, cuda, auto)")
) -> Dict[str, Any]:
    """
    Trains a new basketball referee model using uploaded CVAT annotated datasets.
    """
    global scorer_instance

    print(f"\n=== Training New Model ===")
    print(f"Datasets: {len(cvat_zip_files)} files")
    print(f"Epochs: {epochs}, Batch size: {batch_size}, Model size: {model_size}, Device: {device}")

    # Create temporary directories
    with tempfile.TemporaryDirectory() as temp_cvat_dir, \
            tempfile.TemporaryDirectory() as temp_dataset_dir:

        # Step 1: Save uploaded CVAT files
        uploaded_paths = []
        for i, cvat_file in enumerate(cvat_zip_files):
            cvat_path = Path(temp_cvat_dir) / f"cvat_{i}_{cvat_file.filename}"

            try:
                content = await cvat_file.read()
                with open(cvat_path, "wb") as f:
                    f.write(content)
                uploaded_paths.append(str(cvat_path))
                print(f"Saved CVAT file {i + 1}: {cvat_file.filename} ({len(content) / 1024 / 1024:.2f} MB)")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to save CVAT file {cvat_file.filename}: {e}"
                )

        # Step 2: Convert CVAT to YOLO format
        try:
            print("\nConverting CVAT datasets to YOLO format...")
            converter = CVATDatasetConverter(uploaded_paths, str(temp_dataset_dir))
            converter.convert_multiple_cvat_to_yolo()
            print("✅ Dataset conversion complete")
        except Exception as e:
            print(f"❌ Dataset conversion failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Dataset conversion failed: {e}"
            )

        # Step 3: Train the model
        try:
            print("\nStarting model training...")
            trainer = FreeThrowModelTrainer(str(temp_dataset_dir), model_size=model_size)

            # Train
            training_results = trainer.train_model(
                epochs=epochs,
                batch_size=batch_size,
                device=device
            )

            # Validate
            print("\nValidating model...")
            validation_metrics = trainer.validate_model()

            # Get the best model path
            trained_model_dir = Path("freethrow_training") / f"freethrow_yolov8{model_size}"

            # Find the best.pt file
            best_model_paths = list(trained_model_dir.rglob("best.pt"))
            if not best_model_paths:
                raise ValueError("No best.pt file found after training")

            best_model_path = best_model_paths[0]
            print(f"\nBest model saved at: {best_model_path}")

            # Step 4: Copy the new model to replace the current one
            os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

            # Backup current model if it exists
            if os.path.exists(MODEL_PATH):
                backup_path = MODEL_PATH + ".backup"
                shutil.copy2(MODEL_PATH, backup_path)
                print(f"Current model backed up to: {backup_path}")

            # Copy new model
            shutil.copy2(best_model_path, MODEL_PATH)
            print(f"New model copied to: {MODEL_PATH}")

            # Step 5: Reload the scorer with the new model
            try:
                scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
                print("✅ New model loaded successfully!")
            except Exception as e:
                print(f"⚠️ Warning: Could not reload scorer with new model: {e}")

            # Prepare response
            return {
                "status": "success",
                "message": "Model training complete!",
                "model_path": str(MODEL_PATH),
                "model_size": model_size,
                "epochs_trained": epochs,
                "batch_size": batch_size,
                "device_used": device,
                "datasets_used": len(cvat_zip_files),
                "validation_metrics": {
                    "mAP50": float(validation_metrics.box.map50),
                    "mAP50-95": float(validation_metrics.box.map),
                },
                "class_metrics": {
                    "player": {
                        "AP50": float(validation_metrics.box.ap50[0]) if len(validation_metrics.box.ap50) > 0 else None,
                        "AP50-95": float(validation_metrics.box.ap[0]) if len(validation_metrics.box.ap) > 0 else None
                    },
                    "hoop": {
                        "AP50": float(validation_metrics.box.ap50[1]) if len(validation_metrics.box.ap50) > 1 else None,
                        "AP50-95": float(validation_metrics.box.ap[1]) if len(validation_metrics.box.ap) > 1 else None
                    },
                    "ball": {
                        "AP50": float(validation_metrics.box.ap50[2]) if len(validation_metrics.box.ap50) > 2 else None,
                        "AP50-95": float(validation_metrics.box.ap[2]) if len(validation_metrics.box.ap) > 2 else None
                    }
                }
            }

        except Exception as e:
            print(f"❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"Model training failed: {str(e)}"
            )


if __name__ == "__main__":
    import uvicorn

    # For Railway deployment
    host = "0.0.0.0"
    port = PORT

    print(f"Starting server on {host}:{port}")
    print(f"Environment: {ENVIRONMENT}")

    if ENVIRONMENT == "production":
        # Production settings
        uvicorn.run(app, host=host, port=port, log_level="info")
    else:
        # Development settings
        uvicorn.run(app, host=host, port=port, reload=True, log_level="debug")
