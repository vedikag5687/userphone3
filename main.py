from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from ultralytics import YOLO
import os
from typing import Dict
import tempfile
import shutil

app = FastAPI()

class DriverPhoneDetector:
    def __init__(self):
        # Load the YOLOv8 model
        self.model = YOLO('yolov8n.pt')
        self.confidence_threshold = 0.35

    def is_in_driving_area(self, box, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        driving_area = {
            'x_min': frame_width * 0.2,
            'x_max': frame_width * 0.8,
            'y_min': frame_height * 0.2,
            'y_max': frame_height * 0.9
        }
        
        box_center_x = (x1 + x2) / 2
        box_center_y = (y1 + y2) / 2
        
        return (driving_area['x_min'] < box_center_x < driving_area['x_max'] and
                driving_area['y_min'] < box_center_y < driving_area['y_max'])

    def check_phone_size(self, box, frame_shape):
        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = map(int, box)
        
        box_width = x2 - x1
        box_height = y2 - y1
        
        max_dimension = max(box_width, box_height)
        min_dimension = min(box_width, box_height)
        
        max_allowed = frame_width * 0.4
        min_allowed = frame_width * 0.03
        
        return (min_allowed < max_dimension < max_allowed and
                min_allowed/2 < min_dimension < max_allowed/2)

    def detect_phone_use(self, frame):
        results = self.model(frame, conf=self.confidence_threshold)
        
        phone_detected = False
        max_confidence = 0.0
        in_driving_area = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf[0])
                
                if class_name == 'cell phone':
                    detected_box = box.xyxy[0].cpu().numpy()
                    
                    if not self.check_phone_size(detected_box, frame.shape):
                        continue
                    
                    phone_detected = True
                    max_confidence = max(confidence, max_confidence)
                    
                    if self.is_in_driving_area(detected_box, frame.shape):
                        in_driving_area = True
                        break
        
        return phone_detected, max_confidence, in_driving_area

    async def process_video(self, video_path: str) -> Dict:
        cap = None
        frame_count = 0
        phone_detected_frames = 0
        driving_violations_count = 0
        max_confidence = 0.0
        
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                return {"error": "Could not open video file"}
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                phone_detected, confidence, in_driving_area = self.detect_phone_use(frame)
                
                if phone_detected:
                    phone_detected_frames += 1
                    max_confidence = max(max_confidence, confidence)
                    if in_driving_area:
                        driving_violations_count += 1
            
            if frame_count == 0:
                return {"error": "No frames were processed"}
            
            phone_detection_rate = (phone_detected_frames / frame_count) * 100
            final_phone_detected = phone_detection_rate > 50
            
            return {
                "phone_detected": final_phone_detected,
               }
            
        except Exception as e:
            return {"error": f"Error during video processing: {str(e)}"}
        
        finally:
            if cap is not None:
                cap.release()
                cv2.destroyAllWindows()

# Initialize the detector
detector = DriverPhoneDetector()

@app.post("/detect-phone")
async def detect_phone(video: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Save uploaded file to temporary location
            temp_video_path = os.path.join(temp_dir, 'temp_video.mp4')
            
            with open(temp_video_path, 'wb') as temp_file:
                shutil.copyfileobj(video.file, temp_file)
            
            # Process the video
            results = await detector.process_video(temp_video_path)
            
            return JSONResponse(content=results)
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": f"Error processing video: {str(e)}"}
            )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)