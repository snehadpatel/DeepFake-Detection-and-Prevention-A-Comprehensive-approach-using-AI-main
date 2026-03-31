import base64
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os

from services.detection import detection_service
from services.prevention import prevention_service

app = FastAPI(title="DeepShield AI API", description="Deepfake Detection & AI-Driven Prevention")

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/detect")
async def detect_deepfake(file: UploadFile = File(...)):
    """
    Detects if an image is fake and returns a Grad-CAM heatmap.
    """
    try:
        content = await file.read()
        result = detection_service.predict(content)
        
        # Base64 encode images for frontend
        heatmap_b64 = base64.b64encode(result["heatmap"]).decode('utf-8')
        original_b64 = base64.b64encode(result["original"]).decode('utf-8')
        
        return JSONResponse({
            "isReal": result["is_real"],
            "confidence": result["confidence"],
            "heatmap": f"data:image/jpeg;base64,{heatmap_b64}",
            "original": f"data:image/jpeg;base64,{original_b64}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/protect")
async def protect_image(file: UploadFile = File(...)):
    """
    Applies adversarial noise to protect an image from deepfake manipulation.
    """
    try:
        content = await file.read()
        result = prevention_service.protect_image(content)
        
        protected_b64 = base64.b64encode(result["protected_image"]).decode('utf-8')
        noise_b64 = base64.b64encode(result["protection_noise"]).decode('utf-8')
        original_b64 = base64.b64encode(result["original_image"]).decode('utf-8')
        
        return JSONResponse({
            "protectedImage": f"data:image/png;base64,{protected_b64}",
            "noiseMap": f"data:image/jpeg;base64,{noise_b64}",
            "originalImage": f"data:image/jpeg;base64,{original_b64}"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "service": "DeepShield AI Engine"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
