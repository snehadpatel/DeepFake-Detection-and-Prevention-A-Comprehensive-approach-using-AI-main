import os
import cv2
import numpy as np
import torch
from services.detection import detection_service

# Test images paths
TEST_IMAGES = [
    {
        "name": "VLM_AI_Face_1",
        "path": "/Users/snehapatel/.gemini/antigravity/brain/34e9e8af-44db-4316-a6dd-c14eebaf9481/vlm_generated_face_1_1775109383673.png",
        "ground_truth": "Fake"
    },
    {
        "name": "VLM_AI_Face_2",
        "path": "/Users/snehapatel/.gemini/antigravity/brain/34e9e8af-44db-4316-a6dd-c14eebaf9481/vlm_generated_face_2_1775109413463.png",
        "ground_truth": "Fake"
    },
    {
        "name": "Real_Face_1",
        "path": "real_test_face.jpg",
        "ground_truth": "Real"
    }
]

OUTPUT_DIR = "output_vlm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_test():
    print("\n" + "="*50)
    print("      DEEPSHIELD VLM BLIND TEST      ")
    print("="*50)
    
    for test in TEST_IMAGES:
        if not os.path.exists(test["path"]):
            print(f"[!] Error: {test['name']} not found at {test['path']}")
            continue
            
        print(f"\n[*] Testing: {test['name']} (GT: {test['ground_truth']})")
        
        with open(test["path"], "rb") as f:
            img_bytes = f.read()
            
        results = detection_service.predict(img_bytes)
        
        type_pred = "Real" if results["is_real"] else "Fake"
        status = "✅ PASS" if type_pred == test["ground_truth"] else "❌ FAIL"
        
        print(f"  Result:     {type_pred} ({status})")
        print(f"  Confidence: {results['confidence']*100:.2f}%")
        
        # Save heatmap
        heatmap_path = os.path.join(OUTPUT_DIR, f"{test['name']}_heatmap.jpg")
        with open(heatmap_path, "wb") as f:
            f.write(results["heatmap"])
        print(f"  Heatmap:    {heatmap_path}")
    
    print("\n" + "="*50)
    print("  Blind Test Complete.")
    print("="*50 + "\n")

if __name__ == "__main__":
    run_test()
