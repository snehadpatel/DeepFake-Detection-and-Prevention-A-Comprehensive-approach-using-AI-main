import os
import sys
import cv2
import numpy as np
from services.detection import detection_service

def run_benchmark(test_dir):
    print(f"[*] Running benchmark on {test_dir}...")
    
    results = {"real": {"correct": 0, "total": 0}, "fake": {"correct": 0, "total": 0}}
    
    for category in ["real", "fake"]:
        category_path = os.path.join(test_dir, category)
        if not os.path.exists(category_path):
            continue
            
        files = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        results[category]["total"] = len(files)
        
        for file in files:
            file_path = os.path.join(category_path, file)
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            
            # Use current detection service
            result = detection_service.predict(image_bytes)
            is_real_pred = result["is_real"]
            
            if category == "real" and is_real_pred:
                results[category]["correct"] += 1
            elif category == "fake" and not is_real_pred:
                results[category]["correct"] += 1
    
    total_samples = results["real"]["total"] + results["fake"]["total"]
    accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0
    
    # Calculate simple precision/recall manually for summary
    real_acc = (results["real"]["correct"] / results["real"]["total"] * 100) if results["real"]["total"] > 0 else 0
    fake_acc = (results["fake"]["correct"] / results["fake"]["total"] * 100) if results["fake"]["total"] > 0 else 0

    print("\n" + "="*40)
    print("      DEEPSHIELD AI BENCHMARK        ")
    print("="*40)
    print(f"Dataset:      {test_dir}")
    print(f"Total Images: {total_samples}")
    print("-" * 40)
    print(f"Real Accuracy: {results['real']['correct']}/{results['real']['total']} ({real_acc:.1f}%)")
    print(f"Fake Accuracy: {results['fake']['correct']}/{results['fake']['total']} ({fake_acc:.1f}%)")
    print("-" * 40)
    print(f"OVERALL ACCURACY: {accuracy:.2f}%")
    print("="*40 + "\n")
    
    return accuracy

if __name__ == "__main__":
    test_data_dir = "test_data"
    if len(sys.argv) > 1:
        test_data_dir = sys.argv[1]
        
    run_benchmark(test_data_dir)
