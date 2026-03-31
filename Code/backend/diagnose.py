"""
Diagnostic: Print raw signal values from all analysis components for each test image.
"""
import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os, sys

def ela_analysis(image_bytes, quality=90):
    nparr = np.frombuffer(image_bytes, np.uint8)
    original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, encoded = cv2.imencode('.jpg', original, encode_param)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = cv2.absdiff(original, recompressed).astype(np.float32)
    ela_mean = np.mean(diff)
    ela_std = np.std(diff)
    uniformity = ela_std / (ela_mean + 1e-8)
    return ela_mean, ela_std, uniformity

def freq_analysis(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256)).astype(np.float32)
    dct = cv2.dct(img)
    h, w = dct.shape
    ch, cw = h // 4, w // 4
    low = np.sum(np.abs(dct[:ch, :cw]) ** 2)
    total = np.sum(np.abs(dct) ** 2) + 1e-8
    return 1.0 - (low / total)

def color_analysis(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat_mean = np.mean(hsv[:, :, 1])
    sat_std = np.std(hsv[:, :, 1])
    val_std = np.std(hsv[:, :, 2])
    hue_hist, _ = np.histogram(hsv[:, :, 0], bins=36, range=(0, 180))
    hue_hist = hue_hist / (hue_hist.sum() + 1e-8)
    hue_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-8))
    return sat_mean, sat_std, val_std, hue_entropy

def diagnose(test_dir):
    model_id = "dima806/deepfake_vs_real_image_detection"
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id)
    model.eval()
    labels = model.config.id2label

    for cat in ["real", "fake"]:
        path = os.path.join(test_dir, cat)
        files = sorted([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        
        print(f"\n{'='*80}")
        print(f"  {cat.upper()} IMAGES")
        print(f"{'='*80}")
        
        for f in files:
            fp = os.path.join(path, f)
            with open(fp, "rb") as fh:
                img_bytes = fh.read()
            
            # Neural
            img = Image.open(fp).convert("RGB")
            inputs = processor(images=img, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = F.softmax(logits, dim=-1)[0]
            real_prob = probs[0].item()  # 0: Real
            
            # ELA
            ela_m, ela_s, ela_u = ela_analysis(img_bytes)
            
            # Freq
            hfr = freq_analysis(img_bytes)
            
            # Color
            sm, ss, vs, he = color_analysis(img_bytes)
            
            print(f"  {f}: neural_real={real_prob:.4f} | ela_mean={ela_m:.2f} ela_std={ela_s:.2f} ela_unif={ela_u:.3f} | freq_hf={hfr:.4f} | hue_ent={he:.3f} sat_std={ss:.1f}")

if __name__ == "__main__":
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "test_data")
