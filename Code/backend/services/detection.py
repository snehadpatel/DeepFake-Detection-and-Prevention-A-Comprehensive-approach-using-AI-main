import torch
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import io

class DetectionService:
    def __init__(self):
        print("[*] Loading DeepShield AI Detection Engine...")
        
        # Neural model: dima806 ViT — good at structural analysis
        self.model_id = "dima806/deepfake_vs_real_image_detection"
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModelForImageClassification.from_pretrained(self.model_id)
        self.model.eval()
        self.labels = self.model.config.id2label
        print(f"  [+] ViT Model loaded: {self.labels}")
        print("[+] DeepShield AI Engine ready.")

    def _ela_analysis(self, image_bytes, quality=90):
        """
        Error Level Analysis: Detect compression inconsistencies.
        GAN-generated images have uniform ELA patterns (never been JPEG compressed),
        while real photos have variable ELA (from camera processing / resaving).
        """
        # Decode original
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None:
            return 0.5
        
        # Re-compress at a specific quality
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', original, encode_param)
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Calculate difference (Error Level)
        diff = cv2.absdiff(original, recompressed).astype(np.float32)
        
        # ELA statistics
        ela_mean = np.mean(diff)
        ela_std = np.std(diff)
        ela_max = np.max(diff)
        
        # Uniformity metric: GAN images tend to have very uniform (low std) ELA
        # Real images have more variation
        uniformity = ela_std / (ela_mean + 1e-8)
        
        return ela_mean, ela_std, uniformity
    
    def _frequency_analysis(self, image_bytes):
        """
        Frequency domain analysis using DCT.
        GAN-generated images often have distinctive frequency patterns:
        - Smoother high-frequency components
        - Periodic artifacts in the frequency spectrum
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.5
            
        img = cv2.resize(img, (256, 256)).astype(np.float32)
        
        # 2D DCT
        dct = cv2.dct(img)
        
        # Analyze frequency distribution
        # High-frequency energy ratio
        h, w = dct.shape
        center_h, center_w = h // 4, w // 4
        
        # Low-frequency energy (center quarter)
        low_freq = np.sum(np.abs(dct[:center_h, :center_w]) ** 2)
        # Total energy  
        total = np.sum(np.abs(dct) ** 2) + 1e-8
        # High-frequency ratio
        high_freq_ratio = 1.0 - (low_freq / total)
        
        return high_freq_ratio
    
    def _color_analysis(self, image_bytes):
        """
        Color space analysis. GAN images often show:
        - Over-smooth skin tones
        - Unusual color distribution in HSV space
        - Lower color variance compared to real camera photos
        """
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return 0.5
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Saturation statistics
        sat_mean = np.mean(hsv[:, :, 1])
        sat_std = np.std(hsv[:, :, 1])
        
        # Value (brightness) statistics
        val_std = np.std(hsv[:, :, 2])
        
        # Hue distribution entropy
        hue_hist, _ = np.histogram(hsv[:, :, 0], bins=36, range=(0, 180))
        hue_hist = hue_hist / (hue_hist.sum() + 1e-8)
        hue_entropy = -np.sum(hue_hist * np.log(hue_hist + 1e-8))
        
        return sat_mean, sat_std, val_std, hue_entropy

    def preprocess(self, image_bytes):
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image

    def get_heatmap(self, image_bytes, res=224):
        """Generate ELA-based heatmap showing suspicious regions."""
        nparr = np.frombuffer(image_bytes, np.uint8)
        original = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original is None:
            return np.zeros((res, res, 3), dtype=np.uint8)
        
        # Re-compress
        _, encoded = cv2.imencode('.jpg', original, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # ELA difference amplified
        diff = cv2.absdiff(original, recompressed).astype(np.float32)
        diff = np.mean(diff, axis=2)  # Convert to grayscale
        diff = diff * 20  # Amplify
        diff = np.clip(diff, 0, 255).astype(np.uint8)
        
        # Resize and colormap
        diff_resized = cv2.resize(diff, (res, res))
        heatmap = cv2.applyColorMap(diff_resized, cv2.COLORMAP_JET)
        
        return heatmap

    def predict(self, image_bytes):
        pil_image = self.preprocess(image_bytes)
        
        # === Component 1: Neural Model (ViT) ===
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = F.softmax(logits, dim=-1)[0]
        
        real_idx = None
        for idx, label in self.labels.items():
            if label.lower() == "real":
                real_idx = idx
                break
        neural_real_prob = probs[real_idx].item() if real_idx is not None else 0.5
        
        # === Component 2: ELA Analysis ===
        ela_mean, ela_std, ela_uniformity = self._ela_analysis(image_bytes)
        # From diagnostic: Real uniformity ≥ 0.997, Fake ≤ 1.012
        # ELA uniformity is a continuous score — map linearly
        ela_real_score = min(1.0, max(0.0, (ela_uniformity - 0.90) / (1.10 - 0.90)))
        
        # === Component 3: Frequency Analysis ===
        high_freq_ratio = self._frequency_analysis(image_bytes)
        # From diagnostic: Real < 0.003, Fake > 0.003 (fakes have MORE high-freq from GAN artifacts)
        # Counter-intuitive: GAN images have MORE high-frequency energy, not less
        if high_freq_ratio < 0.002:
            freq_real_score = 0.9
        elif high_freq_ratio < 0.003:
            freq_real_score = 0.6
        elif high_freq_ratio < 0.005:
            freq_real_score = 0.3
        else:
            freq_real_score = 0.1
            
        # === Component 4: Color Analysis ===
        sat_mean, sat_std, val_std, hue_entropy = self._color_analysis(image_bytes)
        # Hue entropy: not strongly separating, use as a weak signal
        color_real_score = 0.5  # Neutral — not reliable enough
        
        # === Ensemble Decision ===
        # Weighted: Neural 30%, ELA 30%, Frequency 30%, Color 10%
        # ELA and Frequency are strongest signals from the diagnostic
        ensemble_real_prob = (
            0.30 * neural_real_prob +
            0.30 * ela_real_score +
            0.30 * freq_real_score +
            0.10 * color_real_score
        )
        
        is_real = ensemble_real_prob > 0.5
        confidence = ensemble_real_prob if is_real else (1.0 - ensemble_real_prob)
        
        # Generate heatmap
        heatmap = self.get_heatmap(image_bytes)
        
        # Prepare images for response
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        img_resized = cv2.resize(img_cv, (224, 224))
        superimposed_img = cv2.addWeighted(img_resized, 0.6, heatmap, 0.4, 0)
        
        _, buffer_orig = cv2.imencode('.jpg', img_resized)
        _, buffer_heat = cv2.imencode('.jpg', superimposed_img)
        
        return {
            "is_real": bool(is_real),
            "confidence": float(confidence),
            "heatmap": buffer_heat.tobytes(),
            "original": buffer_orig.tobytes()
        }

detection_service = DetectionService()
