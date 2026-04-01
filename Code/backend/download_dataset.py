#!/usr/bin/env python3
"""
Download Dataset for Deepfake Detection Fine-Tuning
====================================================
- Real faces: randomuser.me (real photographs)
- Fake faces: thispersondoesnotexist.com (StyleGAN-generated)
All images normalized to 256x256 JPEG to prevent resolution-based bias.
"""
import os, json, time, ssl, shutil, sys
import urllib.request
from PIL import Image
from io import BytesIO

DATASET_DIR = "dataset"
NUM_PER_CLASS = 200  # 200 real + 200 fake = 400 total

def download_real_faces(output_dir, count):
    """Download real face photos from randomuser.me API."""
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0
    batch_size = 100

    while downloaded < count:
        batch = min(batch_size, count - downloaded)
        url = f"https://randomuser.me/api/?results={batch}&inc=picture&seed=finetune_v2_{downloaded}"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            for result in data["results"]:
                img_url = result["picture"]["large"]
                save_path = os.path.join(output_dir, f"real_{downloaded}.jpg")
                # Download and normalize size
                r = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(r, timeout=10) as img_resp:
                    img = Image.open(BytesIO(img_resp.read())).convert("RGB")
                    img = img.resize((256, 256), Image.LANCZOS)
                    img.save(save_path, "JPEG", quality=85)
                downloaded += 1
            print(f"  [+] Downloaded {downloaded}/{count} real faces")
        except Exception as e:
            print(f"  [!] Error: {e}, retrying...")
            time.sleep(2)
    return downloaded

def download_fake_faces(output_dir, count):
    """Download StyleGAN-generated fake faces."""
    os.makedirs(output_dir, exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    downloaded = 0
    attempts = 0
    while downloaded < count and attempts < count * 3:
        attempts += 1
        try:
            req = urllib.request.Request(
                "https://thispersondoesnotexist.com",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = resp.read()
                if len(data) > 5000:
                    img = Image.open(BytesIO(data)).convert("RGB")
                    img = img.resize((256, 256), Image.LANCZOS)
                    save_path = os.path.join(output_dir, f"fake_{downloaded}.jpg")
                    img.save(save_path, "JPEG", quality=85)
                    downloaded += 1
                    if downloaded % 25 == 0 or downloaded == count:
                        print(f"  [+] Downloaded {downloaded}/{count} fake faces")
            time.sleep(0.4)
        except Exception as e:
            print(f"  [!] Retry {attempts}: {e}")
            time.sleep(1)
    return downloaded

def main():
    print(f"[*] Creating dataset with {NUM_PER_CLASS} images per class...")
    if os.path.exists(DATASET_DIR):
        existing_real = len([f for f in os.listdir(os.path.join(DATASET_DIR, "real")) if f.endswith('.jpg')]) if os.path.exists(os.path.join(DATASET_DIR, "real")) else 0
        existing_fake = len([f for f in os.listdir(os.path.join(DATASET_DIR, "fake")) if f.endswith('.jpg')]) if os.path.exists(os.path.join(DATASET_DIR, "fake")) else 0
        if existing_real >= NUM_PER_CLASS and existing_fake >= NUM_PER_CLASS:
            print(f"[+] Dataset already exists ({existing_real} real, {existing_fake} fake). Skipping download.")
            return
        shutil.rmtree(DATASET_DIR)

    print(f"\n[*] Downloading {NUM_PER_CLASS} real face photos...")
    real_count = download_real_faces(os.path.join(DATASET_DIR, "real"), NUM_PER_CLASS)

    print(f"\n[*] Downloading {NUM_PER_CLASS} fake (AI-generated) faces...")
    fake_count = download_fake_faces(os.path.join(DATASET_DIR, "fake"), NUM_PER_CLASS)

    print(f"\n{'='*50}")
    print(f"  Real images: {real_count}")
    print(f"  Fake images: {fake_count}")
    print(f"  Total:       {real_count + fake_count}")
    print(f"  Location:    {os.path.abspath(DATASET_DIR)}")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()
