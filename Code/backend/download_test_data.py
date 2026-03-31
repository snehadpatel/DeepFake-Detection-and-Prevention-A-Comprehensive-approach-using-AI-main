"""
Download test images for deepfake detection benchmark.
- Real faces: from public face image URLs (randomuser.me API)
- Fake faces: from thispersondoesnotexist.com (StyleGAN-generated)
"""
import os
import urllib.request
import shutil
import json

def download_test_images(output_dir, num_per_class=15):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    # Download REAL faces from randomuser.me (real photographs)
    print(f"[*] Downloading {num_per_class} real face photos...")
    url = f"https://randomuser.me/api/?results={num_per_class}&inc=picture"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode())
    
    for i, result in enumerate(data["results"]):
        img_url = result["picture"]["large"]
        save_path = os.path.join(output_dir, "real", f"real_{i}.jpg")
        urllib.request.urlretrieve(img_url, save_path)
    print(f"  [+] Downloaded {num_per_class} real faces")

    # Download FAKE faces from thispersondoesnotexist.com (StyleGAN)
    print(f"[*] Downloading {num_per_class} fake (AI-generated) faces...")
    for i in range(num_per_class):
        fake_url = f"https://thispersondoesnotexist.com"
        save_path = os.path.join(output_dir, "fake", f"fake_{i}.jpg")
        req = urllib.request.Request(fake_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp:
            with open(save_path, "wb") as f:
                f.write(resp.read())
    print(f"  [+] Downloaded {num_per_class} fake faces")

    print(f"\n[+] Test data ready in {output_dir}")

if __name__ == "__main__":
    download_test_images("test_data", num_per_class=15)
