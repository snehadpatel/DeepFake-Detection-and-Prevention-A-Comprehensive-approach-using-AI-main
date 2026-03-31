"""
Download a SEPARATE validation set (unseen data) for blind evaluation.
"""
import os
import shutil
import json
import time
import ssl
import urllib.request

def download_validation_images(output_dir, num_per_class=20):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    os.makedirs(os.path.join(output_dir, "real"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "fake"), exist_ok=True)

    # SSL context that doesn't verify (for thispersondoesnotexist)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    # REAL faces
    print(f"[*] Downloading {num_per_class} UNSEEN real face photos...")
    url = f"https://randomuser.me/api/?results={num_per_class}&inc=picture&seed=blind_val_2026"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, context=ctx, timeout=10) as resp:
        data = json.loads(resp.read().decode())
    
    for i, result in enumerate(data["results"]):
        img_url = result["picture"]["large"]
        r = urllib.request.Request(img_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(r, context=ctx, timeout=10) as resp:
            with open(os.path.join(output_dir, "real", f"real_{i}.jpg"), "wb") as f:
                f.write(resp.read())
    print(f"  [+] Downloaded {num_per_class} real faces")

    # FAKE faces
    print(f"[*] Downloading {num_per_class} UNSEEN fake faces...")
    downloaded = 0
    attempts = 0
    while downloaded < num_per_class and attempts < num_per_class * 3:
        attempts += 1
        try:
            req = urllib.request.Request(
                "https://thispersondoesnotexist.com",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=15) as resp:
                data = resp.read()
                if len(data) > 1000:  # Valid image
                    with open(os.path.join(output_dir, "fake", f"fake_{downloaded}.jpg"), "wb") as f:
                        f.write(data)
                    downloaded += 1
            time.sleep(0.5)
        except Exception as e:
            print(f"  [!] Retry {attempts}: {e}")
            time.sleep(1)
    
    print(f"  [+] Downloaded {downloaded} fake faces")
    print(f"\n[+] Validation data ready in {output_dir}")

if __name__ == "__main__":
    download_validation_images("validation_data", num_per_class=20)
