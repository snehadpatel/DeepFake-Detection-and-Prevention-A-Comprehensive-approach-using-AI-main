#!/usr/bin/env python3
"""
Fine-Tune Deepfake Detection Model
===================================
Complete pipeline: HP search → Full training → Evaluation → Plots

Outputs (in models/):
  best_model.pt, training_history.json, hyperparameter_results.json,
  accuracy_curves.png, loss_curves.png, confusion_matrix.png,
  classification_report.txt
"""
import os, sys, json, time, random, warnings
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

# === Config ===
DATASET_DIR = "dataset"
MODEL_DIR = "models"
BASE_MODEL = "dima806/deepfake_vs_real_image_detection"
SEED = 42
MAX_EPOCHS = 15
PATIENCE = 3

# Device: MPS (Apple Metal) > CUDA > CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"[*] Device: {DEVICE}")
print(f"[*] Base model: {BASE_MODEL}")

# === Dataset ===
class DeepfakeDataset(Dataset):
    def __init__(self, paths, labels, processor, augment=False):
        self.paths, self.labels, self.processor = paths, labels, processor
        self.aug = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.GaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0)),
        ]) if augment else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224))
        if self.aug:
            img = self.aug(img)
        px = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        return px, self.labels[idx]

def load_paths(dataset_dir):
    paths, labels = [], []
    for f in sorted(os.listdir(os.path.join(dataset_dir, "real"))):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            paths.append(os.path.join(dataset_dir, "real", f))
            labels.append(1)
    for f in sorted(os.listdir(os.path.join(dataset_dir, "fake"))):
        if f.lower().endswith(('.jpg', '.jpeg', '.png')):
            paths.append(os.path.join(dataset_dir, "fake", f))
            labels.append(0)
    return paths, labels

# === Training ===
def train_epoch(model, loader, optim, crit, dev):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    for imgs, labs in loader:
        imgs, labs = imgs.to(dev), labs.to(dev)
        optim.zero_grad()
        out = model(pixel_values=imgs).logits
        loss = crit(out, labs)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        loss_sum += loss.item() * imgs.size(0)
        correct += (out.argmax(1) == labs).sum().item()
        total += labs.size(0)
    return loss_sum / total, correct / total

def evaluate(model, loader, crit, dev):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    preds, trues = [], []
    with torch.no_grad():
        for imgs, labs in loader:
            imgs, labs = imgs.to(dev), labs.to(dev)
            out = model(pixel_values=imgs).logits
            loss = crit(out, labs)
            loss_sum += loss.item() * imgs.size(0)
            pred = out.argmax(1)
            correct += (pred == labs).sum().item()
            total += labs.size(0)
            preds.extend(pred.cpu().numpy())
            trues.extend(labs.cpu().numpy())
    return loss_sum / total, correct / total, np.array(preds), np.array(trues)

def freeze_layers(model, unfreeze_last_n=2):
    for p in model.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    if hasattr(model, 'vit'):
        layers = model.vit.encoder.layer
        for i in range(max(0, len(layers) - unfreeze_last_n), len(layers)):
            for p in layers[i].parameters():
                p.requires_grad = True
        for p in model.vit.layernorm.parameters():
            p.requires_grad = True
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# === HP Search ===
def hp_search(processor, tr_p, tr_l, v_p, v_l):
    print("\n" + "="*60)
    print("PHASE 1: Hyperparameter Search (3 epochs each)")
    print("="*60)
    configs = [
        {"lr": 5e-5, "wd": 0.01}, {"lr": 2e-5, "wd": 0.01},
        {"lr": 1e-5, "wd": 0.01}, {"lr": 2e-5, "wd": 0.1},
    ]
    results = []
    for i, cfg in enumerate(configs):
        print(f"\n--- Config {i+1}/{len(configs)}: lr={cfg['lr']}, wd={cfg['wd']} ---")
        tr_ds = DeepfakeDataset(tr_p, tr_l, processor, augment=True)
        v_ds = DeepfakeDataset(v_p, v_l, processor, augment=False)
        tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=0)
        v_dl = DataLoader(v_ds, batch_size=16, shuffle=False, num_workers=0)
        
        model = AutoModelForImageClassification.from_pretrained(BASE_MODEL)
        freeze_layers(model, 2)
        model = model.to(DEVICE)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=cfg['lr'], weight_decay=cfg['wd'])
        crit = nn.CrossEntropyLoss(label_smoothing=0.1)
        best_va = 0
        for ep in range(3):
            tl, ta = train_epoch(model, tr_dl, opt, crit, DEVICE)
            vl, va, _, _ = evaluate(model, v_dl, crit, DEVICE)
            best_va = max(best_va, va)
            print(f"  Ep {ep+1}: Train={ta:.4f}, Val={va:.4f}")
        results.append({"config": cfg, "best_val_acc": best_va})
        del model
    
    best = max(results, key=lambda x: x["best_val_acc"])
    print(f"\n[*] Best: lr={best['config']['lr']}, wd={best['config']['wd']} → {best['best_val_acc']:.4f}")
    return best["config"], results

# === Full Training ===
def full_train(processor, tr_p, tr_l, v_p, v_l, te_p, te_l, cfg):
    print("\n" + "="*60)
    print(f"PHASE 2: Full Training (lr={cfg['lr']}, wd={cfg['wd']})")
    print("="*60)
    os.makedirs(MODEL_DIR, exist_ok=True)

    tr_ds = DeepfakeDataset(tr_p, tr_l, processor, augment=True)
    v_ds = DeepfakeDataset(v_p, v_l, processor, augment=False)
    te_ds = DeepfakeDataset(te_p, te_l, processor, augment=False)
    tr_dl = DataLoader(tr_ds, batch_size=16, shuffle=True, num_workers=0)
    v_dl = DataLoader(v_ds, batch_size=16, shuffle=False, num_workers=0)
    te_dl = DataLoader(te_ds, batch_size=16, shuffle=False, num_workers=0)

    model = AutoModelForImageClassification.from_pretrained(BASE_MODEL)
    freeze_layers(model, 2)
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg['lr'], weight_decay=cfg['wd'])
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_va, patience_ctr, best_ep = 0, 0, 0

    print(f"\n{'Ep':>4} | {'TrLoss':>8} | {'TrAcc':>8} | {'VaLoss':>8} | {'VaAcc':>8} | Time")
    print("-" * 62)

    for ep in range(MAX_EPOCHS):
        t0 = time.time()
        tl, ta = train_epoch(model, tr_dl, opt, crit, DEVICE)
        vl, va, _, _ = evaluate(model, v_dl, crit, DEVICE)
        dt = time.time() - t0

        history["train_loss"].append(tl)
        history["train_acc"].append(ta)
        history["val_loss"].append(vl)
        history["val_acc"].append(va)

        mark = ""
        if va > best_va:
            best_va, best_ep, patience_ctr = va, ep+1, 0
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "best_model.pt"))
            mark = " ★ saved"
        else:
            patience_ctr += 1

        print(f"{ep+1:>4} | {tl:>8.4f} | {ta:>8.4f} | {vl:>8.4f} | {va:>8.4f} | {dt:.1f}s{mark}")

        if patience_ctr >= PATIENCE:
            print(f"\n[*] Early stopping at epoch {ep+1}")
            break

    print(f"\n[*] Best: Epoch {best_ep}, Val Acc: {best_va:.4f}")

    # Load best and evaluate
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "best_model.pt"), map_location=DEVICE))
    _, fin_tr, _, _ = evaluate(model, tr_dl, crit, DEVICE)
    _, fin_va, _, _ = evaluate(model, v_dl, crit, DEVICE)
    _, fin_te, te_pred, te_true = evaluate(model, te_dl, crit, DEVICE)

    # === Results ===
    print("\n" + "="*60)
    print("PHASE 3: Final Results")
    print("="*60)
    print(f"  TRAIN Accuracy:      {fin_tr*100:.2f}%")
    print(f"  VALIDATION Accuracy: {fin_va*100:.2f}%")
    print(f"  TEST Accuracy:       {fin_te*100:.2f}%")
    gap = fin_tr - fin_va
    if gap > 0.10:
        print(f"\n  ⚠️  OVERFITTING: Train-Val gap = {gap*100:.1f}%")
    elif fin_tr < 0.70:
        print(f"\n  ⚠️  UNDERFITTING: Train acc only {fin_tr*100:.1f}%")
    else:
        print(f"\n  ✅ Well-fit! Train-Val gap = {gap*100:.1f}%")

    report = classification_report(te_true, te_pred, target_names=["Fake", "Real"])
    print(f"\n{report}")

    with open(os.path.join(MODEL_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Train: {fin_tr*100:.2f}%\nVal: {fin_va*100:.2f}%\nTest: {fin_te*100:.2f}%\n")
        f.write(f"Gap: {gap*100:.1f}%\n\n{report}")

    return history, te_pred, te_true, fin_tr, fin_va, fin_te

# === Plotting ===
def save_plots(history, te_pred, te_true):
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_acc"], 'b-o', label='Train', ms=4)
    ax.plot(epochs, history["val_acc"], 'r-o', label='Validation', ms=4)
    ax.set(xlabel='Epoch', ylabel='Accuracy', title='Train vs Validation Accuracy')
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(f"{MODEL_DIR}/accuracy_curves.png", dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, history["train_loss"], 'b-o', label='Train', ms=4)
    ax.plot(epochs, history["val_loss"], 'r-o', label='Validation', ms=4)
    ax.set(xlabel='Epoch', ylabel='Loss', title='Train vs Validation Loss')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(f"{MODEL_DIR}/loss_curves.png", dpi=150); plt.close()

    cm = confusion_matrix(te_true, te_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(cm, display_labels=["Fake", "Real"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix (Test Set)")
    plt.tight_layout(); plt.savefig(f"{MODEL_DIR}/confusion_matrix.png", dpi=150); plt.close()
    print("[+] Saved: accuracy_curves.png, loss_curves.png, confusion_matrix.png")

# === Main ===
def main():
    t0 = time.time()
    paths, labels = load_paths(DATASET_DIR)
    print(f"\n[*] Dataset: {len(paths)} images ({sum(l==1 for l in labels)} real, {sum(l==0 for l in labels)} fake)")

    # 70/15/15 split
    tr_p, temp_p, tr_l, temp_l = train_test_split(paths, labels, test_size=0.3, random_state=SEED, stratify=labels)
    v_p, te_p, v_l, te_l = train_test_split(temp_p, temp_l, test_size=0.5, random_state=SEED, stratify=temp_l)
    print(f"  Train: {len(tr_p)}, Val: {len(v_p)}, Test: {len(te_p)}")

    processor = AutoImageProcessor.from_pretrained(BASE_MODEL)

    # Phase 1: HP Search
    best_cfg, hp_results = hp_search(processor, tr_p, tr_l, v_p, v_l)
    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/hyperparameter_results.json", "w") as f:
        json.dump([{"lr": r["config"]["lr"], "wd": r["config"]["wd"],
                     "val_acc": r["best_val_acc"]} for r in hp_results], f, indent=2)

    # Phase 2: Full Training
    history, te_pred, te_true, tr_acc, va_acc, te_acc = full_train(
        processor, tr_p, tr_l, v_p, v_l, te_p, te_l, best_cfg)
    with open(f"{MODEL_DIR}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Phase 3: Plots
    save_plots(history, te_pred, te_true)

    elapsed = time.time() - t0
    print(f"\n[*] Total time: {elapsed/60:.1f} minutes")
    print(f"[*] Model saved: {os.path.abspath(MODEL_DIR)}/best_model.pt")

if __name__ == "__main__":
    main()
