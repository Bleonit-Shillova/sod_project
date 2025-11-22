import os
import torch
import torch.nn as nn
from tqdm import tqdm

from data_loader import create_dataloaders
from sod_model import SimpleUNet


# ---------- IoU metric ----------
def iou_score(pred, target, eps=1e-6):
    """
    pred, target: [B, 1, H, W] in [0,1]
    """
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    inter = (pred_bin * target_bin).sum(dim=(1, 2, 3))
    union = pred_bin.sum(dim=(1, 2, 3)) + target_bin.sum(dim=(1, 2, 3)) - inter + eps

    return (inter / union).mean()


def train_model(
    dataset_root,
    img_size=224,
    batch_size=8,
    num_epochs=20,                # 15–25 as required
    lr=1e-3,                      # Adam lr = 1e-3
    device=None,
    model_class=SimpleUNet,
    checkpoint_path="checkpoints/best_model_224.pth",
    resume=False,                 # bonus: resume training from checkpoint
):
    # ---- Device ----
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("🔥 Using CUDA GPU")
        else:
            device = torch.device("cpu")
            print("⚠ Using CPU")

    # ---- Data ----
    print("📂 Loading data...")
    train_loader, val_loader, _ = create_dataloaders(
        dataset_root, img_size=img_size, batch_size=batch_size
    )

    # ---- Model / loss / optimizer ----
    print("🧠 Building model...")
    model = model_class().to(device)
    bce = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    # ---- Checkpoints folder ----
    ckpt_dir = os.path.dirname(checkpoint_path) or "."
    os.makedirs(ckpt_dir, exist_ok=True)

    # Optional: resume feature
    start_epoch = 1
    best_val_loss = float("inf")
    last_ckpt_path = os.path.join(ckpt_dir, "last_checkpoint.pth")

    if resume and os.path.exists(last_ckpt_path):
        print(f"🔁 Resuming from {last_ckpt_path}")
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from epoch {ckpt['epoch']}")

    patience_counter = 0
    patience_limit = 5

    print("\n🚀 Starting training...\n")

    for epoch in range(start_epoch, num_epochs + 1):

        # ---------- TRAIN ----------
        model.train()
        train_loss = 0.0
        train_iou  = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} Training"):
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            preds = model(imgs)   # sigmoid output

            iou = iou_score(preds, masks)

            # Required loss: BCE + 0.5 * (1 - IoU)
            loss = bce(preds, masks) + 0.5 * (1.0 - iou)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_iou  += iou.item()

        train_loss /= len(train_loader)
        train_iou  /= len(train_loader)

        # ---------- VALIDATION ----------
        model.eval()
        val_loss = 0.0
        val_iou  = 0.0

        with torch.no_grad():
            for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} Validation"):
                imgs, masks = imgs.to(device), masks.to(device)

                preds = model(imgs)
                iou = iou_score(preds, masks)
                loss = bce(preds, masks) + 0.5 * (1.0 - iou)

                val_loss += loss.item()
                val_iou  += iou.item()

        val_loss /= len(val_loader)
        val_iou  /= len(val_loader)

        scheduler.step(val_loss)

        print(f"\nEpoch {epoch}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f} | Train IoU: {train_iou:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   IoU: {val_iou:.4f}")

        # ---------- Save "last" checkpoint (for resume) ----------
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, last_ckpt_path)

        # ---------- Save "best" model ----------
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"💾 Saved NEW BEST model to {checkpoint_path}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"⏳ No improvement: patience {patience_counter}/{patience_limit}")

        # ---------- Early stopping ----------
        if patience_counter >= patience_limit:
            print("\n⛔ Early stopping activated!")
            break

    print("\n🎉 Training completed!")
    return model
