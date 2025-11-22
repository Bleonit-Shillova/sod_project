import torch
import matplotlib.pyplot as plt
from data_loader import create_dataloaders
from sod_model import SimpleUNet


def compute_metrics(pred, target, eps=1e-6):
    """
    pred, target: [B, 1, H, W]
    Returns IoU, Precision, Recall, F1, MAE
    """
    pred_bin = (pred > 0.5).float()
    target_bin = (target > 0.5).float()

    tp = (pred_bin * target_bin).sum()
    fp = (pred_bin * (1 - target_bin)).sum()
    fn = ((1 - pred_bin) * target_bin).sum()

    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    iou       = tp / (tp + fp + fn + eps)

    mae       = (pred - target).abs().mean()

    return (
        iou.item(),
        precision.item(),
        recall.item(),
        f1.item(),
        mae.item()
    )


def evaluate_model(dataset_root, checkpoint_path, model_class=SimpleUNet,
                   img_size=224, batch_size=4, device="cuda"):

    _, _, test_loader = create_dataloaders(dataset_root, img_size, batch_size)

    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_scores = []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            all_scores.append(compute_metrics(preds, masks))

    scores = torch.tensor(all_scores)  # [N, 5]
    print("\n====== TEST METRICS ======")
    print("IoU      :", scores[:,0].mean().item())
    print("Precision:", scores[:,1].mean().item())
    print("Recall   :", scores[:,2].mean().item())
    print("F1       :", scores[:,3].mean().item())
    print("MAE      :", scores[:,4].mean().item())


def show_examples(dataset_root, checkpoint_path, model_class=SimpleUNet,
                  img_size=224, batch_size=4, device="cuda", num_batches=1):

    _, _, test_loader = create_dataloaders(dataset_root, img_size, batch_size)

    model = model_class().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    with torch.no_grad():
        for b, (images, masks) in enumerate(test_loader):
            if b >= num_batches:
                break

            images = images.to(device)
            masks  = masks.to(device)
            preds  = model(images)

            imgs_np = images.cpu().numpy()
            gt_np   = masks.cpu().numpy()
            pr_np   = preds.cpu().numpy()

            for i in range(len(imgs_np)):
                img = imgs_np[i].transpose(1, 2, 0)
                gt  = gt_np[i, 0]
                pr  = pr_np[i, 0]

                plt.figure(figsize=(12, 4))
                plt.subplot(1,4,1); plt.imshow(img); plt.title("Input");      plt.axis("off")
                plt.subplot(1,4,2); plt.imshow(gt, cmap="gray"); plt.title("GT Mask");   plt.axis("off")
                plt.subplot(1,4,3); plt.imshow(pr, cmap="gray"); plt.title("Prediction");plt.axis("off")
                plt.subplot(1,4,4)
                plt.imshow(img)
                plt.imshow(pr, cmap="jet", alpha=0.4)
                plt.title("Overlay")
                plt.axis("off")
                plt.show()
