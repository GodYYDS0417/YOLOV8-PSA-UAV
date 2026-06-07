import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ========= 你要改的路径 =========
IMG_PATH = "./datasets/cvn/images/val/1x-7-_jpg.rf.1385effb6af1202dafaaacd8b106add4.jpg" # 你的测试图片
W1 = "./runs/detect/原/weights/best.pt" # 原始 YOLOv8 权重
W2 = "./runs/detect/PSA/weights/best.pt" # YOLOv8+PSA 权重
OUT_DIR = "./runs/detect/feature_compare" # 输出目录
IMG_SIZE = 640

# ========= 已经按你的模型结构改好 =========
TARGET_IDX_YOLOV8 = 21
TARGET_IDX_PSA = 22

TOPK = 6
os.makedirs(OUT_DIR, exist_ok=True)


def preprocess(img_path, img_size=640):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"图片不存在: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    img_float = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(img_float, (2, 0, 1))).unsqueeze(0)
    return rgb, tensor


def normalize_map(x):
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 1e-8:
        x = x / x.max()
    return x


def extract_feature(weight_path, target_idx, img_path):
    yolo = YOLO(weight_path)
    model = yolo.model.eval()
    device = next(model.parameters()).device

    _, tensor = preprocess(img_path, IMG_SIZE)
    tensor = tensor.to(device)

    feats = {}

    def hook_fn(module, inp, out):
        feats["x"] = out.detach().cpu()

    handle = model.model[target_idx].register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(tensor)

    handle.remove()

    feat = feats["x"][0]  # [C, H, W]
    return feat


def save_feature_fig(feat, save_prefix):
    # 平均特征图
    mean_map = normalize_map(feat.mean(0).numpy())

    plt.figure(figsize=(5, 5))
    plt.imshow(mean_map, cmap="jet")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_mean.jpg", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close()

    # Top-K 通道图
    channel_score = feat.view(feat.shape[0], -1).mean(1)
    topk_idx = torch.topk(channel_score, k=min(TOPK, feat.shape[0])).indices.tolist()

    ncols = 3
    nrows = (len(topk_idx) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, topk_idx):
        fmap = normalize_map(feat[idx].numpy())
        ax.imshow(fmap, cmap="jet")
        ax.set_title(f"ch={idx}")
        ax.axis("off")

    for ax in axes[len(topk_idx):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(f"{save_prefix}_topk.jpg", dpi=300)
    plt.close()


if __name__ == "__main__":
    feat1 = extract_feature(W1, TARGET_IDX_YOLOV8, IMG_PATH)
    feat2 = extract_feature(W2, TARGET_IDX_PSA, IMG_PATH)

    save_feature_fig(feat1, os.path.join(OUT_DIR, "yolov8"))
    save_feature_fig(feat2, os.path.join(OUT_DIR, "yolov8_psa"))

    print(f"特征图已保存到: {OUT_DIR}")