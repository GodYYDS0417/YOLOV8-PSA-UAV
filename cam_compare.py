import os
import cv2
import torch
import numpy as np
import torch.nn as nn
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ========= 你要改的路径 =========
IMG_PATH = "./datasets/cvn/images/val/1x-7-_jpg.rf.1385effb6af1202dafaaacd8b106add4.jpg" # 你的测试图片
W1 = "./runs/detect/原/weights/best.pt" # 原始 YOLOv8 权重
W2 = "./runs/detect/PSA/weights/best.pt" # YOLOv8+PSA 权重
OUT_DIR = "./runs/detect/cam" # 输出目录
IMG_SIZE = 640
# ========= 你的层号 =========
TARGET_IDX_YOLOV8 = 21
TARGET_IDX_PSA = 22

os.makedirs(OUT_DIR, exist_ok=True)


class YOLOWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        return out


def load_img(img_path, img_size=640):
    bgr = cv2.imread(img_path)
    if bgr is None:
        raise FileNotFoundError(f"图片不存在: {img_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size))
    rgb_float = rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(np.transpose(rgb_float, (2, 0, 1))).unsqueeze(0)
    return rgb, rgb_float, tensor


def make_cam(weight_path, target_idx, img_path, save_path):
    yolo = YOLO(weight_path)
    raw_model = yolo.model.eval()
    model = YOLOWrapper(raw_model).eval()

    device = next(raw_model.parameters()).device
    rgb, rgb_float, input_tensor = load_img(img_path, IMG_SIZE)
    input_tensor = input_tensor.to(device)
    model = model.to(device)

    target_layer = model.model.model[target_idx]

    with EigenCAM(model=model, target_layers=[target_layer]) as cam:
        grayscale_cam = cam(input_tensor=input_tensor)[0]

    cam_img = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    cv2.imwrite(save_path, cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    make_cam(W1, TARGET_IDX_YOLOV8, IMG_PATH, os.path.join(OUT_DIR, "yolov8_cam.jpg"))
    make_cam(W2, TARGET_IDX_PSA, IMG_PATH, os.path.join(OUT_DIR, "yolov8_psa_cam.jpg"))
    print(f"热力图已保存到: {OUT_DIR}")