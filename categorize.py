import os
import glob
import numpy as np
from PIL import Image
import shutil


def classify_drone_images(image_dir, label_dir, output_base_dir):
    # 创建带子目录的输出结构
    output_dirs = {
        'small': {
            'images': os.path.join(output_base_dir, 'small/images'),
            'labels': os.path.join(output_base_dir, 'small/labels')
        },
        'medium': {
            'images': os.path.join(output_base_dir, 'medium/images'),
            'labels': os.path.join(output_base_dir, 'medium/labels')
        },
        'large': {
            'images': os.path.join(output_base_dir, 'large/images'),
            'labels': os.path.join(output_base_dir, 'large/labels')
        }
    }

    # 创建所有目录
    for category in output_dirs.values():
        os.makedirs(category['images'], exist_ok=True)
        os.makedirs(category['labels'], exist_ok=True)

    # 保存图片尺寸和最大面积
    image_sizes = {}
    max_areas = []

    # 第一次遍历：收集元数据
    label_files = list(glob.glob(os.path.join(label_dir, '*.txt')))
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_name = f"{base_name}.jpg"
        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            print(f"警告：图片文件缺失 {image_path}")
            continue

        # 读取图片尺寸
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
                image_sizes[base_name] = (img_width, img_height)
        except Exception as e:
            print(f"图片读取错误 {image_path}: {e}")
            continue

        # 计算最大目标面积
        max_area = 0
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    _, _, _, w_norm, h_norm = map(float, parts)
                except ValueError:
                    continue

                width = w_norm * img_width
                height = h_norm * img_height
                area = width * height
                max_area = max(max_area, area)

        if max_area > 0:
            max_areas.append(max_area)

    if not max_areas:
        print("错误：未找到有效标注数据")
        return

    # 计算动态阈值
    small_thresh = np.percentile(max_areas, 33)
    medium_thresh = np.percentile(max_areas, 66)
    print(f"分类阈值 - 小: {small_thresh:.2f}, 中: {medium_thresh:.2f}, 大: ≥{medium_thresh:.2f}")

    # 第二次遍历：分类文件
    for label_file in label_files:
        base_name = os.path.splitext(os.path.basename(label_file))[0]
        image_name = f"{base_name}.jpg"
        image_path = os.path.join(image_dir, image_name)

        if base_name not in image_sizes:
            continue

        img_w, img_h = image_sizes[base_name]
        current_max = 0

        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                try:
                    _, _, _, w_norm, h_norm = map(float, parts)
                except ValueError:
                    continue

                width = w_norm * img_w
                height = h_norm * img_h
                current_max = max(current_max, width * height)

        # 确定分类
        if current_max < small_thresh:
            category = 'small'
        elif current_max < medium_thresh:
            category = 'medium'
        else:
            category = 'large'

        # 复制图片和标签
        try:
            # 复制图片
            img_dest = os.path.join(output_dirs[category]['images'], image_name)
            shutil.copy2(image_path, img_dest)

            # 复制标签
            label_name = f"{base_name}.txt"
            label_src = os.path.join(label_dir, label_name)
            label_dest = os.path.join(output_dirs[category]['labels'], label_name)
            shutil.copy2(label_src, label_dest)

            print(f"已分类: {base_name} -> {category}")
        except Exception as e:
            print(f"文件复制失败 {base_name}: {e}")


if __name__ == "__main__":
    classify_drone_images(
        image_dir="./datasets/cvn/images/val",
        label_dir="./datasets/cvn/labels/val",
        output_base_dir="./datasets/cvn/categorized",
    )
