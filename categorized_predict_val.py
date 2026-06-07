from ultralytics import YOLO


def evaluate_on_cpu(model_path, data_config):
    model = YOLO(model_path).to('cpu')
    metrics = model.val(
        data=data_config,
        batch=4,
        imgsz=640,
        device='cpu',
        rect=True
    )

    # 修正后的指标获取方式
    print(f"\n✅ {data_config} 验证结果:")
    print(f"精确率 (单类别): {metrics.box.p[0]:.3f}")  # 单类别直接取索引0
    print(f"召回率 (单类别): {metrics.box.r[0]:.3f}")
    print(f"mAP50: {metrics.box.map50:.3f}")  # map50是全局标量
    print(f"mAP50-95: {metrics.box.map:.3f}")  # map是全局标量


if __name__ == "__main__":
    categories = ['small', 'medium', 'large']
    for cat in categories:
        yaml_path = f'./datasets/cvn/categorized/{cat}/data.yaml'
        print(f"\n🔍 正在验证 {cat} 目标...")
        evaluate_on_cpu('./原-best.pt', yaml_path)
