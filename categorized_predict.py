from ultralytics import YOLO

# 加载训练好的模型（替换为你的.pt路径）
model = YOLO('./best.pt')

# 执行预测（替换为你的数据路径）
results = model.predict(
    source='./ultralytics/assets/示例4.jpg',  # 支持图片/视频/目录
    save=True,          # 保存带标注结果
    conf=0.5,           # 置信度阈值
    show_labels=True,   # 显示标签
    show_boxes=True     # 显示方框
)