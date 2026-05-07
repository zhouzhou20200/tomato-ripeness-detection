from ultralytics import YOLO
import cv2

# ===================== 你的模型路径 =====================
model = YOLO("runs/detect/runs/train/exp-3/weights/best.pt")

# ===================== 要检测的图片 =====================
# 把这里换成你要测试的番茄图片路径！
img_path = "test.jpg"

# ===================== 开始检测 =====================
results = model.predict(
    source=img_path,
    conf=0.55,       # 最佳置信度（我从你曲线里算出来的最优值）
    save=True,       # 自动保存检测后的图片
    show=True        # 自动弹出窗口显示结果
)

# 打印结果
print("\n===== 检测完成！=====")
for r in results:
    print(f"图片: {r.path}")
    print(f"检测到 {len(r.boxes)} 个番茄")
    for box in r.boxes:
        cls = model.names[int(box.cls)]
        conf = float(box.conf)
        print(f"→ {cls}  置信度: {conf:.2f}")