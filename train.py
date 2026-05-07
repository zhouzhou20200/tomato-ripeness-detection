from ultralytics import YOLO

if __name__ == '__main__':
    # 加载 YOLO11 模型（你搭的环境是YOLO11，用这个！）
    model = YOLO('yolo11n.pt')

    # 训练模型
    results = model.train(
        # 基础配置
        data='data/data.yaml',
        epochs=50,        # 新手足够（训练轮数）
        imgsz=640,

        # 数据加载（最稳定配置）
        workers=2,        # 笔记本必须2，不卡
        batch=8,          # 最稳
        cache=False,      # 笔记本容易爆内存，关掉更稳

        # 优化器（官方推荐 AdamW）
        optimizer='AdamW',
        lr0=0.001,       # AdamW 用 0.001 最稳
        lrf=0.01,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8,

        # 损失函数（默认最优，不用改）
        box=7.5,
        cls=0.5,
        dfl=1.5,
        label_smoothing=0.0,

        # 早停
        patience=10,      # 10轮没提升自动停，节省时间
        val=True,
        save_period=-1,

        # 输出
        project='runs/train',
        name='exp',
        exist_ok=False,
        plots=True
    )