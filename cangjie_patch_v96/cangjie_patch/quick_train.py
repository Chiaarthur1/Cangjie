from ultralytics import RTDETR

model = RTDETR("rtdetr-x.pt")
model.train(
    data="./cangjie_yolo_dataset/data.yaml",
    epochs=25,
    batch=2,
    imgsz=320,
    device="cpu",
    val=False,
    lr0=0.001,
    project="cangjie_rtdetr_runs",
    name="overfit_test",
    exist_ok=True
)
