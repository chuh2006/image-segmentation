import cv2
from ultralytics import YOLO
import numpy as np

model = YOLO("models/yolov8x-seg.pt")
path = 'img-read/34f6247b527223eceda7c97205cc8f75.mp4'
cap = cv2.VideoCapture(path)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
output_path = 'output_video.mp4'
fourcc_h264 = cv2.VideoWriter_fourcc(*'H264')
out = cv2.VideoWriter(output_path, fourcc_h264, fps, (width, height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame)
    mask = np.zeros_like(frame)
    for result in results:
        masks = result.masks
        classes = result.boxes.cls.cpu().numpy().astype(int)
        if masks is not None:
            for mask_single, cls in zip(masks.data, classes):
                if cls == 0:  # 0 是 person 类别的索引
                    mask_single = mask_single.cpu().numpy().astype(np.uint8) * 255
                    mask_single_resized = cv2.resize(mask_single, (frame.shape[1], frame.shape[0]))
                    mask = cv2.add(mask, cv2.merge([mask_single_resized, mask_single_resized, mask_single_resized]))

    combined = cv2.bitwise_and(frame, frame, mask=mask[:, :, 0])
    out.write(combined)
    cv2.imshow('Segmentation', combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()