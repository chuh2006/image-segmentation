import cv2
from ultralytics import YOLO
import numpy as np
from matplotlib import pyplot as plt

model = YOLO("models/yolo11x-seg.pt")
path = 'img-read/haonan2.jpg'
img = cv2.imread(path)
if img is None:
    print('Image not found')
    exit()

results = model(img)
mask = np.zeros_like(img)
class_names = model.names
img_out = []

for result in results:
    masks = result.masks
    classes = result.boxes.cls.cpu().numpy().astype(int)
    labels = [class_names[cls] for cls in classes]
    for mask_single, cls in zip(masks.data, classes):
        mask_single = mask_single.cpu().numpy().astype(np.uint8) * 255
        mask_single_resized = cv2.resize(mask_single, (img.shape[1], img.shape[0]))
        contours, _ = cv2.findContours(mask_single_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
        WandH = (w + h) // 500
        if WandH % 2 == 0:
            WandH += 1
        mask_single_resized = cv2.GaussianBlur(cv2.morphologyEx(mask_single_resized, cv2.MORPH_ERODE, np.ones((WandH, WandH), np.uint8)), (WandH, WandH), 0)
        mask_single_resized = cv2.threshold(mask_single_resized, 152, 255, cv2.THRESH_BINARY)[1]
        mask = cv2.merge([mask_single_resized, mask_single_resized, mask_single_resized])
        img_out.append(cv2.bitwise_and(img, img, mask=mask[:, :, 0]))

num_images = len(img_out)
num_cols = 5
num_rows = (num_images + num_cols - 1) // num_cols

plt.figure(figsize=(15, 5 * num_rows))
plt.subplot(num_rows, num_cols + 1, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Original')

for index, img in enumerate(img_out):
    plt.subplot(num_rows + 1, num_cols, index + 2)
    contours, _ = cv2.findContours(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
    roi = img[y:y + h, x:x + w]
    plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.title(labels[index])

plt.tight_layout()
plt.show()

for index, img in enumerate(img_out):
    cv2.imwrite(f'./img-save/output_{index}.jpg', img)
