# Gerekli kütüphaneleri içe aktarın
from ultralytics import YOLO
import cv2
import requests
import numpy as np
import matplotlib.pyplot as plt

# YOLOv8 modelini yükleyin
model = YOLO('yolov8n.pt')  # YOLOv8 Nano modeli, hızlı analiz için

# Webden görüntüyü almak için bir fonksiyon
def load_image_from_url(url):
    response = requests.get(url)
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return img

# 10 adet görüntü URL'sini listeye ekleyin
image_urls = [
    "https://i.pinimg.com/236x/93/11/df/9311dfe0d86c225a7b9c03fdac6bb7b6.jpg",
    "https://i.pinimg.com/236x/bf/bf/20/bfbf204a01a2c6cdacc06bc49bfb6d0d.jpg",
    "https://i.pinimg.com/236x/f1/f0/d2/f1f0d2fff9d83c7b0007d0133eb5cdd6.jpg",
    "https://i.pinimg.com/236x/95/a0/27/95a027e77decbd5efa594d5cf23edde6.jpg",
    "https://i.pinimg.com/236x/8d/11/eb/8d11ebbf98152113dcab3a64494f8a0c.jpg",
    "https://i.pinimg.com/236x/05/5a/6d/055a6ddbdf5aa5467a58ce8addd81b62.jpg",
    "https://i.pinimg.com/236x/8a/6a/b2/8a6ab2e6f9b3b809f0194ac7642209bd.jpg",
    "https://i.pinimg.com/236x/5c/da/60/5cda60dd60d1045d52874100a1656048.jpg",
    "https://i.pinimg.com/236x/42/0a/f6/420af6de58543bc42856e0c95a2fa190.jpg",
    "https://i.pinimg.com/236x/7d/64/84/7d6484c2e0e64b7b7818a5277f8afc0b.jpg"
]

# Görselleştirme için 2x5'lik bir alt-üst düzen hazırlayın
fig, axes = plt.subplots(2, 5, figsize=(20, 10))

# Her görüntüyü indir, analiz et ve sonuçları alt-üst şekilde göster
for i, image_url in enumerate(image_urls):
    img = load_image_from_url(image_url)  # Görüntüyü indir
    results = model(img)  # Görüntüyü model ile analiz et

    print(f"Görüntü {i+1}:")
    # Tespit edilen nesneleri kontrol edin
    for detection in results[0].boxes:
        class_id = int(detection.cls)
        class_name = model.names[class_id]
        if class_name == "person":
            print("İnsan tespit edildi.")
        else:
            print(f"Diğer nesne tespit edildi: {class_name}")

    # Tespit edilen nesneleri görüntü üzerinde göster
    annotated_img = results[0].plot()
    
    # Alt grafiklere görüntüyü ekle
    row, col = divmod(i, 5)  # Satır ve sütun numarasını hesapla
    axes[row, col].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[row, col].axis('off')
    axes[row, col].set_title(f"Görüntü {i+1}")

plt.tight_layout()
plt.show()
