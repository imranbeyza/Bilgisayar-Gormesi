from ultralytics import YOLO
import cv2
import os
import matplotlib.pyplot as plt

# YOLOv8 modelini yükle
model = YOLO('yolov8n.pt')

# Görsellerin bulunduğu klasör
image_folder = 'images'
output_folder = 'runs/detect'

# Tüm görsellerin listesini al
image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Her bir görsel için nesne tespiti yap ve sonuçları kaydet
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)

    # Nesne tespiti yap
    results = model(image_path)

    # Tespit edilen nesneleri görselleştir
    for result in results:  # Sonuçlar bir liste olarak döner
        result.show()  # Her bir sonucun gösterimi

    # Algılama sonuçlarını belirttiğimiz klasöre kaydet
    for result in results:
        result.save(save_dir=output_folder)

print(f"Algılama tamamlandı, sonuçlar {output_folder} klasörüne kaydedildi.")
