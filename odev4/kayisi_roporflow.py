from roboflow import Roboflow
import os
from ultralytics import YOLO

# Roboflow API anahtarınızı burada tanımlayın
rf = Roboflow(api_key="70lG0WE73JsUzlqHe003")

# Çalışma alanınızı ve projenizi tanımlayın
project = rf.workspace("imranbeyza").project("kayisi_roboflow")
version = project.version(1)
dataset = version.download("yolov8")  # YOLOv8 formatında dataset indirildi


model = YOLO('C:/Users/Acer/OneDrive/Masaüstü/odev4/models/best.pt')  # Eğitilen modelin yolu


image_folder = r"C:\Users\Acer\OneDrive\Masaüstü\odev4\images"  # Görsellerin bulunduğu yol
results_folder = "results"  # Sonuçların kaydedileceği klasör
os.makedirs(results_folder, exist_ok=True)

# Görsellerde tahmin yap ve kaydet
for idx, image_file in enumerate(os.listdir(image_folder), start=1):
    if image_file.endswith(".jpg") or image_file.endswith(".png"):  # Yalnızca .jpg ve .png dosyalarını işleme
        image_path = os.path.join(image_folder, image_file)

        # Tahmin yapma
        results = model.predict(source=image_path, conf=0.45, iou=0.45)

        # Tahmin edilen görüntüyü kaydet
        results[0].save(os.path.join(results_folder, f"prediction_{idx}.jpg"))
        print(f"Processed {image_file} - Results saved as prediction_{idx}.jpg in 'results' folder")
