import cv2
import numpy as np

# Webcam'i aç (0, varsayılan kamerayı temsil eder)
cap = cv2.VideoCapture(0)

while True:
    # Web kameradan bir kare oku
    ret, frame = cap.read()
    
    if not ret:
        print("Kamera açılamadı!")
        break
    
    # Ortalama filtre uygula
    blurred = cv2.blur(frame, (5, 5))
    
    # Laplace filtresi uygula
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Filtrelenmiş görüntülerin tipini uygun hale getir
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Orijinal ve filtrelenmiş görüntüleri yan yana birleştir
    combined = np.hstack((frame, blurred, laplacian))
    
    # Görüntüyü ekranda göster
    cv2.imshow("Orijinal | Ortalama Filtre | Laplace Filtre", combined)
    
    # 'Esc' tuşuna basıldığında çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()
