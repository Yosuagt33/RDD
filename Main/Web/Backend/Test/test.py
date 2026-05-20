from ultralytics import YOLO
import cv2  
model_path = r"C:\Users\ASUS\Documents\Final-Project\Main\DataSet\RDD.pt"
image_path = r"C:\Users\ASUS\Documents\Final-Project\Main\image\tes\image1.png" 

model = YOLO(model_path)

results = model.predict(source=image_path, conf=0.1, save=False, show=False)

annotated_frame = results[0].plot()

cv2.imshow("Hasil Deteksi AI", annotated_frame)

detected_objects = len(results[0].boxes)
print(f"AI berhasil mendeteksi {detected_objects} objek.")
print("Membuka gambar... Tekan tombol APA SAJA di keyboard untuk menutup jendela gambar.")

cv2.waitKey(0) 

cv2.destroyAllWindows()