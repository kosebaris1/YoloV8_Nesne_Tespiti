import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QPushButton, 
                             QFileDialog, QVBoxLayout, QHBoxLayout, QWidget, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class YoloApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv8 Nesne Tespiti - USD/TL Sayacı")
        self.setGeometry(100, 100, 1200, 700) 

        try:
            print("Model yükleniyor...")
            self.model = YOLO("best.pt")
            print("Model başarıyla yüklendi!")
        except Exception as e:
            QMessageBox.critical(self, "Hata", f"Model yüklenirken hata oluştu!\n'best.pt' dosyasının bu klasörde olduğundan emin olun.\nHata: {str(e)}")

        self.image_path = None 
        self.tagged_image = None 
        self.initUI()

    def initUI(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout()

        image_layout = QHBoxLayout()

        self.lbl_original = QLabel("Original Image Paneli\n(Resim Seçilmedi)")
        self.lbl_original.setAlignment(Qt.AlignCenter)
        self.lbl_original.setStyleSheet("border: 2px dashed gray; background-color: #e8e8e8; font-weight: bold;")
        self.lbl_original.setFixedSize(550, 500)
        
        self.lbl_tagged = QLabel("Tagged Image Paneli\n(Sonuç Burada Görünecek)")
        self.lbl_tagged.setAlignment(Qt.AlignCenter)
        self.lbl_tagged.setStyleSheet("border: 2px dashed gray; background-color: #e8e8e8; font-weight: bold;")
        self.lbl_tagged.setFixedSize(550, 500)

        image_layout.addWidget(self.lbl_original)
        image_layout.addWidget(self.lbl_tagged)
        main_layout.addLayout(image_layout)

        btn_layout = QHBoxLayout()

        self.btn_select = QPushButton("Select Image (Resim Seç)")
        self.btn_select.setMinimumHeight(40)
        self.btn_select.setStyleSheet("font-size: 14px; background-color: #3498db; color: white;")
        self.btn_select.clicked.connect(self.select_image)

        self.btn_test = QPushButton("Test Image (Tespiti Başlat)")
        self.btn_test.setMinimumHeight(40)
        self.btn_test.setStyleSheet("font-size: 14px; background-color: #27ae60; color: white;")
        self.btn_test.clicked.connect(self.detect_objects)
        
        self.btn_save = QPushButton("Save Image (Kaydet)")
        self.btn_save.setMinimumHeight(40)
        self.btn_save.setStyleSheet("font-size: 14px; background-color: #e67e22; color: white;")
        self.btn_save.clicked.connect(self.save_image)
        self.btn_save.setEnabled(False) 

        btn_layout.addWidget(self.btn_select)
        btn_layout.addWidget(self.btn_test)
        btn_layout.addWidget(self.btn_save)
        
        main_layout.addLayout(btn_layout)
        
        self.lbl_result = QLabel("Sonuçlar: Henüz işlem yapılmadı.")
        self.lbl_result.setStyleSheet("font-size: 16px; font-weight: bold; color: #333; margin-top: 10px;")
        self.lbl_result.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.lbl_result)

        central_widget.setLayout(main_layout)

    def select_image(self):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(self, "Resim Seç", "", "Resim Dosyaları (*.jpg *.png *.jpeg);;Tüm Dosyalar (*)", options=options)
        
        if file_path:
            self.image_path = file_path
            
            pixmap = QPixmap(file_path)
            self.lbl_original.setPixmap(pixmap.scaled(self.lbl_original.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
            self.lbl_tagged.clear()
            self.lbl_tagged.setText("Tagged Image Paneli")
            self.lbl_result.setText("Sonuçlar: Resim yüklendi, tespiti başlatabilirsiniz.")
            self.btn_save.setEnabled(False)

    def detect_objects(self):
        if not self.image_path:
            QMessageBox.warning(self, "Uyarı", "Lütfen önce bir resim seçin!")
            return

        results = self.model(self.image_path)
        result = results[0]

        self.tagged_image = result.plot() 
        
        rgb_image = cv2.cvtColor(self.tagged_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        self.lbl_tagged.setPixmap(pixmap.scaled(self.lbl_tagged.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.btn_save.setEnabled(True)

        class_counts = {}
        names = result.names # Modelin bildiği sınıf isimleri (USD, TL vb.)
        
        for box in result.boxes:
            cls_id = int(box.cls[0]) # Sınıf ID'si (0 veya 1)
            cls_name = names[cls_id] # Sınıf Adı (USD veya TL)
            
            if cls_name in class_counts:
                class_counts[cls_name] += 1
            else:
                class_counts[cls_name] = 1
        
        text_res = "TESPİT SONUÇLARI:  "
        if not class_counts:
            text_res += "Nesne bulunamadı."
        else:
            for name, count in class_counts.items():
                text_res += f"[{name}: {count} adet]   "
        
        self.lbl_result.setText(text_res)
        QMessageBox.information(self, "İşlem Tamam", "Nesne tespiti tamamlandı!\n" + text_res)

    def save_image(self):
        if self.tagged_image is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Resmi Kaydet", "sonuc.jpg", "JPG Files (*.jpg);;PNG Files (*.png)")
            if file_path:
                cv2.imwrite(file_path, self.tagged_image)
                QMessageBox.information(self, "Başarılı", "Görüntü başarıyla kaydedildi.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YoloApp()
    window.show()
    sys.exit(app.exec_())