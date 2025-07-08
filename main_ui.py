import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTextEdit, QGridLayout, QFrame, QDialog, QScrollArea
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# ---- Model and Class Setup ----
model_directory = './Model'
classes = ['bkl', 'bcc', 'akiec', 'vasc', 'nv', 'mel', 'df']
class_names = {
    'bkl': 'Benign keratosis-like lesions (Tổn thương giống dày sừng lành tính)',
    'bcc': 'Basal cell carcinoma (Ung thư biểu mô tế bào đáy)',
    'akiec': 'Actinic keratoses and intraepithelial carcinoma (Dày sừng quang hóa và ung thư biểu mô tại chỗ)',
    'vasc': 'Vascular lesions (Tổn thương mạch máu)',
    'nv': 'Melanocytic nevi (Nốt ruồi sắc tố)',
    'mel': 'Melanoma (Ung thư hắc tố)',
    'df': 'Dermatofibroma (U xơ da)'
}
danger_levels = {
    'bkl': 'Lành tính',
    'bcc': 'Ung thư nhẹ',
    'akiec': 'Tiền ung thư',
    'vasc': 'Ít nguy hiểm',
    'nv': 'Lành tính',
    'mel': 'Rất nguy hiểm',
    'df': 'Lành tính'
}

# Load model
processor = ViTImageProcessor.from_pretrained(model_directory)
model = ViTForImageClassification.from_pretrained(model_directory)
model.eval()
if torch.cuda.is_available():
    model.to("cuda")

def predict_single_image(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    predicted_probability = torch.max(probabilities).item()
    predicted_class_idx = torch.argmax(logits, dim=1).item()
    predicted_class = classes[predicted_class_idx]

    return predicted_class, predicted_probability

# ---- Bảng chi tiết loại bệnh ----
class DiseaseInfoDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧾 Thông tin chi tiết các loại bệnh")
        self.setMinimumSize(600, 500)

        layout = QVBoxLayout()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)

        content = QWidget()
        content_layout = QVBoxLayout()

        font = QFont("Arial", 14)
        for label in classes:
            section = QLabel()
            section.setText(
                f"<b>{label.upper()}</b>:<br>"
                f"{class_names[label]}<br>"
                f"<span style='color:#b30000;'>⚠️ Mức độ nguy hiểm:</span> <b>{danger_levels[label]}</b><br><br>"
            )
            section.setFont(font)
            section.setWordWrap(True)
            content_layout.addWidget(section)

        content.setLayout(content_layout)
        scroll.setWidget(content)

        layout.addWidget(scroll)
        self.setLayout(layout)

# ---- GUI Application ----
class SkinCancerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Skin Cancer Diagnosis")
        self.setGeometry(300, 100, 800, 600)
        self.setAcceptDrops(True)

        font_title = QFont("Arial", 16, QFont.Bold)
        font_text = QFont("Arial", 14)

        self.image_label = QLabel("Thả ảnh vào đây.")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(font_text)
        self.image_label.setFrameShape(QFrame.Box)
        self.image_label.setFixedSize(300, 300)
        self.image_label.setStyleSheet("background-color: #f0f0f0;")

        self.result_label = QLabel("Kết quả: Chưa có")
        self.result_label.setFont(font_text)
        self.confidence_label = QLabel("Độ tin cậy: -")
        self.confidence_label.setFont(font_text)
        self.risk_label = QLabel("Mức độ nguy hiểm: -")
        self.risk_label.setFont(font_text)

        self.select_button = QPushButton("Chọn ảnh từ máy")
        self.select_button.setFont(font_text)
        self.select_button.clicked.connect(self.select_image)

        self.detail_button = QPushButton("Thông tin loại bệnh")
        self.detail_button.setFont(font_text)
        self.detail_button.clicked.connect(self.show_disease_info)

        layout = QGridLayout()
        layout.addWidget(self.image_label, 0, 0, 4, 1)
        layout.addWidget(self.result_label, 0, 1)
        layout.addWidget(self.confidence_label, 1, 1)
        layout.addWidget(self.risk_label, 2, 1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.detail_button)

        main_layout = QVBoxLayout()
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)

        self.setLayout(main_layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.process_image(file_path)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        pixmap = QPixmap(file_path).scaled(300, 300, Qt.KeepAspectRatio)
        self.image_label.setPixmap(pixmap)

        try:
            predicted_class, confidence = predict_single_image(file_path)
            self.result_label.setText(f"Kết quả: {class_names[predicted_class]}")
            self.confidence_label.setText(f"Độ tin cậy: {confidence:.2%}")
            self.risk_label.setText(f"Mức độ nguy hiểm: {danger_levels[predicted_class]}")
        except Exception as e:
            self.result_label.setText("Lỗi khi dự đoán.")
            print("Lỗi:", e)

    def show_disease_info(self):
        dialog = DiseaseInfoDialog()
        dialog.exec_()

# ---- Run the App ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = SkinCancerApp()
    window.show()
    sys.exit(app.exec_())
