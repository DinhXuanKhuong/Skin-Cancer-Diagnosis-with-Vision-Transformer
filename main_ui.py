import sys
import torch
from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGridLayout, QFrame, QDialog, QScrollArea
)
from PyQt5.QtGui import QPixmap, QFont, QPalette, QColor
from PyQt5.QtCore import Qt
from PIL import Image
from transformers import ViTImageProcessor, ViTForImageClassification

# ---- Model and Class Setup ----
model_directory = './Model'
classes = ['bkl', 'bcc', 'akiec', 'vasc', 'nv', 'mel', 'df']
class_names_vietnamese = {
    'bkl': 'Tổn thương giống dày sừng lành tính',
    'bcc': 'Ung thư biểu mô tế bào đáy',
    'akiec': 'Dày sừng quang hóa và ung thư biểu mô tại chỗ',
    'vasc': 'Tổn thương mạch máu',
    'nv': 'Nốt ruồi sắc tố',
    'mel': 'Ung thư hắc tố',
    'df': 'U xơ da'
}
class_names_english = {
    'bkl': 'Benign keratosis-like lesions',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses and intraepithelial carcinoma',
    'vasc': 'Vascular lesions',
    'nv': 'Melanocytic nevi',
    'mel': 'Melanoma',
    'df': 'Dermatofibroma'
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
danger_levels_english = {
    'bkl': 'Benign',
    'bcc': 'Mild cancer',
    'akiec': 'Pre-cancerous',
    'vasc': 'Low risk',
    'nv': 'Benign',
    'mel': 'Very dangerous',
    'df': 'Benign'
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
    def __init__(self, is_english=False):
        super().__init__()
        self.setWindowTitle("🧾 Disease Information" if is_english else "🧾 Thông tin chi tiết các loại bệnh")
        self.setFixedSize(650, 550)
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
                border: 1px solid #d0d0d0;
                border-radius: 10px;
            }
            QLabel {
                color: #333333;
                padding: 5px;
            }
        """)

        layout = QVBoxLayout()
        layout.setContentsMargins(15, 15, 15, 15)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: #f8f9fa;
                border-radius: 8px;
            }
            QScrollBar:vertical {
                border: none;
                background: #f0f0f0;
                width: 10px;
                margin: 0px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #888888;
                border-radius: 5px;
            }
        """)

        content = QWidget()
        content_layout = QVBoxLayout()
        content_layout.setSpacing(10)

        font = QFont("Segoe UI", 12)
        for label in classes:
            section = QLabel()
            section.setText(
                f"<b>{label.upper()}</b>:<br>"
                f"{class_names_english[label]}<br>"
                f"<span style='color:#d32f2f;'>⚠️ Danger Level:</span> <b>{danger_levels_english[label]}</b><br>" if is_english else
                f"<b>{label.upper()}</b>:<br>"
                f"{class_names_vietnamese[label]}<br>"
                f"<span style='color:#d32f2f;'>⚠️ Mức độ nguy hiểm:</span> <b>{danger_levels[label]}</b><br>"
            )
            section.setFont(font)
            section.setWordWrap(True)
            section.setStyleSheet("padding: 10px; background-color: #ffffff; border-radius: 5px;")
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
        self.setFixedSize(900, 450)
        self.setAcceptDrops(True)
        self.setStyleSheet("""
            QWidget {
                background-color: #f5f7fa;
                font-family: 'Segoe UI';
            }
        """)
        self.is_english = False

        # Fonts
        font_title = QFont("Segoe UI", 14, QFont.Bold)
        font_text = QFont("Segoe UI", 12)

        # Image Label
        self.image_label = QLabel("Drop an image here or click 'Select Image'" if self.is_english else "Thả ảnh vào đây hoặc nhấn 'Chọn ảnh'")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFont(font_text)
        self.image_label.setFixedSize(350, 350)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px dashed #cccccc;
                border-radius: 10px;
                background-color: #ffffff;
                color: #666666;
            }
            QLabel:hover {
                border: 2px dashed #3b82f6;
            }
        """)

        # Result Labels with Word Wrap
        self.result_label = QLabel("Result: Not available" if self.is_english else "Kết quả: Chưa có")
        self.result_label.setFont(font_text)
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet("color: #333333; padding: 10px;")
        self.confidence_label = QLabel("Confidence: -" if self.is_english else "Độ tin cậy: -")
        self.confidence_label.setFont(font_text)
        self.confidence_label.setWordWrap(True)
        self.confidence_label.setStyleSheet("color: #333333; padding: 10px;")
        self.risk_label = QLabel("Danger Level: -" if self.is_english else "Mức độ nguy hiểm: -")
        self.risk_label.setFont(font_text)
        self.risk_label.setWordWrap(True)
        self.risk_label.setStyleSheet("color: #333333; padding: 10px;")

        # Buttons
        self.select_button = QPushButton("Select Image" if self.is_english else "Chọn ảnh từ máy")
        self.select_button.setFont(font_text)
        self.select_button.setFixedHeight(40)
        self.select_button.setStyleSheet("""
            QPushButton {
                background-color: #3b82f6;
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #2563eb;
            }
            QPushButton:pressed {
                background-color: #1e40af;
            }
        """)
        self.select_button.clicked.connect(self.select_image)

        self.detail_button = QPushButton("Disease Info" if self.is_english else "Thông tin loại bệnh")
        self.detail_button.setFont(font_text)
        self.detail_button.setFixedHeight(40)
        self.detail_button.setStyleSheet("""
            QPushButton {
                background-color: #10b981;
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #059669;
            }
            QPushButton:pressed {
                background-color: #047857;
            }
        """)
        self.detail_button.clicked.connect(self.show_disease_info)

        self.lang_button = QPushButton("Tiếng Anh (English)")
        self.lang_button.setFont(font_text)
        self.lang_button.setFixedHeight(40)
        self.lang_button.setStyleSheet("""
            QPushButton {
                background-color: #f59e0b;
                color: white;
                border-radius: 8px;
                padding: 8px;
            }
            QPushButton:hover {
                background-color: #d97706;
            }
            QPushButton:pressed {
                background-color: #b45309;
            }
        """)
        self.lang_button.clicked.connect(self.toggle_language)

        # Layout
        layout = QGridLayout()
        layout.setSpacing(15)
        layout.addWidget(self.image_label, 0, 0, 4, 1, Qt.AlignCenter)
        layout.addWidget(self.result_label, 0, 1)
        layout.addWidget(self.confidence_label, 1, 1)
        layout.addWidget(self.risk_label, 2, 1)

        button_layout = QHBoxLayout()
        button_layout.setSpacing(10)
        button_layout.addWidget(self.select_button)
        button_layout.addWidget(self.detail_button)
        button_layout.addWidget(self.lang_button)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        main_layout.addLayout(layout)
        main_layout.addLayout(button_layout)
        main_layout.addStretch()

        self.setLayout(main_layout)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        for url in event.mimeData().urls():
            file_path = url.toLocalFile()
            self.process_image(file_path)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Image" if self.is_english else "Chọn ảnh", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_path:
            self.process_image(file_path)

    def process_image(self, file_path):
        pixmap = QPixmap(file_path).scaled(350, 350, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(pixmap)
        self.image_label.setText("")  # Clear placeholder text when image is loaded

        try:
            predicted_class, confidence = predict_single_image(file_path)
            if self.is_english:
                self.result_label.setText(f"Result: {class_names_english[predicted_class].split(' (')[0]}")
                self.confidence_label.setText(f"Confidence: {confidence:.2%}")
                self.risk_label.setText(f"Danger Level: {danger_levels_english[predicted_class]}")
            else:
                self.result_label.setText(f"Kết quả: {class_names_vietnamese[predicted_class]}")
                self.confidence_label.setText(f"Độ tin cậy: {confidence:.2%}")
                self.risk_label.setText(f"Mức độ nguy hiểm: {danger_levels[predicted_class]}")
        except Exception as e:
            self.result_label.setText("Error in prediction." if self.is_english else "Lỗi khi dự đoán.")
            print("Lỗi:", e)

    def show_disease_info(self):
        dialog = DiseaseInfoDialog(self.is_english)
        dialog.exec_()

    def toggle_language(self):
        self.is_english = not self.is_english
        self.image_label.setText("Drop an image here or click 'Select Image'" if self.is_english else "Thả ảnh vào đây hoặc nhấn 'Chọn ảnh'")
        self.result_label.setText("Result: Not available" if self.is_english else "Kết quả: Chưa có")
        self.confidence_label.setText("Confidence: -" if self.is_english else "Độ tin cậy: -")
        self.risk_label.setText("Danger Level: -" if self.is_english else "Mức độ nguy hiểm: -")
        self.select_button.setText("Select Image" if self.is_english else "Chọn ảnh từ máy")
        self.detail_button.setText("Disease Info" if self.is_english else "Thông tin loại bệnh")
        self.lang_button.setText("Vietnamese (Tiếng Việt)" if self.is_english else "Tiếng Anh (English)")

# ---- Run the App ----
if __name__ == "__main__":
    app = QApplication(sys.argv)
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor("#f5f7fa"))
    app.setPalette(palette)
    app.setStyle("Fusion")
    window = SkinCancerApp()
    window.show()
    sys.exit(app.exec_())