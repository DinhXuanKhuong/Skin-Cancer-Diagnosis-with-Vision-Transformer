import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor


model_directory = './Model'


processor = ViTImageProcessor.from_pretrained(model_directory)
model = ViTForImageClassification.from_pretrained(model_directory)

model.eval()

if torch.cuda.is_available():
    model.to('cuda')
    print("Mô hình đã được chuyển sang GPU.")
else:
    print("Không tìm thấy GPU, mô hình sẽ chạy trên CPU.")

id2label = model.config.id2label


classes = ['bkl', 'bcc', 'akiec', 'vasc', 'nv', 'mel', 'df']
class_names = {
    'bkl': 'Benign keratosis-like lesions (Vết sẹo lành tính giống bệnh sừng)',
    'bcc': 'Basal cell carcinoma (Ung thư biểu mô tế bào đáy)',
    'akiec': 'Actinic keratoses (Sừng hóa do ánh sáng)',
    'vasc': 'Vascular lesions (Vết tổn thương mạch máu)',
    'nv': 'Melanocytic nevi (Nốt ruồi sắc tố)',
    'mel': 'Melanoma (Ung thư hắc tố)',
    'df': 'Dermatofibroma (U xơ da)'
}

def predict_single_image(image_path):

    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    if torch.cuda.is_available():
        inputs = {k: v.to('cuda') for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)

    predicted_probability = torch.max(probabilities).item()
    predicted_class_idx = torch.argmax(logits, dim=1).item()

    predicted_label = class_names[classes[predicted_class_idx]]

    return predicted_label, predicted_probability

test_image_path = 'Test/bkl.jpg' 

try:
    predicted_label, confidence = predict_single_image(test_image_path)
    print(f"\nẢnh: {test_image_path}")
    print(f"Dự đoán: {predicted_label}")
    print(f"Độ tự tin: {confidence:.4f}")
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy tệp ảnh tại đường dẫn '{test_image_path}'. Vui lòng kiểm tra lại đường dẫn.")
except Exception as e:
    print(f"Đã xảy ra lỗi khi dự đoán: {e}")