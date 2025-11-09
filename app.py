# -----------------------------------------------------
# 🧠 Deepfake Image Detection Flask App (ResNet50 - PyTorch)
# -----------------------------------------------------

from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from PIL import Image
import torch, torchvision
import torch.nn as nn
from torchvision import models, transforms
import numpy as np

# -----------------------------------------------------
# 1️⃣ Flask App Configuration
# -----------------------------------------------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# -----------------------------------------------------
# 2️⃣ Load Model
# -----------------------------------------------------
MODEL_PATH = 'model.pkl'

# Define same architecture used in training
model = models.resnet50(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)  # Assuming 2 classes: Real / Fake

try:
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"⚠️ Failed to load model: {e}")

# -----------------------------------------------------
# 3️⃣ Image Transform (same as training)
# -----------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------------------------------
# 4️⃣ Routes
# -----------------------------------------------------
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error_message="No file uploaded!")

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error_message="Please select a file to upload!")

    # Save file
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Prediction
    try:
        image_tensor = transform(Image.open(filepath).convert("RGB")).unsqueeze(0)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            fake_prob = probs[0, 1].item()
            real_prob = probs[0, 0].item()
            pred_class = int(torch.argmax(probs))

        label = "Real" if pred_class == 0 else "Fake"
        prob = float(max(fake_prob, real_prob))
        color = "result-real" if label == "Real" else "result-fake"

        return render_template(
            'index.html',
            prediction_label=label,
            prediction_prob=prob,
            model_name="ResNet50 (PyTorch)",
            color=color
        )

    except Exception as e:
        return render_template('index.html', error_message=f"Prediction failed: {e}")

# -----------------------------------------------------
# 5️⃣ Run Flask App
# -----------------------------------------------------
if __name__ == '__main__':
    print("🚀 Starting Flask server on http://127.0.0.1:5000 ...")
    app.run(debug=True)
