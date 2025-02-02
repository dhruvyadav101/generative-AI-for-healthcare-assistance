from flask import Flask, request, render_template, url_for
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Set upload folder for saving images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load your trained models (update paths as needed)
efficientnet_model = torch.load(r'C:\Users\death\Desktop\major project\project\best_efficientnet_model_entire.pth', map_location=device).to(device).eval()
mobilenet_model   = torch.load(r'C:\Users\death\Desktop\major project\project\best_mobilenet_model.pth', map_location=device).to(device).eval()
densenet_model    = torch.load(r'C:\Users\death\Desktop\major project\project\best_densenet_model.pth', map_location=device).to(device).eval()
resnet_model      = torch.load(r'C:\Users\death\Desktop\major project\project\best_resnet50_model.pth', map_location=device).to(device).eval()

# Group all models into a list
models = [efficientnet_model, mobilenet_model, densenet_model, resnet_model]

# Define image preprocessing transforms (must match training transforms)
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def ensemble_predict_single_image(models, image, device):
    """
    Processes a single image through the ensemble of models and returns
    the predicted class (using max confidence across models) and its confidence.
    """
    image_tensor = transform(image).unsqueeze(0).to(device)
    max_confidence = None
    final_pred = None

    with torch.no_grad():
        for model in models:
            outputs = torch.softmax(model(image_tensor), dim=1)
            confidence, pred = torch.max(outputs, dim=1)
            if max_confidence is None:
                max_confidence = confidence
                final_pred = pred
            else:
                if confidence > max_confidence:
                    max_confidence = confidence
                    final_pred = pred

    return final_pred.item(), max_confidence.item()

# Mapping from class index to disease names.
# When using ImageFolder with your TRAIN folder (alphabetical order):
#   0: BCC, 1: BKL, 2: MEL, 3: NV, 4: SCC
disease_labels = {
    0: "BCC",
    1: "BKL",
    2: "MEL",
    3: "NV",
    4: "SCC"
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('predict.html', error="No file part in the request.")
        file = request.files['image']
        if file.filename == '':
            return render_template('predict.html', error="No file selected.")
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image = Image.open(filepath).convert('RGB')
        except Exception as e:
            return render_template('predict.html', error="Invalid image file.")
        
        predicted_class, confidence = ensemble_predict_single_image(models, image, device)
        label = disease_labels.get(predicted_class, f"Class {predicted_class}")
        return render_template('result.html', label=label, confidence=confidence, filename=filename)
    
    return render_template('predict.html')

if __name__ == '__main__':
    app.run(debug=True)
