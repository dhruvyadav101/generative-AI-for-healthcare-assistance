from flask import Flask, request, render_template, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
import requests
import json
import logging
import random
import re
from flask import jsonify

# Initialize Flask app
app = Flask(__name__)

# Set upload folder for saving images
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 🔹 Hardcoded API Key
API_KEY = "-"

# Gemini API Configuration
MODEL_NAME = "models/gemini-1.5-pro"
GEMINI_API_URL = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

# Define logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Import required classes for model deserialization
from timm.models.efficientnet import EfficientNet
from torch.nn.modules.conv import Conv2d

# Register safe globals for deserialization
torch.serialization.add_safe_globals([EfficientNet, Conv2d])

# Load models
efficientnet_model = torch.load('efficientnet_model.pth', map_location=device, weights_only=False).to(device).eval()
mobilenet_model = torch.load('mobilenet_model.pth', map_location=device, weights_only=False).to(device).eval()
densenet_model  = torch.load('densenet_model.pth', map_location=device, weights_only=False).to(device).eval()
resnet_model    = torch.load('resnet_model.pth', map_location=device, weights_only=False).to(device).eval()

# Group all models for ensemble prediction
models = [efficientnet_model, mobilenet_model, densenet_model, resnet_model]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def ensemble_predict_max_confidence(models, image, device):
    """ Predicts disease using an ensemble of models based on max confidence. """
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
                # Update prediction if the current model has a higher confidence
                if confidence > max_confidence:
                    max_confidence = confidence
                    final_pred = pred

    return final_pred.item(), max_confidence.item()


# Disease Labels Mapping
disease_labels = {
    0: "Basal cell carcinoma",
    1: "Benign keratosis-like lesions",
    2: "Malignant melanoma",
    3: "Cannot detect the disease",
    4: "Melanocytic nevus",
    5: "Squamous cell carcinoma"
}

def get_disease_info(disease):
    """
    Fetches structured medical information about a disease using Gemini-1.5 Pro.
    """
    prompt = f"""
    Act as a medical expert. Provide structured information about {disease} in **strict JSON format**.
    ```json
    {{
        "Disease Name": "{disease}",
        "Description": "Brief overview of the disease",
        "Causes": "Main causes",
        "Symptoms": ["Symptom 1", "Symptom 2", "Symptom 3"],
        "Diagnosis": "How it is diagnosed",
        "Treatment": "Available treatments",
        "Prevention": ["Preventive measure 1", "Preventive measure 2"],
        "Severity Levels": "Different levels of severity",
        "Contagious": "Yes/No and how it spreads",
        "Complications": "Possible complications if untreated"
    }}
    ```
    Do not include citations or references. Just return a valid JSON response.
    """

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 1000}
    }

    try:
        print(f"\nDEBUG: Sending request to API: {GEMINI_API_URL}\n")  
        response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=data, headers=headers)

        print("\nDEBUG: API Response Code:", response.status_code)  
        print("DEBUG: Full API Response:", response.text[:500])  

        if response.status_code == 200:
            result = response.json()
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
            
            json_match = re.search(r"```json\n(.*?)\n```", raw_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))

            print("DEBUG: JSON not found in API response.")
        else:
            print(f"ERROR: API Request Failed - {response.status_code} - {response.text}")

    except Exception as e:
        print(f"ERROR: Exception during API request - {e}")

    return {"error": "API request failed. No data available."}

def get_disease_images(disease):
    """ Fetches random disease-related images from local storage. """
    DISEASE_IMAGE_FOLDER = "static/disease_images"

    # Folder Mapping for Disease Images
    FOLDER_MAPPING = {
        "Basal cell carcinoma": "Basal_cell_carcinoma",
        "Benign keratosis-like lesions": "benign_keratosis_like_lesions",
        "Malignant melanoma": "malignant_melanoma",
        "Melanocytic nevus": "Melanocytic_nevus",
        "Squamous cell carcinoma": "squamous_cell_carcinoma"
    }

    folder_name = FOLDER_MAPPING.get(disease, disease.replace(" ", "_"))
    disease_folder = os.path.join(DISEASE_IMAGE_FOLDER, folder_name)

    if os.path.exists(disease_folder) and os.path.isdir(disease_folder):
        images = [f for f in os.listdir(disease_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        random.shuffle(images)
        return images[:4]  
    return []  

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files.get('image')
        if not file or file.filename == '':
            return render_template('predict.html', error="No file selected.")
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        image = Image.open(filepath).convert('RGB')
        predicted_class, confidence = ensemble_predict_max_confidence(models, image, device)

        label = disease_labels.get(predicted_class, f"Class {predicted_class}")

        # ✅ If no disease is detected, do NOT call the API
        if label == "Cannot detect the disease":
            disease_info = {
                "Description": "No skin disease detected.",
                "Causes": "N/A",
                "Symptoms": ["N/A"],
                "Diagnosis": "N/A",
                "Treatment": "N/A",
                "Prevention": ["NA"],
                "Severity Levels": "N/A",
                "Contagious": "N/A",
                "Complications": "N/A"
            }
            disease_images = []  # No images needed for no disease case

        else:
            # 🔹 Call API only when a disease is detected
            disease_info = get_disease_info(label)
            disease_images = get_disease_images(label)

        # 🔹 Debugging Output
        print("\nDEBUG: Disease Info Retrieved from Gemini API or Predefined Data:")
        print(json.dumps(disease_info, indent=4))
        print("\nDEBUG: Retrieved Disease Images:", disease_images)

        return render_template('result.html', 
                               label=label, 
                               filename=filename,
                               disease_info=disease_info, 
                               disease_images=disease_images)

    return render_template('predict.html')
@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    if not user_input:
        return jsonify({"response": "Please enter a valid question."})

    prompt = f"You are a skin disease expert. Answer this query in around 20 to 30 words: {user_input}"

    headers = {"Content-Type": "application/json"}
    data = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.5, "maxOutputTokens": 500}
    }

    try:
        response = requests.post(f"{GEMINI_API_URL}?key={API_KEY}", json=data, headers=headers)
        if response.status_code == 200:
            result = response.json()
            reply = result["candidates"][0]["content"]["parts"][0]["text"]
            return jsonify({"response": reply.strip()})
        else:
            return jsonify({"response": "Failed to get response from Gemini API."})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})


if __name__ == '__main__':
    app.run(debug=True)
