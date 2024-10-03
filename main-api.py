from typing import Union
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import requests
from PIL import Image
import timm
import torch
from torchvision import transforms
from io import BytesIO
import torch.nn.functional as F

app = FastAPI()

# Global variables
model = None
transform = None
class_dataset = []
confidence_cutoff = 80

def init():
    global model, transform, class_dataset
    
    NUM_CLASSES = 10
    model = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load('model\image_classification_checkpoint.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set the model to evaluation mode

    image_size = 288
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),  # Resize the image
        transforms.ToTensor(),                          # Convert image to PyTorch tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
    ])
    class_dataset = list(checkpoint['label'].keys())

init()  # Initialize the model and transformations

@app.get("/")
def read_root():
    return {"welcome": "model image prediction"}

@app.post("/predict_url/")
def predict_from_url(image_url: str = Form(...)):
    try:
        # Ensure the image URL is provided
        if not image_url:
            raise HTTPException(status_code=400, detail="image_url is required")

         # Fetch the image from the provided URL
        response = requests.get(image_url)
        response.raise_for_status()  # Raise an error for bad responses

        # Open the image
        image = Image.open(BytesIO(response.content))

        # Apply the transformation
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            outputs = model(image_tensor)

        tenser_out = F.softmax(outputs, dim=1)
        normalized_output = tenser_out * 100
        predicted_label_index = torch.argmax(normalized_output, dim=1).item()
        confidence_percent = normalized_output[0][predicted_label_index].item()
        confidence_percent = round(confidence_percent, 2)

        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs.data, 1)

        # Get the predicted class label
        predicted_label = class_dataset[predicted.item()]
        
        if confidence_percent > confidence_cutoff:

            return {
                "predicted_class": predicted_label,
                "confidence_percent" : confidence_percent
                }
        else:
            predicted_label = "Not in Available Classes"
            return {
                "predicted_class": predicted_label,
                "confidence_percent" : confidence_percent
                }
        
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for predicting from an uploaded image file
@app.post("/predict_file/")
def predict_from_file(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)

        # Apply the transformation
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

        # Forward pass
        with torch.no_grad():
            outputs = model(image_tensor)

        tenser_out = F.softmax(outputs, dim=1)
        normalized_output = tenser_out * 100
        predicted_label_index = torch.argmax(normalized_output, dim=1).item()
        confidence_percent = normalized_output[0][predicted_label_index].item()
        confidence_percent = round(confidence_percent, 2)

        # Get the predicted class with the highest score
        _, predicted = torch.max(outputs.data, 1)

        # Get the predicted class label
        predicted_label = class_dataset[predicted.item()]

        if confidence_percent > confidence_cutoff:

            return {
                    "predicted_class": predicted_label,
                    "confidence_percent" : confidence_percent
                    }
        else:
            predicted_label = "Not in Available Classes"
            return {
                    "predicted_class": predicted_label,
                    "confidence_percent" : confidence_percent
                }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
