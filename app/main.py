from fastapi import FastAPI, HTTPException, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import requests
from PIL import Image
import timm
import torch
from torchvision import transforms
import torch.nn.functional as F
from io import BytesIO
import base64
import os

app = FastAPI()
# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory = os.path.join(os.getcwd(),"app/templates"))

# Global variables for the model, transform, and class dataset
model = None
transform = None
class_dataset = []
confidence_cutoff = 80
print("-----------------------------> ", os.path.join(os.getcwd(),'model/image_classification_checkpoint.pth'))

# Initialize the model and the transformations
def init():
    global model, transform, class_dataset

    NUM_CLASSES = 10
    model = timm.create_model('resnet50d.ra4_e3600_r224_in1k', pretrained=False, num_classes=NUM_CLASSES)
    checkpoint = torch.load(os.path.join(os.getcwd(),'app/model/image_classification_checkpoint.pth'), map_location=torch.device('cpu'))
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

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "class_dataset": class_dataset  # Pass the class dataset to the template
    })

# Endpoint for predicting from an image URL
@app.post("/predict_url/")
def predict_from_url(request: Request, image_url: str = Form(...)):
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

        # Create a low-resolution version of the image (thumbnail)
        thumbnail_size = (100, 100)  # Size for the thumbnail
        image.thumbnail(thumbnail_size)
        
        # Save the thumbnail to a BytesIO object
        thumbnail_io = BytesIO()
        image.save(thumbnail_io, format="JPEG")
        thumbnail_io.seek(0)  # Move to the beginning of the BytesIO object

        # Convert thumbnail to base64 for embedding in the HTML response
        thumbnail_base64 = base64.b64encode(thumbnail_io.getvalue()).decode('utf-8')

        if confidence_percent > confidence_cutoff:
            # Render the HTML template with the predicted label and thumbnail
            return templates.TemplateResponse("prediction_result.html", {
                "request": request,
                "predicted_label": predicted_label,
                "confidence_percent": confidence_percent,
                "thumbnail_base64": thumbnail_base64
            })
        else:
            predicted_label = "Not in Available Classes"
            # Render the HTML template with the predicted label and thumbnail
            return templates.TemplateResponse("prediction_result.html", {
                "request": request,
                "predicted_label": predicted_label,
                "confidence_percent": confidence_percent,
                "thumbnail_base64": thumbnail_base64
            })

    except requests.exceptions.RequestException as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": str(e)
        })
    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "An unexpected error occurred: " + str(e)
        })

# Endpoint for predicting from an uploaded image file
@app.post("/predict_file/")
def predict_from_file(request: Request, file: UploadFile = File(...)):
# def predict_from_file(file: UploadFile = File(...)):
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

        # Create a low-resolution version of the image (thumbnail)
        thumbnail_size = (100, 100)  # Size for the thumbnail
        image.thumbnail(thumbnail_size)
        
        # Save the thumbnail to a BytesIO object
        thumbnail_io = BytesIO()
        image.save(thumbnail_io, format="JPEG")
        thumbnail_io.seek(0)  # Move to the beginning of the BytesIO object

        # Convert thumbnail to base64 for embedding in the HTML response
        thumbnail_base64 = base64.b64encode(thumbnail_io.getvalue()).decode('utf-8')

        # Render the HTML template with the predicted label and thumbnail
        if confidence_percent > confidence_cutoff:
            # Render the HTML template with the predicted label and thumbnail
            return templates.TemplateResponse("prediction_result.html", {
                "request": request,
                "predicted_label": predicted_label,
                "confidence_percent": confidence_percent,
                "thumbnail_base64": thumbnail_base64
            })
        else:
            predicted_label = "Not in Available Classes"
            # Render the HTML template with the predicted label and thumbnail
            return templates.TemplateResponse("prediction_result.html", {
                "request": request,
                "predicted_label": predicted_label,
                "confidence_percent": confidence_percent,
                "thumbnail_base64": thumbnail_base64
            })


    except Exception as e:
        return templates.TemplateResponse("error.html", {
            "request": request,
            "error_message": "An unexpected error occurred: " + str(e)
        })
