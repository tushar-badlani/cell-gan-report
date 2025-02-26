import io

import numpy as np
from fastapi import FastAPI, UploadFile, File
from PIL import Image

from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from starlette.responses import Response
from torchvision import models

from torchvision import transforms

from app.model import generate_report
from app.util import get_lime

origins = ["*"]

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

image_size = 224

label_mapping = {0: 'ASC_H', 1: 'ASC_US', 2: 'HSIL', 3: 'LSIL', 4: 'NILM'}

transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.densenet121(pretrained=False)  # No pretrained weights
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 5)  # Modify for 5 classes

# Load weights
weights_path = "densenet121_fold5.pth"
try:
    # Load the saved file
    checkpoint = torch.load(weights_path, map_location=torch.device('cpu'))

    # If checkpoint is a full state dict with extra info, extract the model state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'module.' prefix if model was saved with DataParallel
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    # Load state dict into model
    model.load_state_dict(state_dict)
    print("Weights loaded successfully!")
except Exception as e:
    print(f"Error loading weights: {e}")
    raise

# Move to device and set to evaluation mode
model.eval()

print("Model loaded and ready for inference!")


#
# models.Base.metadata.create_all(bind=engine)
#

@app.get("/")
async def root():
    return {"message": "welcome to my api!!"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read the image file
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # Preprocess the image
    image = transform(image).unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        predicted = torch.argmax(outputs, 1).item()

    predicted_class = label_mapping[predicted]

    return predicted_class


@app.post("/lime")
async def lime(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = transform(image).unsqueeze(0)

    highlighted_image = get_lime(image)

    highlighted_image = (highlighted_image * 255).astype(np.uint8)  # Scale to 0-255
    highlighted_image = Image.fromarray(highlighted_image)

    buffer = io.BytesIO()
    highlighted_image.save(buffer, format="PNG")
    buffer.seek(0)

    # Return the processed image as a response
    return Response(content=buffer.getvalue(), media_type="image/png")


@app.post("/report")
async def report(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    image = transform(image)

    result = generate_report(image)

    return result

