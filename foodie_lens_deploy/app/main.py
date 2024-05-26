import torch
from torchvision import transforms
from fastapi import FastAPI, File, UploadFile
import torch
from PIL import Image
import io
import torch.nn as nn
import json

# Define your custom model class
class TinyVGG(nn.Module):
    """Model architecture copying TinyVGG from CNN Expaliner"""
    def __init__(self,
                input_shape: int,
                hidden_units: int,
                output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                        stride=2) #default stride value is same as kernel_size
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,
                        stride=2) #default stride value is same as kernel_size
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*66*66,
                    out_features=output_shape)
        )


    def forward(self, x):
        x = self.conv_block_1(x)
        # print(x.shape)
        x = self.conv_block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

# Class names for the model
class_names = ['asaro', 'eforiro', 'ekuru', 'ewagoyin', 'moimoi']

# Load the recipes from the JSON file with the correct encoding
with open('app/recipe.json', 'r', encoding='utf-8') as f:
    recipe = json.load(f)

# Load the saved PyTorch model
device = torch.device('cpu')
model = TinyVGG(input_shape=3, hidden_units=20, output_shape=len(class_names))
model.load_state_dict(torch.load('app/model/foodie_lens_model.pth', map_location=device))
model.to(device)
model.eval()

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()
])

def predict(image: Image.Image):
    # Apply the transforms to the image
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    image = image.to(device)  # Move image to the appropriate device

    # Make the prediction
    with torch.no_grad():
        output = model(image)
    
    # Get the predicted class
    _, predicted = torch.max(output, 1)
    return predicted.item()

# Create the FastAPI application
app = FastAPI()

# Function to handle image upload and prediction
@app.post("/predict")
async def predict_image(image: UploadFile = File(...)):
    try:
        # Read the uploaded image
        img = Image.open(image.file)

        #Preprocess the image
        img = transform(img)
        img = img.unsqueeze(0)  # Add a batch dimension for model compatibility
        img = img.to(device)  # Move image to the appropriate device

        # Make prediction using the model
        with torch.no_grad():  # Disable gradient calculation for prediction
            outputs = model(img)

        # Get the top predicted class and its probability
        _, predicted = torch.max(outputs.data, 1)
        probability = torch.nn.functional.softmax(outputs, dim=1)[0][predicted.item()]

        # Return the prediction results
        return {
            "predicted_class": class_names[predicted.item()],  # Class name representing the class
            "probability": probability.item(),
            "recipe": recipe[class_names[predicted.item()]]  # Recipe for the predicted class
        }
    except Exception as e:
        return {"error": str(e)}  # Handle potential errors graceful
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
