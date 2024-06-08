import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import json

# Define your custom model class
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 66 * 66, out_features=output_shape)
        )

    def forward(self, x):
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        x = self.classifier(x)
        return x

class_names = ['asaro', 'eforiro', 'ekuru', 'ewagoyin', 'moimoi']

# Load the recipes from the JSON file
with open('recipe.json', 'r') as f:
    recipes = json.load(f)

# Load the saved PyTorch model
device = torch.device('cpu')
model = TinyVGG(input_shape=3, hidden_units=20, output_shape=len(class_names))
model.load_state_dict(torch.load('foodie_lens_model2.pth', map_location=device))
model.to(device)
model.eval()

# Define the image transforms
transform = transforms.Compose([
    transforms.Resize(size=(256, 256)),
    transforms.ToTensor()
])

def predict(image):
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

# Streamlit interface
st.title("Foodie Lens")
st.markdown("""
**Foodie Lens** helps you identify various traditional Yoruba dishes from an image and provides recipes and cooking instructions for the identified dish. Currently, you can upload images of Asaro, Eforiro, Ekuru, Ewagoyin, and Moimoi. Stay tuned as more classes of food will be added soon. Upload an image to get started!
""")
# Improved CSS styling for better visibility
st.markdown(
     """
    <style>
    .main { 
        background-color: #F1E5D1;  /* Light grey background */
        color: #333333;
    }
    .stButton>button { 
        background-color: #987070; 
        color: white; 
    }
    h1, h2, h3, h4, h5, h6, p, label {
        color: #333333;
    }
    .uploadedImage {
        border: 2px solid #f58216;
        border-radius: 5px;
        padding: 10px;
        background-color: #fff;  /* White background */
    }
    .stFileUploader {
        background-color: #987070 !important;  /* Light grey background */
        color: #F1E5D1;
        border: 2px dashed #f58216;
        border-radius: 5px;
        padding: 20px;
    }
    </style>
    """
    ,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="uploader")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.markdown('<div class="uploadedImage">', unsafe_allow_html=True)
    st.write("Classifying...")
    label = predict(image)
    predicted_class = class_names[label]
    st.write(f"**Predicted Class:** {predicted_class.capitalize()}")


    
    # Display the recipe and instructions

    if predicted_class in recipes:
        recipe = recipes[predicted_class]
        
        st.markdown(f"### Recipe for {predicted_class.capitalize()}")
        st.markdown(f"**Ingredients:** {', '.join(recipe['Ingredients'])}")
        st.markdown(f"**Instructions:** {', '.join(recipe['Instructions'])}")
    else:
        st.write(f"Recipe not found for the predicted class '{predicted_class}'.")
    st.markdown('</div>', unsafe_allow_html=True)
