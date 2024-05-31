### Project Title: Foodie lens
Foodie Lens is a machine learning-based application designed to identify various traditional Yoruba dishes from images and provide the corresponding recipes and cooking instructions.

### Try it out here: https://huggingface.co/spaces/Aramide/foodie_lens

### Getting Started
These instructions will help you get a copy of the project up and running on your local machine

### Installation
- Clone the repository
  `git clone https://github.com/Adebesin-Aramide/Foodie_Lens.git`
   `cd Foodie_Lens` 

- Install dependencies
  `pip install -r requirements.txt`

- Run the application locally
  `streamlit run app.py`

### Deployment on Hugging Face
The application is deployed on Hugging Face Spaces. To deploy it:

1. Create a Hugging Face account if you don't have one.
2. Create a new Space and select Streamlit as the SDK.
3. Push your code to the new Space's repository.
4. Hugging Face will automatically build and deploy the application.

### Usage
- Upload an image: Use the file uploader to select an image of a traditional Yoruba dish (limited to five dishes for now: asaro, ewagoyin, ekuru, moimoi, eforiro).
- View the prediction: The application will classify the dish and display the predicted class.
- Get the recipe: The application will provide the recipe and cooking instructions for the identified dish.

### Data Collection
The dataset for this project was collected by web scraping images of traditional Yoruba dishes from Google. The images were then labeled and used to train the deep learning model.

### Model
The custom model is based on a simplified version of VGG (TinyVGG) built using PyTorch. The model architecture includes two convolutional blocks followed by a classifier.

