import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from model import face_mask_detection
    
# Load model and preprocessing pipeline
model = face_mask_detection()
model.load_state_dict(torch.load('face_mask.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

# Preprocessing pipeline
preprocessing_pipeline = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

class_labels = [0,1]

# Streamlit app
st.title("Face Mask Detection")
st.write("Upload an image to determine if a person is wearing a mask or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image = image.convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    input_tensor = preprocessing_pipeline(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = class_labels[predicted.item()]

    if label==1:
        st.write(f"Prediction: **{"With Mask"}**")
    else:
        st.write(f"Prediction: **{"Without Mask"}**")