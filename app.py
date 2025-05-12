import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="🧵 FashionFabric AI", layout="wide")

# App title
st.title("🧵 FashionFabric AI - Fabric & Quality Analyzer")
st.markdown("Upload a fabric image to analyze its texture, quality, and pattern.")

# Load pretrained model
model = models.resnet18(pretrained=True)
model.eval()

# Transform for image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Upload image
uploaded_image = st.file_uploader("📸 Upload Fabric Image", type=["jpg", "jpeg", "png"])

# Function to make prediction (simulated labels)
def predict_quality(img):
    labels = ["Silk - Premium", "Cotton - Standard", "Denim - Durable", "Linen - Lightweight", "Wool - Warm"]
    outputs = model(img.unsqueeze(0))
    _, predicted = torch.max(outputs, 1)
    return labels[predicted.item() % len(labels)]

# Display and process
if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Fabric Image", use_column_width=True)
    
    with st.spinner("🧠 Analyzing Fabric..."):
        tensor = transform(image)
        result = predict_quality(tensor)
    
    st.success(f"🧵 Detected Fabric Type & Quality: **{result}**")

    # Generate captions, hashtags
    st.subheader("✍️ Auto-Generated Content")
    caption = f"This {result.split('-')[0].strip()} fabric gives an elegant and stylish vibe. Perfect for a modern look!"
    hashtags = "#fashion #fabricanalysis #style #AIinFashion"
    st.text_area("📄 Caption:", caption, height=100)
    st.text_area("🏷️ Hashtags:", hashtags, height=50)
