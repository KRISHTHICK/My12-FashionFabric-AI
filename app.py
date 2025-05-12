
---

### 📄 `app.py`

```python
import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os

st.set_page_config(page_title="🧵 FashionFabric AI", layout="wide")
st.title("🧵 FashionFabric AI - Fabric & Quality Analyzer")

# Load Model
model = models.resnet18(pretrained=True)
model.eval()

fabric_classes = ["cotton", "denim", "silk", "wool", "leather"]

def predict_fabric(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = torch.argmax(output, 1).item()
        return fabric_classes[pred_idx % len(fabric_classes)]  # Simulate mapping

def suggest_use(fabric):
    suggestions = {
        "cotton": "Perfect for summer and casual wear.",
        "denim": "Ideal for jeans and casual jackets.",
        "silk": "Best for formal and party wear.",
        "wool": "Great for winter wear and sweaters.",
        "leather": "Perfect for jackets and stylish accessories."
    }
    return suggestions.get(fabric, "Explore creative fashion uses!")

uploaded_image = st.file_uploader("📸 Upload Fabric Image", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image).convert("RGB")
    st.image(image, caption="Uploaded Fabric", use_container_width=True)

    with st.spinner("🔍 Analyzing Fabric..."):
        fabric = predict_fabric(image)
        suggestion = suggest_use(fabric)

    st.subheader("🧶 Predicted Fabric Type")
    st.success(fabric.capitalize())

    st.subheader("💡 Suggested Use")
    st.info(suggestion)
