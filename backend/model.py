import torch
import timm
from PIL import Image
import torchvision.transforms as transforms
import json

# Load pretrained model
model = timm.create_model('efficientnet_b4', pretrained=True)
model.eval()

# Load ImageNet labels (OFFLINE)
with open("imagenet_labels.json", "r") as f:
    imagenet_labels = json.load(f)

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def detect_animal(image: Image.Image):
    # 🔥 FIX: ensure 3 channels (RGB)
    image = image.convert("RGB")

    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top_prob, top_catid = torch.topk(probabilities, 1)
    confidence = top_prob.item() * 100
    label = imagenet_labels[top_catid.item()]

    # Animal filter (same as before)
    if top_catid.item() > 397:
        return "No animal detected"

    if confidence < 40:
        return "No animal detected"

    return f"Animal: {label}, Confidence: {confidence:.2f}%"