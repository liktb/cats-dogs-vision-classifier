import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1. SETUP DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- USING DEVICE: {device} ---")

# 2. LOAD SAVED CHECKPOINT (`vision_model_best.pt`)
checkpoint = torch.load('vision_model_best.pt', map_location=device)
class_names = checkpoint['class_names']

# 3. REBUILD ARCHITECTURE & LOAD WEIGHTS
model = models.efficientnet_b0(weights=None)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Sequential(
    nn.Linear(num_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, len(class_names))
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval() # Set model to evaluation mode

# 4. PREPROCESS A TEST IMAGE
# Place any sample .jpg image inside your folder and update this filename
image_path = "cats_dogs_244/test/cat/cat_000001.jpg" 

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open(image_path).convert("RGB")
input_tensor = transform(image).unsqueeze(0).to(device) # Add batch dimension [1, 3, 224, 224]

# 5. RUN PREDICTION
with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs, dim=1)
    confidence, predicted_idx = torch.max(probabilities, 1)

predicted_class = class_names[predicted_idx.item()]
print(f"\n--- INFERENCE RESULT ---")
print(f"File Tested     : {image_path}")
print(f"Predicted Class : {predicted_class}")
print(f"Confidence      : {confidence.item() * 100:.2f}%")