import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

if __name__ == '__main__':
    # 1. SETUP DEVICE
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- USING DEVICE: {device} ---")

    # 2. LOAD SAVED CHECKPOINT
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
    model.eval()

    # 4. LOAD TEST DATASET
    test_dir = "cats_dogs_244/test"
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    print(f"\n--- EVALUATING ON {len(test_dataset)} TEST IMAGES ---")

    all_preds = []
    all_labels = []

    # 5. RUN EVALUATION LOOP
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 6. CALCULATE METRICS
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    accuracy = np.mean(all_preds == all_labels) * 100

    print(f"\n================ EVALUATION RESULTS ================")
    print(f"Overall Test Accuracy : {accuracy:.2f}%")
    print(f"====================================================")
    
    print("\nDetailed Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))