import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

if __name__ == '__main__':
    # 1. SETUP DEVICE (Leverage RTX 4070 Mobile GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- USING DEVICE: {device} ---")
    if device.type == "cuda":
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. DEFINE DATA TRANSFORMS & AUGMENTATIONS
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 3. LOAD DATASETS USING RELATIVE PATHS
    train_dir = "cats_dogs_244/train"
    val_dir = "cats_dogs_244/test"

    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    class_names = train_dataset.classes
    print(f"Detected Classes: {class_names}")

    # Optimized batch size for 8GB VRAM
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    # 4. SET UP TRANSFER LEARNING MODEL (EfficientNet-B0)
    print("\n--- LOADING PRETRAINED MODEL ---")
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)

    # Freeze backbone feature extractor layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace classification head
    num_features = model.classifier[1].in_features
    model.classifier[1] = nn.Sequential(
        nn.Linear(num_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, len(class_names))
    )

    model = model.to(device)

    # 5. DEFINE LOSS & OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # 6. TRAINING LOOP
    num_epochs = 5
    print(f"\n--- STARTING TRAINING FOR {num_epochs} EPOCHS ---")

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total_train += labels.size(0)
            correct_train += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total_train
        epoch_acc = correct_train / total_train

        # Validation Phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total_val += labels.size(0)
                correct_val += predicted.eq(labels).sum().item()

        val_epoch_loss = val_loss / total_val
        val_epoch_acc = correct_val / total_val

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc*100:.2f}% | "
              f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc*100:.2f}%")

    # 7. SAVE MODEL ARTIFACT
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names
    }, 'vision_model_best.pt')

    print("\nTraining complete! Saved model weights to 'vision_model_best.pt'")