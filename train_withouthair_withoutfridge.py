import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Parametry
batch_size = 16
num_epochs = 5
learning_rate = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformacje (resize, tensor, normalizacja)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # dla RGB
])

# Wczytaj dane z folderów
train_data = datasets.ImageFolder('train_wh', transform=transform)
test_data = datasets.ImageFolder('test_wh', transform=transform)

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

# Model: ResNet18 z modyfikacją wyjścia do 1 klasy (binary)
model = models.resnet18(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, 1)  # Binary classification → 1 neuron
for param in model.fc.parameters():
    param.requires_grad = True
model = model.to(device)

# Strata i optymalizator
criterion = nn.BCEWithLogitsLoss()  # Binary Cross Entropy z logitami
optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

# Trenowanie
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)  # BCE expects shape (B, 1)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"[{epoch+1}/{num_epochs}] Loss: {running_loss/len(train_loader):.4f}")

# Ewaluacja
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predictions = (torch.sigmoid(outputs) > 0.5).int().squeeze()
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

print(f"Accuracy on test set: {100 * correct / total:.2f}%")
