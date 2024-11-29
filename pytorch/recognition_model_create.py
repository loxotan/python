import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import pandas as pd
from PIL import Image

from sklearn.preprocessing import LabelEncoder

class CustomDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.folder_path = folder_path
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_encoder = LabelEncoder()

        for label_folder in os.listdir(folder_path):
            label_folder_path = os.path.join(folder_path, label_folder)
            if os.path.isdir(label_folder_path):
                for image_name in os.listdir(label_folder_path):
                    self.image_paths.append(os.path.join(label_folder_path, image_name))
                    self.labels.append(label_folder)
        
        # Fit label encoder
        self.labels = self.label_encoder.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label


def excel_list_up():
    excel_path = "C:\\Users\\user\\Desktop\\숫자 확인 모델\\patient_list.xlsx"
    df = pd.read_excel(excel_path)
    patient_dict = {str(row[0]): row[1] for _, row in df.iterrows()}
    return patient_dict


def train_model():
    data_path = "C:\\Users\\user\\Desktop\\숫자 확인 모델"
    train_folder = os.path.join(data_path, 'train')
    test_folder = os.path.join(data_path, 'test')
    model_save_path = "C:/Users/user/Desktop/숫자 확인 모델/recognition_model.pth"

    # Prepare Dataset and DataLoader
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    train_dataset = CustomDataset(train_folder, transform=transform)
    test_dataset = CustomDataset(test_folder, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # Load Pre-trained Model
    model = models.resnet18(pretrained=True)
    num_classes = len(set(train_dataset.labels))
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify for your number of classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Loop
    patience = 3
    best_accuracy = 0.0
    patience_counter = 0

    for epoch in range(10):
        model.train()
        running_loss = 0.0
        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Evaluate on Test Set
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                images, labels = data
                images = images.to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        print(f'Epoch [{epoch+1}/10], Loss: {running_loss:.4f}, Accuracy: {accuracy:.4f}')

        # Early Stopping Check
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience or (1 - accuracy) < 0.001:
            print("Premature ending triggered.")
            break

    # Save Model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')


if __name__ == "__main__":
    patient_dict = excel_list_up()
    train_model()
