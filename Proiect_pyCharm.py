import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F
import time
import os
import shutil
import logging
from PIL import Image
from torchvision import utils
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import accuracy_score, recall_score

logging.basicConfig(level=logging.INFO)

learning_rate = 0.003
epochs = 0
batch_size = 64
weight_decay = 0.01  # asta e factorul de regularizare


# Șterge folderul de checkpoints
if os.path.isdir(os.path.join('./Plants_2/train', '.ipynb_checkpoints')):
    shutil.rmtree(os.path.join('./Plants_2/train', '.ipynb_checkpoints'))

def plot_class_distribution(dataset, title):
    # Obține etichetele din dataset
    print("class")
    labels = [label for _, label in dataset]
    print("label")

    # Calculează frecvența fiecărei clase
    unique, counts = np.unique(labels, return_counts=True)
    print("uniquq")

    # Creează un dicționar pentru a păstra frecvențele
    class_distribution = dict(zip(unique, counts))

    print("afisez hist")

    # Afișează histograma
    plt.figure(figsize=(10, 5))
    plt.bar(class_distribution.keys(), class_distribution.values(),
            tick_label=[str(k) for k in class_distribution.keys()])
    plt.xlabel('Clase')
    plt.ylabel('Frecvență')
    plt.title(title)
    plt.show()

def imshow(img, title):
    img = img / 2 + 0.5  # De-normalizează imaginea
    npimg = img.cpu().numpy()  # Adaugă .cpu() înainte de a converti în NumPy
    plt.imshow(np.transpose(npimg, (1, 2, 0)))  # afișați imaginea
    plt.title(title)
    plt.show()

class SimpleDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            image = Image.open(path).convert('RGB')
            if image.size != (32, 32):
                image = image.resize((32, 32))
            image = self.transform(image)
            return image, target
        except Exception as e:
            logging.error(f"Error loading image {index} from {path}: {e}")
            raise e


def run_model():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = SimpleDataset(root='./Plants_2/train', transform=transform)
    valid_dataset = SimpleDataset(root='./Plants_2/valid', transform=transform)
    test_dataset = SimpleDataset(root='./Plants_2/test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, num_workers=4, persistent_workers=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, persistent_workers=True, pin_memory=True)

    #plot_class_distribution(train_dataset, "Distribuția claselor în setul de antrenare")
    #plot_class_distribution(valid_dataset, "Distribuția claselor în setul de validare")
    # plot_class_distribution(test_dataset, "Distribuția claselor în setul de antrenare")

    #distributie train

    # # Obținerea unui batch de imagini
    # dataiter = iter(train_loader)
    # images, labels = next(dataiter)
    #
    # images = images.to(device)
    # labels = labels.to(device)
    #
    # # Obținerea claselor din setul de date
    # classes = train_dataset.classes
    #
    # # Afișarea imaginilor cu etichetele lor
    # imshow(utils.make_grid(images), title=[classes[label] for label in labels])

    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 22)
            #self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            #x = self.dropout(x)
            return x

    model = SimpleCNN()
    model = model.to(device)

    # Verifică dacă există un model salvat și încarcă-l, dacă este cazul
    if os.path.exists('modelIncercare2.pth'):
        model.load_state_dict(torch.load('modelIncercare2.pth'))
        print("Modelul a fost încărcat cu succes!")
    else:
        print("Nu există niciun model salvat. Se va începe antrenarea de la zero.")

    # Cross Entropy Loss
    criterion = nn.CrossEntropyLoss()

    # # Weighted Cross Entropy Loss
    # weights = torch.tensor([1.0] * 22, dtype=torch.float)
    # criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    # # Kullback-Leibler Divergence Loss
    # criterion = torch.nn.KLDivLoss(reduction='batchmean')

    # # SGD (Stochastic Gradient Descent)
    # optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Adam (Adaptive Moment Estimation)
   # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # # RMSprop (Root Mean Square Propagation)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)

    # Define the learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    print("iterez")

    train_losses, valid_losses = [], []

    for epoch in range(epochs):
        start_time = time.time()  # Start timing
        model.train()
        train_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}.. ", flush=True)
        for idx, (images, labels) in enumerate(train_loader, 1):
            #print(f"Batch {idx}/{len(train_loader)} loaded successfully", flush=True)
            images, labels = images.to(device), labels.to(device)  # Modificare aici
            optimizer.zero_grad()
            outputs = model(images)  # Modificare aici
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_losses.append(train_loss / len(train_loader))
        print(f"Train loss: {train_losses[-1]:.3f}", flush=True)

        model.eval()
        valid_loss = 0
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        valid_losses.append(valid_loss / len(valid_loader))
        accuracy = accuracy_score(all_labels, all_predictions)
        recall = recall_score(all_labels, all_predictions, average='macro', zero_division=1)
        print(f"Valid loss: {valid_losses[-1]:.3f}, Accuracy: {accuracy:.3f}, Recall: {recall:.3f}", flush=True)

        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt=".0f", linewidths=.5, square=True, cmap='Blues_r')
        plt.ylabel('Actual label')
        plt.xlabel('Predicted label')
        plt.title('Matricea de Confuzie', size=15)
        plt.show()

        end_time = time.time()
        epoch_duration = end_time - start_time
        print(f"Timpul de execuție al epocii {epoch + 1}: {epoch_duration:.2f} secunde")

        # # Update learning rate scheduler
        # scheduler.step()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(valid_losses, label='Valid Loss')
    plt.title('Loss pe parcursul antrenării')
    plt.xlabel('Epoci')
    plt.ylabel('Pierdere')
    plt.legend()
    plt.show()

    # Salvarea modelului antrenat
    torch.save(model.state_dict(), 'modelIncercare2.pth')

    #evaluare

    model.eval()

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    print(f"Test Loss: {test_loss / len(test_loader):.3f}")
    print(f"Test Accuracy: {100 * correct / total:.2f}%")


def prediction_function(image_path, model_path):
    class SimpleCNN(nn.Module):
        def __init__(self):
            super(SimpleCNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
            self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.fc1 = nn.Linear(128 * 4 * 4, 256)
            self.fc2 = nn.Linear(256, 22)
            #self.dropout = nn.Dropout(p=0.5)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = x.view(-1, 128 * 4 * 4)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            #x = self.dropout(x)
            return x

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("sunt in functia de predictie")

    transformSet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Define the class names (make sure these match your dataset's class names)
    train_dataset = SimpleDataset(root='./Plants_2/train', transform=transformSet)
    class_names = train_dataset.classes

    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    # Load the model
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    print("fac predictia")

    # Make predictions
    image = image.to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    # Display the image with the prediction
    plt.imshow(Image.open(image_path))
    plt.title(f'Prediction: {predicted_class}')
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    #run_model()

    # Call the function for a single image
    image_path = './Plants_2/test/Alstonia Scholaris diseased (P2a)/0014_0006.JPG'
    model_path = 'modelIncercare2.pth'
    prediction_function(image_path,model_path)

    # Call the function for a single image
    image_path = './Plants_2/test/Mango diseased (P0b)/0012_0003.JPG'
    model_path = 'modelIncercare2.pth'
    prediction_function(image_path, model_path)

    # Call the function for a single image
    image_path = './Plants_2/test/Pongamia Pinnata healthy (P7a)/0007_0003.JPG'
    model_path = 'modelIncercare2.pth'
    prediction_function(image_path, model_path)

    # Call the function for a single image
    image_path = './Plants_2/test/Lemon healthy (P10a)/0010_0004.JPG'
    model_path = 'modelIncercare2.pth'
    prediction_function(image_path, model_path)

