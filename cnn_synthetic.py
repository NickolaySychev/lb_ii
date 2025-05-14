import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 2 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc(x)
        return x

def load_cifar10_test(batch_size=64):
    transform = transforms.Compose([transforms.ToTensor()])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return test_loader

def train_cnn(lab4_test_acc): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    synthetic_images = torch.load('synthetic_images.pt')
    synthetic_labels = torch.load('synthetic_labels.pt')
    train_dataset = TensorDataset(synthetic_images, synthetic_labels)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    test_loader = load_cifar10_test(batch_size=64)

    num_epochs = 50
    train_losses, test_losses, test_accuracies = [], [], [] 

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        test_loss = test_loss / len(test_loader.dataset)
        test_acc = correct / total
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        print(f"Эпоха {epoch}/{num_epochs} | Потери (train): {train_loss:.4f} | Потери (test): {test_loss:.4f} | Точность (test): {test_acc:.4f}")

    torch.save(model.state_dict(), 'cnn_synthetic.pt')

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Потери на обучении')
    plt.plot(test_losses, label='Потери на тесте')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(test_accuracies, label='Точность на тесте')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()
    plt.grid(True)
    plt.savefig('cnn_synthetic_metrics.png')
    plt.show()

    print(f"\nСравнение результатов:")
    print(f"Точность ЛР4 (TensorFlow, истинные данные): {lab4_test_acc:.2f}")
    print(f"Точность текущей модели (PyTorch, синтетические данные): {test_acc:.2f}")
    difference = test_acc - lab4_test_acc
    print(f"Разница в точности: {difference:.2f} ({'выше' if difference > 0 else 'ниже'})")

    with open('comparison_results.txt', 'w') as f:
        f.write(f"Точность ЛР4 (TensorFlow, истинные данные): {lab4_test_acc:.2f}\n")
        f.write(f"Точность текущей модели (PyTorch, синтетические данные): {test_acc:.2f}\n")
        f.write(f"Разница в точности: {difference:.2f} ({'выше' if difference > 0 else 'ниже'})\n")

    return test_acc

if __name__ == '__main__':
    lab4_test_acc = 0.8669 
    final_test_acc = train_cnn(lab4_test_acc)
    print(f"Итоговая точность на тестовом наборе: {final_test_acc:.2f}")