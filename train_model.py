import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import os

# Шляхи до директорій з даними
data_dir = '/Users/leat/Documents/ai/car/car/car_data/car_data'
train_dir = data_dir + '/train'
test_dir = data_dir + '/test'

# Перетворення для тренувальних даних, валідаційних та тестових даних
train_transforms = transforms.Compose([transforms.Resize((244,244)),
                                       transforms.RandomRotation(30),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Валідаційний набір використовуватиме ті ж перетворення, що й тестовий набір
test_transforms = transforms.Compose([transforms.Resize((244,244)),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

validation_transforms = transforms.Compose([transforms.Resize((244,244)),
                                            transforms.CenterCrop(224),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Завантаження наборів даних за допомогою ImageFolder
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

# Використання наборів даних та перетворень для визначення загрузчиків даних
# З trainloader, shuffle=True, щоб порядок зображень не впливав на модель
trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=True)

# Використання попередньо навченої моделі ResNet34
model = models.resnet34(pretrained=True)
# Заміна класифікатора
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 196)

# Визначення критерію та оптимізатора
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold=0.9)

# Реалізація функції для перевірки валідації
def validation(model, testloader, criterion):
    valid_loss = 0
    accuracy = 0
    
    # Перемикання моделі на режим "cpu"
    model.to('cpu')

    # Проходження даних через testloader
    for ii, (images, labels) in enumerate(testloader):
    
        # Перемикання зображень та міток на режим "cpu"
        images, labels = images.to('cpu'), labels.to('cpu')

        # Проходження зображень через модель для передбачення
        output = model.forward(images)
        # Обрахунок втрат
        valid_loss += criterion(output, labels).item()
        # Обрахунок ймовірності
        ps = torch.exp(output)
        
        # Обрахунок точності
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    
    return valid_loss, accuracy

epochs = 1
steps = 0
print_every = 40

# перемикання в режим "cpu"
model.to('cpu')
model.train()
for e in range(epochs):

    running_loss = 0
    
    # Ітерація по даним для здійснення кроку навчання
    for ii, (inputs, labels) in enumerate(trainloader):
        steps += 1
        
        inputs, labels = inputs.to('cpu'), labels.to('cpu')
        
        # обнулення градієнтів параметрів
        optimizer.zero_grad()
        # Прямий та зворотний проходи
        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Здійснення кроку валідації
        if steps % print_every == 0:
            # перемикання моделі на режим "оцінювання" під час валідації
            model.eval()
            
            # Вимкнення обрахунків градієнтів
            with torch.no_grad():
                valid_loss, accuracy = validation(model, testloader, criterion)
            
            print(f"No. epochs: {e+1}, \
            Training Loss: {round(running_loss/print_every,3)} \
            Valid Loss: {round(valid_loss/len(testloader),3)} \
            Valid Accuracy: {round(float(accuracy/len(testloader)),3)}")
            
            
            # Повернення моделі в режим навчання
            model.train()
            lrscheduler.step(accuracy * 100)
           
        
correct = 0
total = 0
model.to('cpu')

with torch.no_grad():
    for data in testloader:
        images, labels = data
        images, labels = images.to('cpu'), labels.to('cpu')
        # Отримання ймовірностей
        outputs = model(images)
        # Перетворення ймовірностей у передбачення
        _, predicted_outcome = torch.max(outputs.data, 1)
        # Загальна кількість зображень
        total += labels.size(0)
        # Підрахунок кількості правильних передбачень
        correct += (predicted_outcome == labels).sum().item()

print(f"Точність моделі на тесті: {round(100 * correct / total,3)}%")

# Збереження: ваги ознак, новий model.fc, відповідність індексу-класу, стан оптимізатора та кількість епох
checkpoint = {'state_dict': model.state_dict(),
              'model': model.fc,
              'class_to_idx': train_data.class_to_idx,
              'opt_state': optimizer.state_dict,
              'num_epochs': epochs}

torch.save(checkpoint, 'my_checkpoint12.pth')

