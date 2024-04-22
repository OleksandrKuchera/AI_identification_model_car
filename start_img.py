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
#valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

def load_checkpoint(filepath):
    # Оголошуємо модель всередині функції load_checkpoint
    model = models.resnet34(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 196)

    checkpoint = torch.load(filepath)
    
    # Завантажуємо параметри моделі з чекпоінту
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

# Завантаження моделі
model = load_checkpoint('my_checkpoint1.pth')
# Перевірка моделі, вона має мати 196 вихідних одиниць у класифікаторі
print(model)

# Перевірка наявності GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Паралелізація моделі для використання на багатьох пристроях
model = torch.nn.DataParallel(model)

# Приймає та обробляє зображення в NumPY код
def process_image(image):
    
    # Обробка зображення типу PIL для використання в моделі PyTorch

    # Конвертація зображення до типу PIL за шляхом до файлу зображення
    pil_im = Image.open(f'{image}' + '.jpg')

    # Створення перетворення зображення
    transform = transforms.Compose([transforms.Resize((244,244)),
                                    #transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], 
                                                         [0.229, 0.224, 0.225])]) 
    
    # Застосування перетворення до зображення для використання в мережі
    pil_tfd = transform(pil_im)
    
    # Конвертація в масив NumPy
    array_im_tfd = np.array(pil_tfd)

    print('Перетворене зображення в NumPy', array_im_tfd)
    return array_im_tfd

def imshow(image, ax=None, title=None):
    if ax is None:
        fig, ax = plt.subplots()
    
    # Тензори PyTorch передбачають, що колірний канал - перший розмір,
    # але matplotlib передбачає, що це третій розмір
    image = image.transpose((1, 2, 0))
    
    # Скасування попередньої обробки
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Обмеження значень зображення від 0 до 1 для коректного відображення
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)

    print('Зображення нормалізовано від 0 до 1' ,ax)
    return ax

#imshow(process_image(data_dir + '/train/Bugatti Veyron 16.4 Coupe 2009/00267'))

def predict(image_path, model, topk=5):
    # Реалізація коду для передбачення класу за файлом зображення   
    
    # Завантаження моделі - використання .cpu() для роботи з CPU
    loaded_model = load_checkpoint(model).cpu()
    # Попереднє оброблення зображення
    img = process_image(image_path)
    # Конвертація в тензор PyTorch з масиву NumPy
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor)
    # Додавання розмірності до зображення для відповідності (B x C x W x H) вхідному формату моделі
    img_add_dim = img_tensor.unsqueeze_(0)

    # Встановлення моделі у режим інференсу та вимкнення обрахунків градієнтів
    loaded_model.eval()
    with torch.no_grad():
        # Передача зображення через мережу
        output = loaded_model.forward(img_add_dim)
        
    #conf, predicted = torch.max(output.data, 1)   
    probs_top = output.topk(topk)[0]
    predicted_top = output.topk(topk)[1]
    
    # Конвертація ймовірностей та виходів в список
    conf = np.array(probs_top)[0]
    predicted = np.array(predicted_top)[0]
        
    #return probs_top_list, index_top_list
    return conf, predicted

# Зв'язування індексів класів з їх назвами

def find_classes(dir):
    classes = os.listdir(dir)
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx
classes, c_to_idx = find_classes(data_dir+"/test")

#print(classes, c_to_idx)

model_path = 'my_checkpoint1.pth'
image_path = data_dir + '/test/Audi RS 4 Convertible 2008/00748'
#/test/Bugatti Veyron 16.4 Convertible 2009/03409
#/test/Audi RS 4 Convertible 2008/00748'



conf1, predicted1 = predict(image_path, model_path, topk=5)

print(conf1)
print(classes[predicted1[0]])



# Вхідними даними є шляхи до збереженої моделі та тестового зображення
carname = 'I'


conf2, predicted1 = predict(image_path, model_path, topk=5)
# Конвертація класів у назви
names = []
for i in range(5):
  
    names += [classes[predicted1[i]]]

# Створення зображення типу PIL
image = Image.open(image_path+'.jpg')

# Побудова тестового зображення та передбачених ймовірностей
f, ax = plt.subplots(2,figsize = (6,10))

ax[0].imshow(image)
ax[0].set_title(carname)

y_names = np.arange(len(names))
ax[1].barh(y_names, conf2/conf2.sum(), color='darkblue')
ax[1].set_yticks(y_names)
ax[1].set_yticklabels(names)
ax[1].invert_yaxis() 

plt.show()
