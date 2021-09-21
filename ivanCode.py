import numpy as np
import os
import cv2

DATADIR = "/content/drive/MyDrive/Data1000"  # шлях до папки з відповідними данними на гугл диску
Categories = ["OneBand", "TwoBands"]
training_data = []
IMG_SIZE = 100


def CreateTrainingData():
    for category in Categories:
        path = os.path.join(DATADIR, category)  #  шлях до папок "OneBand" та "TwoBands"
    class_num = Categories.index(category)
    try:
        for img in os.listdir(path):
            img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            training_data.append([new_array, class_num])
    except Exception as e:
        pass


CreateTrainingData()  # загалом, даний блок є блоком підготовки данних до обробки
def CreateTestingData():
    for category in Categories_test: 
        path = os.path.join(DATADIR,category) # шлях до папок "OneBand" та "TwoBands"
        class_num=Categories_test.index(category) 
        try:
            for img in os.listdir(path):
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
                testing_data.append([new_array ,class_num])
        except Exception as e:
          pass
CreateTestingData()
import random

random.shuffle(training_data)
X = []
y = []
for feature, label in training_data:
    X.append(feature)
y.append(label)
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
y = np.array(y).reshape(-1)
random.shuffle(testing_data) 
X_test=[]
y_test=[]
for feature,label in testing_data:
    X_test.append(feature)
    y_test.append(label)
X_test=np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,1)
y_test=np.array(y_test).reshape(-1)
# розділення данних на вхідні (Х) та вихідні (y)
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import matplotlib

X = X / 255.0

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = X.shape[1:])) # додаємо згортковий шар
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) # додаємо агрегувальинй шар

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten()) # перетворюємо двовимірні дані в одновимірні

model.add(Dense(64)) #  додаємо шар з 64 нейронами

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer ='adam') #  метод оптимізації нейронної мережі
loss = binary_crossentropy',   #  функція втрат
metrics = ['accuracy'])  # функція оцінки якості роботи моделі
history = model.fit(X, y, batch_size = 50, epochs = 4, validation_split = 0.3)

#  функція навчання нейронної мережі, яка містить вхідні й вихідні дані,
#  кількість епох навчання, відсоток данних, що буде використовуватись для перевірки ефективності моделі

history.history['loss']
history.history['val_loss']
history.history['accuracy']
history.history['val_accuracy']
matplotlib.pyplot.plot(history.history['val_accuracy'], label = 'val_accuracy')
matplotlib.pyplot.plot(history.history['accuracy'], label = 'accuracy')
matplotlib.pyplot.title('Training and validation accuracy on Data500')
matplotlib.pyplot.legend()
model.evaluate(X_test, y_test)
