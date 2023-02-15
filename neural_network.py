import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
from sklearn.model_selection import train_test_split
import numpy as np

#cargar y preprocesar las imagenes
path = 'Dataset/'
categories = os.listdir(path)

images = []
labels = []
for i, category in enumerate(categories):
    for image_name in os.listdir(path + category):
        image = cv2.imread(path + category + '/' + image_name)
        image = cv2.resize(image, (50,50))
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        images.append(image)
        labels.append(i)
X = np.array(images)
Y = np.array(labels)

#Dividir el nuero de imagenes y etiquetas en un conjunto de entrenemiento y un conjunto de prueba
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2)

#Convertir las etiquetas a un vector de categorias
y_train = tf.keras.utils.to_categorical(y_train, 4) # Change the number of categories here
y_test = tf.keras.utils.to_categorical(y_test, 4) # Change the number of categories here

model = keras.Sequential()
model.add(layers.Conv2D(32,(3,3),activation = 'relu', input_shape = (50,50,3))) # Change the input shape here
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(4, activation = 'softmax')) # Change the number of categories here

#Compilacion del modelo
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

#Entrenamiento del modelo
model.fit(x_train, y_train, batch_size = 32, epochs = 30, validation_data = (x_test, y_test))

#Evaluacion del modelo
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('Test accuracy:', test_acc)

#Guardar el modelo entrenado
model.save("fruit_classifier.h5")