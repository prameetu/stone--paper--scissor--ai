import os 
import cv2 as cv
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.utils import to_categorical

img_data_path = 'Image_Data'


classes = {'rock' : 0, 'paper' : 1 , 'scissor' : 2, 'random' : 3}

num_class = 4

def mapper(x):
    return classes[x]


data = []

for directory in os.listdir(img_data_path):

    path = os.path.join(img_data_path, directory)

    for i in os.listdir(path):
        img = cv.imread(os.path.join(path, i))
        img = cv.cvtColor(img , cv.COLOR_BGR2RGB)
        img = cv.resize(img, (227, 227))
        data.append([img, directory])

data_without_label ,labels = zip(*data)
labels = list(map(mapper, labels))

#One hot encode the labels

labels = to_categorical(labels)

vgg = VGG16(input_shape=[227,227,3], weights='imagenet', include_top=False)

for layer in vgg.layers:
    layer.trainable = False

x = Flatten()(vgg.output)

prediction = Dense(4, activation= 'softmax')(x)

model = Model(inputs = vgg.input ,outputs = prediction)

model.compile(optimizer = 'adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

model.fit(np.array(data_without_label), np.array(labels), epochs=5)

model.save("rock-paper-scissors-model.h5")
