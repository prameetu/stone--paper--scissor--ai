import numpy as np
import sys
import cv2 as cv
from tensorflow.keras.models import load_model

img_path = sys.argv[1]

class_map  = { 0 : 'rock', 1 : 'paper' , 2 : 'scissor' , 3 : 'Random'}
model = load_model('rock-paper-scissors-model.h5')

img = cv.imread(img_path)

img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img = cv.resize(img, (227,227))
pred = model.predict(np.array([img]))

res = np.argmax(pred)

print(f'The image is of {class_map[res]}')

