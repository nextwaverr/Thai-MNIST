import keras
import cv2
from keras.models import load_model
import sys
import numpy as np

model = load_model('model/thaimnist.h5')
img_path = sys.argv
print(str(img_path[1]))
image = cv2.imread(str(img_path[1]),0)
image = image.reshape(1,28,28,1)

print('Image Shape ' + str(image.shape))
num =  model.predict(image)
print('One-hot Prediction ' + str(num[0]))
label = np.where(num[0] == 1)
print('Label Prediction = ' + str(label[0][0]))