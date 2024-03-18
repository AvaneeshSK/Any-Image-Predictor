# Any Image Predictor (Pretrained Model)

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

def preprocessing(image):
    img = tf.io.read_file(image)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)
    img = tf.image.resize(img, [224, 224])
    img = tf.reshape(img, [1, 224, 224, 3])
    return img


img_path = 'Machine Learning 2\\my_images\\orange.jpg'
labels_path = 'Machine Learning 2\\ImageNetLabels.txt'
img = preprocessing(img_path)
model = hub.load('https://www.kaggle.com/models/google/mobilenet-v3/frameworks/TensorFlow2/variations/large-075-224-classification/versions/1')


words = []
with open(labels_path, 'r') as f:
    words = f.readlines()
new_words = []
for word in words:
    new_words.append(word.replace('\n', ''))

prediction = model(img)
indx = np.argmax(prediction[0]) # some models require -1
confidence = np.sort(prediction[0])[-1]
print(f'Prediction : {new_words[indx]}')
confidence = format(confidence*10, '.2f')
print(f'Confidence : {confidence}%')
