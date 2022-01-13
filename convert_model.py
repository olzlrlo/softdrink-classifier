import os
from tensorflow.python.keras.models import save_model, load_model
import tensorflow as tf

base_dir = '/Users/jieun/Desktop/softdrink_classifier'
model_dir = os.path.join(base_dir, 'model')
model = load_model(model_dir, compile=False)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

new_model_dir = model_dir + '.tflite'
f = open(new_model_dir, 'wb')
f.write(tflite_model)
f.close()
