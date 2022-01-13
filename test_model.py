import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import numpy as np
from set_vars import classes, test_dir, model_dir

def test_model(model, files):

    acc_count = 0

    for fname in files:
        img_path = os.path.join(test_dir, fname)
        img = image.load_img(img_path, target_size=(150, 150))

        x = image.img_to_array(img)
        x = x / 255. # 이미지 rescale
        x = np.expand_dims(x, axis=0)
        images = np.vstack([x])

        predict = model.predict(images, batch_size=4, verbose=0)
        np.set_printoptions(precision=3, suppress=True)

        result = predict.argmax()

        print("IMAGE NAME: {:13}, RESULT: {:7}, PROBA: {:.3f}"
                .format(fname, classes[result], predict.max()), end='')

        if fname[:2] == classes[result][:2]:
            acc_count += 1
            print()
        else:
            print("  !!!")


    print("\nTotal Accuracy: {:.3f}" .format(acc_count/len(files)))

if __name__ == '__main__':

    test_files = os.listdir(test_dir)
    if '.DS_Store' in test_files:
        test_files.remove('.DS_Store')

    model = load_model(model_dir, compile=False)
    test_model(model, test_files)
