import io
import os
from PIL import Image
from google.cloud import vision
from google.cloud.vision_v1 import types
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model
import numpy as np
from set_vars import classes, test_dir, model_dir

def with_model(model, files):
    def ocr(img_path):
        client = vision.ImageAnnotatorClient()

        img_file = Image.open(img_path)
        width, length = img_file.size[0] // 10, img_file.size[1] // 10
        area = (width * 2, length * 2, width * 7, length * 9)
        cropped, buffer = img_file.crop(box = area), io.BytesIO()
        cropped.save(buffer, "JPEG")
        content = buffer.getvalue()
        img = types.Image(content=content)

        response = client.text_detection(image=img)
        labels = response.text_annotations

        for label in labels:
            if label.description in drink_info.keys():
                return drink_info[label.description]
            if label.description.lower() in drink_info.keys():
                return drink_info[label.description.lower()]
        else:
            return -1

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

        if classes[result] == 'none':
            result = ocr(img_path)
            if result == -1:
                print("IMAGE NAME: {:13} ---------- OCR 결과 없음, 다른 사진 요청".format(fname))
                continue

        if fname[:2] == classes[result][:2]:
            print("IMAGE NAME: {:13}, RESULT: {:7}".format(fname, classes[result]))
            acc_count += 1
        else:
            print("IMAGE NAME: {:13}, RESULT: {:7} !!!!!!!!!".format(fname, classes[result]))

    print("\nTotal Accuracy: {:.3f}" .format(acc_count/len(files)))


if __name__ == '__main__':

    test_files = os.listdir(test_dir)
    if '.DS_Store' in test_files:
        test_files.remove('.DS_Store')

    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="/Users/jieun/Desktop/softdrink_classifier/python-ocr-test-jleeun-84f11becbc60.json"

    drink_info = dict()
    drink_info['cider'] = 0
    drink_info['coca'] = 1
    drink_info['cola'] = 1
    drink_info['fanta'] = 2
    drink_info['milkis'] = 3
    drink_info['monster'] = 4
    drink_info['mountain'] = 5
    drink_info['dew'] = 5
    drink_info['beenzino'] = 5
    drink_info['pepsi'] = 7
    drink_info['demisoda'] = 8
    drink_info['sprite'] = 9
    drink_info['toreta'] = 10
    drink_info['welchs'] = 11

    model = load_model(model_dir, compile=False)
    with_model(model, test_files)
