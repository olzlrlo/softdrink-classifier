import os
from tensorflow.python.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import softdrink_classifier as sftdrk
from set_vars import train_dir, test_dir, model_dir, new_model_dir
from set_vars import BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS, CLASSES

def add_class(model):
    new_model = Sequential()

    # 새 모델에 기존의 layer 추가, 마지막 두 개의 layer는 제외
    for layer in model.layers[:-2]:
        new_model.add(layer)
        print(layer)

    # 이미 훈련된 layer는 훈련 안 함
    for layer in new_model.layers:
        layer.trainable = False

    # 새로운 layer를 추가
    new_model.add(Dense(512, name='new_Dense', activation='relu'))  # 완전 연결 계층
    new_model.add(Dense(CLASSES, activation='softmax'))  # 분류기

    new_model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy', metrics=['accuracy'])

    return new_model

if __name__ == '__main__':

    model = load_model(model_dir, compile=False)

    train_gen, validation_gen = sftdrk.generate_image()
    model = add_class(model)

    result = sftdrk.fit_model(new_model_dir, model, train_gen, validation_gen)
    sftdrk.print_result(result)
