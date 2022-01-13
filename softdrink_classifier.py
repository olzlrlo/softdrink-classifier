# Python Keras를 이용한 다중 클래스 음료 분류

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
from set_vars import train_dir, validation_dir, model_dir
from set_vars import CLASSES, IMAGE_SIZE, BATCH_SIZE, EPOCHS, STEPS_PER_EPOCH, VALIDATION_STEPS

# 1. 이미지 처리
def generate_image():
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255., # 이미지 픽셀 값을 0 ~ 1로 정규화
        rotation_range=40, # 정해진 각도 범위에서 이미지 회전
        width_shift_range=0.2, # 정해진 수평 방향 이동 범위에서 이미지 이동
        height_shift_range=0.2, # 정해진 수직 방향 이동 범위에서 이미지 이동
        shear_range=0.2, # 정해진 층밀리기 강도 범위에서 이미지 변형
        horizontal_flip=True) # 수평방향 뒤집기

    validation_datagen = ImageDataGenerator(rescale=1.0 / 255.)

    train_generator = train_datagen.flow_from_directory(
        train_dir, batch_size=BATCH_SIZE, class_mode='categorical', target_size=(IMAGE_SIZE, IMAGE_SIZE))
    validation_generator = validation_datagen.flow_from_directory(
        validation_dir, batch_size=BATCH_SIZE, class_mode='categorical', target_size=(IMAGE_SIZE, IMAGE_SIZE))

    return train_generator, validation_generator

# 2. 모델 생성
def create_model():
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Conv2D(512, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Dropout(0.25),

        Flatten(),
        Dense(1000, activation='relu'),
        Dropout(0.25),
        Dense(CLASSES, activation='softmax')
    ])

    # RMSprop (Root Mean Square Propagation) : 훈련 중에 학습률을 적절히 조절
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    return model

# 3. 모델 훈련
def fit_model(save_dir, model, train_gen, validation_gen):
    checkpoint = ModelCheckpoint(save_dir, monitor='val_loss', mode='min', verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
    history = model.fit(train_gen,
                        validation_data=validation_gen,
                        epochs=EPOCHS,
                        steps_per_epoch=STEPS_PER_EPOCH,
                        validation_steps=VALIDATION_STEPS,
                        verbose=2,
                        callbacks=[checkpoint, early_stopping])
                        #callbacks=[checkpoint])
    return history

# 4. 결과 출력
def print_result(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and Validation accuracy')
    plt.legend()
    plt.figure()

    plt.plot(epochs, loss, 'go', label='Training Loss')
    plt.plot(epochs, val_loss, 'g', label='Validation Loss')
    plt.title('Training and Validation loss')
    plt.legend()
    plt.show()

if __name__ == '__main__':

    train_gen, validation_gen = generate_image()
    model = create_model()

    result = fit_model(model_dir, model, train_gen, validation_gen)
    print_result(result)

    scores = model.evaluate(validation_gen)
    print(scores)
