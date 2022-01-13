# 경로 설정
import os
base_dir = '/Users/jieun/Desktop/softdrink_classifier'

image_dir = os.path.join(base_dir, 'datasets/drinks')
train_dir = os.path.join(image_dir, 'train')
validation_dir = os.path.join(image_dir, 'validation')
test_dir = os.path.join(image_dir, 'test')
video_dir = os.path.join(base_dir, 'video/')
new_img_dir = os.path.join(image_dir, 'newimg/')

model_dir = os.path.join(base_dir, 'model')
new_model_dir = os.path.join(base_dir, 'new_model')

# 모델 훈련에 필요한 변수
classes = os.listdir(train_dir)
classes = sorted(classes)
#print(classes)

if '.DS_Store' in classes:
    classes.remove('.DS_Store')

CLASSES = len(classes)

IMAGE_SIZE = 150
BATCH_SIZE = 100
EPOCHS = 100
STEPS_PER_EPOCH = CLASSES
VALIDATION_STEPS = 2
