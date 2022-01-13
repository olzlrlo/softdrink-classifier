import cv2
from set_vars import classes, video_dir, new_img_dir, IMAGE_SIZE

for label in classes:
    if label == 'none':
        continue

    captured = cv2.VideoCapture(video_dir + label + '.mp4')
    count = 0

    while(count < 200):
        _, image = captured.read()
        #image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))

        # 2 frame 당 image 하나
        if(int(captured.get(1)) % 2 == 0):
            cv2.imwrite(new_img_dir + label + '/' + label + str(count) + '.jpeg', image)
            count += 1

    print(label + " finished")

captured.release()
