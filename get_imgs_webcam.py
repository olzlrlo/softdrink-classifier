import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
from set_vars import new_dir

# 웹캠 열기
webcam = cv2.VideoCapture(0)
if not webcam.isOpened():
    print("Failed to open webcam")
    exit()

sample_num, captured_num = 0, 0

while webcam.isOpened():

    status, frame = webcam.read()
    sample_num = sample_num + 1

    if not status:
        break

    cv2.imshow("captured frames", frame)
    bbox, object, confidence = cv.detect_common_objects(frame)
    out = draw_bbox(frame, bbox, object, confidence, write_conf = True)
    cv2.imshow("captured frames", out)

    #  # loop through detected objects
    for b, obj in zip(bbox, object):
        if obj != 'bottle' and obj != 'cup':
            break

        print(b)
        print(obj)
        (startX, startY) = b[0], b[1]
        (endX, endY) = b[2], b[3]

        captured_num = captured_num + 1
        object_in_img = frame[startY:endY, startX:endX, :]
        cv2.imwrite(new_dir + str(captured_num) + '.jpeg', object_in_img)
        print(captured_num)

    # 종료 조건
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

webcam.release()
cv2.destroyAllWindows()
