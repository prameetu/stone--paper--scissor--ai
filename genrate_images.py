import cv2 as cv
import os
import sys

try:
    label_name = sys.argv[1]
    num_samples = int(sys.argv[2])
except:
    print('Arguments are missing')
    exit(-1)

parent_path = 'Image_Data'

class_path = os.path.join(parent_path, label_name)

try:
    os.mkdir(parent_path)
except FileExistsError:
    pass

try:
    os.mkdir(class_path)
except FileExistsError:
    print(f'{class_path} directory already exists')

cam = cv.VideoCapture(0)

start = False
cnt = 0

while True:
    
    ret, frame = cam.read()
    
    if not ret:
        continue
    if cnt == num_samples:
        break

    cv.rectangle(frame, (25, 100), (325, 400), (0,255,0), 2)

    if start:
        img = frame[100:400, 25:325]
        saving_path = os.path.join(class_path, f'{label_name}_{cnt + 1}.jpg')
        cv.imwrite(saving_path, img)
        cnt = cnt + 1

    font = cv.FONT_HERSHEY_TRIPLEX

    cv.putText(frame, f'Collecting Images for {label_name}',(5, 50), font, 0.5, (0,0,0), 1,cv.LINE_AA)
    cv.putText(frame, f'Image Counter :{cnt}',(5, 70), font, 0.3, (0,0,0), 1,cv.LINE_AA)

    cv.imshow('Collecting Images ', frame)

    process = cv.waitKey(10)

    if process == ord('a'):
        start = not start
    if process == ord('q'):
        break 

print(f'\n {cnt} images saved to {class_path}')

cam.release()
cv.destroyAllWindows()
    
 