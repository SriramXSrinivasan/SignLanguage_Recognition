import cv2
import os
import time
import uuid

imagespath = '.\\data'
labels = ['hello', 'thanks', 'good', 'home', 'meet']
number_imgs = 50

for label in labels:
    # Using os.makedirs to create parent directories if they don't exist
    os.makedirs(os.path.join(imagespath, label), exist_ok=True)
    
    # Correcting the typo in 'VideoCapture'
    cap = cv2.VideoCapture(0)
    
    print('Collecting images for {}'.format(label))
    time.sleep(5)
    
    for imgnum in range(number_imgs):
        ret, frame = cap.read()
        imagename = os.path.join(imagespath, label, '{}.jpg'.format(str(uuid.uuid1())))
        cv2.imwrite(imagename, frame)
        cv2.imshow('frame', frame)
        time.sleep(2)

        # Correcting the typo in 'waitKey'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()

cv2.destroyAllWindows()