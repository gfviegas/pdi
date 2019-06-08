import cv2

import numpy as np

cap = cv2.VideoCapture(0)
while(1):
    ret, frame=cap.read()
    frame = cv2.flip(frame, 1)

    # TODO
    # mandar o frame para a função de avaliação

    cv2.imshow("Original", frame)
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break;

cap.release()
cv2.destroyAllWindows()
