import cv2

import numpy as np

cap = cv2.VideoCapture(0)
framerate = cap.get(cv2.CAP_PROP_FPS)
framecount = 0

while(1):
    ret, frame=cap.read()
    frame = cv2.flip(frame, 1)
    framecount += 1

    # Check if this is the frame closest to 5 seconds
    if framecount == (framerate * 5):
        framecount = 0
        cv2.imshow("Imagem capturada", frame)

      # TODO
      # mandar o frame para a função de avaliação

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break;

cap.release()
cv2.destroyAllWindows()
