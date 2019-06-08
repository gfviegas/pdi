import cv2

import numpy as np

def writeLetter(letter):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height) = cv2.getTextSize(letter, font, 6, thickness=1)[0]
    text_offset_x = 10
    text_offset_y = frame.shape[0] - 25
    box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width - 2, text_offset_y - text_height - 2))

    cv2.rectangle(frame, box_coords[0], box_coords[1], (0,0,0), cv2.FILLED)
    cv2.putText(frame, letter, (text_offset_x, text_offset_y), font, 6, (255,255,255), 3)
    return

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
        writeLetter("B")
        cv2.imshow("Imagem capturada", frame)

      # TODO
      # mandar o frame para a função de avaliação

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break;

cap.release()
cv2.destroyAllWindows()
