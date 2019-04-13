import cv2
import numpy as np

# define range of blue color in HSV e RGB
# estes valores de pixels normalmente são amostrados de imagens
# conhecidas e utilizados para segmentar imagens novas.


def segmentaImagem(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # AZUL
    lower_HSV = np.array([110, 130, 20])
    upper_HSV = np.array([140, 255, 255])

    # Threshold the HSV image to get only blue colors
    maskHSV = cv2.inRange(hsv, lower_HSV, upper_HSV)
    # Bitwise-AND mask and original image
    return cv2.bitwise_and(img, img, mask=maskHSV)


# Abre um vídeo gravado em disco
camera = cv2.VideoCapture('run.mp4')
# Também é possível abrir a próprio webcam
# do sistema para isso segue código abaixo
#camera = cv2.VideoCapture(0)
outputFile = 'saida.avi'
(sucesso, frame) = camera.read()
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc(
    'M', 'J', 'P', 'G'), 60, (frame.shape[1], frame.shape[0]))
numero = 1

while True:
    # read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = camera.read()
    if not sucesso:  # final do vídeo
        break

    frame = segmentaImagem(frame)
    # grava um frame como imagem
    #cv2.imwrite('./video/nome'+str(numero)+'.jpg', frame)
    numero = numero+1

    # converte para L*a*b   e vai gravando  video em disco
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
    vid_writer.write((lab).astype(np.uint8))

    cv2.imshow("Exibindo video", frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

vid_writer.release()
cv2.destroyAllWindows()
