import cv2
import numpy as np
#Abre um vídeo gravado em disco
camera = cv2.VideoCapture('run.mp4')
#Também é possível abrir a próprio webcam
#do sistema para isso segue código abaixo
#camera = cv2.VideoCapture(0)
outputFile = 'saida.avi'
(sucesso, frame) = camera.read()
vid_writer = cv2.VideoWriter(outputFile, cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame.shape[1],frame.shape[0]))
numero=1    
while True:
    #read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = camera.read()
    if not sucesso: #final do vídeo
        break
# grava um frame como imagem
    cv2.imwrite('nome'+str(numero)+'.jpg',frame)
    numero=numero+1


# converte para L*a*b   e vai gravando  video em disco
    lab= cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
    vid_writer.write((lab).astype(np.uint8))



   
    #converte para tons de cinza
    #frame_pb = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Exibindo video", frame)
    #cv2.waitKey(0)
    #Espera que a tecla 's' seja pressionada para sair
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break
vid_writer.release()
cv2.destroyAllWindows()


