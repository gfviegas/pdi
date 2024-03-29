# Relatório 3 - PDI
# Bruno Marra - 3029
# Gustavo Viegas - 3026
# Heitor Passeado - 3055
#
# Descrição da Atividade:
#
# 1) Para cada imagem de cubo(10 imagens), extraia 5 valores de R,G,B,L,a,b,
# extraia e calcule a media e desvio padrão para ver a variação da matiz de cor
# em cada agrupamento. Faça isto para duas cores distintas do cubo.
# Ao final desta operação, voce vai ter , para uma cor, um total de 50 pixels
# amostrados. Calcule novamente a media e desvio padrao desta amostra, para
# verificar se ocorreu muita variação de cor. Analise principalemnte os valores
# de L,a,b. Lembre-se de fazer isto para duas cores do cubo. Utilize a lib numpy
# ou uma planilha eletronica para visualizar os resultados.
#
# 2) Abra arquivo caneta.jpg e escolha 12 pixels amostrados da cor azul, tomando
# cuidado para se extrairem pixels das pontas e meio da imagem. Ache os valores
# maximos e minimos para os pixels, e usando o script reconhecimentoHSV,
# apresente a segmentação obtida nas 4 imagens de caneta, e analise o resultado.
#
# 3) Da mesma forma que realizado no item 2, realize o mesmo procedimento para
# o filme videocaneta, realizando a segmentação para as 4 cores presentes,
# obtendo 4 filmes resultantes. Estes filmes devem ser obtidos aplicando a
# mascara de segmentação obtida, exiba o filme apresentando somente a ponta da
# caneta segmentada se movendo...
#
# 4) Da mesma forma que no item 4, gere imagens do arquivo RUN.AVI, selecione
# somente amostras de pixels dos corredores que usam camisa azul escura, e faça
# o procedimento de segmentação para exibir somente estes corredores.
#
# Este arquivo se refere a atividade 4).

# Carregando as bibliotecas necessárias
import cv2
import numpy as np

# Define range de cor azul em HSV.
# Estes valores de pixels normalmente são amostrados de imagens
# conhecidas e utilizados para segmentar imagens novas.
def segmentaImagem(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Intervalo de cores azul
    lower_HSV = np.array([100, 130, 20])
    upper_HSV = np.array([140, 255, 255])

    # Aplica máscara HSV pra pegar apenas as cores azuis nesse intervalo
    maskHSV = cv2.inRange(hsv, lower_HSV, upper_HSV)
    # Bitwise-AND mask and original image
    return cv2.bitwise_and(img, img, mask=maskHSV)


# Abre um vídeo gravado em disco
video = cv2.VideoCapture('run.mp4')
outputFile = 'resultados/saidaAtiv4.avi'
(sucesso, frame) = video.read()

# Criando stream de output pro video de resultado
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
outputVideo = cv2.VideoWriter(outputFile, fourcc, 60, (frame.shape[1],
                                                      frame.shape[0]))

# numero = 1
while True:
    # read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = video.read()
    if not sucesso:  # final do vídeo
        break

    cv2.imshow("Original", frame)
    frame = segmentaImagem(frame)

    # Grava um frame como imagem
    # cv2.imwrite('resultados/ativ4_video{}.png'.format(numero), frame)
    # numero += 1

    outputVideo.write(frame)
    cv2.imshow("Exibindo video", frame)

    # Converte para LAB e vai gravando  video em disco
    # lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)
    # outputVideo.write((lab).astype(np.uint8))

    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

outputVideo.release()
video.release()
cv2.destroyAllWindows()
