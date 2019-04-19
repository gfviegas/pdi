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
# Este arquivo se refere a atividade 3).

# Carregando as bibliotecas necessárias
import cv2
import numpy as np
import argparse

# Lê a cor escolhida pela linha de comando
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--color", default='blue',
    choices=['blue', 'yellow', 'black', 'red'],
    help="Cor que deseja analisar")
args = vars(ap.parse_args())
chosedColor = args.get('color', False)

# Segmenta a imagem pra exibir apenas cores azuis
def segmentaImagem(img, color='blue'):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if (color == 'blue'):
        # AZUL
        lower_HSV = np.array([110, 150, 20])
        upper_HSV = np.array([130, 255, 255])
    elif (color == 'yellow'):
        # AMARELO
        lower_HSV = np.array([30, 200, 190])
        upper_HSV = np.array([35, 255, 255])
    elif (color == 'black'):
        # PRETO
        lower_HSV = np.array([100, 50, 10])
        upper_HSV = np.array([130, 170, 60])
    elif (color == 'red'):
        # VERMELHO
        lower_HSV = np.array([175, 185, 100])
        upper_HSV = np.array([180, 255, 200])
    else:
        return

    # Máscara aplicando a cor escolhida
    maskHSV = cv2.inRange(hsv, lower_HSV, upper_HSV)

    # Se foi escolhido a cor preta, inverte o resultado pra melhor exibir
    if (color == 'black'):
        return cv2.bitwise_or(cv2.bitwise_not(img), img, mask=maskHSV)

    return cv2.bitwise_and(img, img, mask=maskHSV)


# Abre um vídeo gravado em disco
video = cv2.VideoCapture('videocanetas.mp4')
outputFile = 'resultados/saidaAtiv3_{}.avi'.format(chosedColor)
(sucesso, frame) = video.read()

# Criando stream de output pro video de resultado
fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
outputVideo = cv2.VideoWriter(outputFile, fourcc, 60,
                             (frame.shape[1], frame.shape[0]))

# numero = 1
while True:
    # read() retorna 1-Se houve sucesso e 2-O próprio frame
    (sucesso, frame) = video.read()
    if not sucesso:  # Final do vídeo
        break

    frame = segmentaImagem(frame, color=chosedColor)

    # Grava um frame como imagem
    # cv2.imwrite('resultados/ativ3_video{}.jpg'.format(numero), frame)
    # numero = numero+1

    outputVideo.write(frame)
    cv2.imshow("Exibindo video", frame)
    if cv2.waitKey(1) & 0xFF == ord("s"):
        break

outputVideo.release()
video.release()
cv2.destroyAllWindows()
