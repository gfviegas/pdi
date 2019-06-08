# -*- coding: utf-8 -*-
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
# Este arquivo se refere a atividade 1).

# Carregando as bibliotecas necessárias
import cv2,argparse,glob
import numpy as np

# Calcula estatísticas de média e desvio padrao de cada banda (BGR, LAB)
# pra uma amostra dada.
def calculaStats(amostra):
    amostraSqueezed = np.squeeze(amostra)
    amostraBGR = [a[0] for a in amostra]
    amostraLAB = [a[1] for a in amostra]

    # BGR
    amostraBGR = np.squeeze(amostraBGR)
    B = [c[0] for c in amostraBGR]
    G = [c[1] for c in amostraBGR]
    R = [c[2] for c in amostraBGR]

    print('')
    print("*** Azul: ***")
    print("Media: {}, Desvio: {}".format(np.mean(B), np.std(B)))

    print("*** Verde: ***")
    print("Media: {}, Desvio: {}".format(np.mean(G), np.std(G)))

    print("*** Vermelho: ***")
    print("Media: {}, Desvio: {}".format(np.mean(R), np.std(R)))

    # LAB
    amostraLAB = np.squeeze(amostraLAB)
    L = [c[0] for c in amostraLAB]
    A = [c[1] for c in amostraLAB]
    B = [c[2] for c in amostraLAB]

    print("*** L: ***")
    print("Media: {}, Desvio: {}".format(np.mean(L), np.std(L)))

    print("*** A: ***")
    print("Media: {}, Desvio: {}".format(np.mean(A), np.std(A)))

    print("*** B: ***")
    print("Media: {}, Desvio: {}".format(np.mean(B), np.std(B)))
    print('')

# Função de callback do mouse
def showPixelValue(event,x,y,flags,param):
    global img, combinedResult, placeholder, amostra
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = img[y,x]
        lab = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2Lab)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]

        # Limita o tamanho da amostra de cada imagem em 5 cliques apenas.
        if (len(amostraImgAtual) < 5):
            amostraImgAtual.append([bgr, lab])
            amostraTotal.append([bgr, lab])

    if event == cv2.EVENT_MOUSEMOVE:
        bgr = img[y,x]

        lab = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2Lab)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]

        # Cria um placeholder vazio pra mostrar outros formatos de cores
        placeholder = np.zeros((img.shape[0],400,3),dtype=np.uint8)

        # Preenche o placeholder
        cv2.putText(placeholder, "BGR {}".format(bgr), (20, 70),
                    cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "HSV {}".format(hsv), (20, 140),
                    cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "LAB {}".format(lab), (20, 280),
                    cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)

        # Combina os resultados e os mostra
        combinedResult = np.hstack([img,placeholder])
        cv2.imshow(msg, combinedResult)


if __name__ == '__main__' :
    global img, msg, amostraImgAtual, amostraTotal
    amostraImgAtual = list()
    amostraTotal = list()

    #files = glob.glob('images/rub*.jpg')
    #files.sort()
    img = cv2.imread("MOEDA_VALIOSA.jpg")
    img = cv2.resize(img,(400,400))
    msg = "Pressione  A para Anterior, P para proxima imagem, X pra computar"
    cv2.imshow(msg, img)

    cv2.namedWindow(msg)
    cv2.setMouseCallback(msg, showPixelValue)

    i = 0
    while(1):
        k = cv2.waitKey(1) & 0xFF
        if k == ord('p'):
            i += 1
            print("Stats da imagem {} que possui {} amostras"
                  .format(i, len(amostraImgAtual)))
            calculaStats(amostraImgAtual)
            amostraImgAtual = list()
            img = cv2.imread(files[i%len(files)])
            img = cv2.resize(img,(400,400))
            cv2.imshow(msg, img)
        elif k == ord('a'):
            i -= 1
            print("Stats da imagem {} que possui {} amostras"
                  .format(i, len(amostraImgAtual)))
            calculaStats(amostraImgAtual)
            amostraImgAtual = list()
            img = cv2.imread(files[i%len(files)])
            img = cv2.resize(img,(400,400))
            cv2.imshow(msg, img)
        elif k == ord('x'): # Ao apertar X, mostra estatísticas total das amostras coletadas
            print("Stats do total da amostra de {} cores.".format(len(amostraTotal)))
            calculaStats(amostraTotal)
        elif k == 27:
            cv2.destroyAllWindows()
            break
