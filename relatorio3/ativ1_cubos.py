import cv2,argparse,glob
import numpy as np

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


# mouse callback function
def showPixelValue(event,x,y,flags,param):
    global img, combinedResult, placeholder, amostra
    if event == cv2.EVENT_LBUTTONDOWN:
        bgr = img[y,x]
        ycb = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2YCrCb)[0][0]
        lab = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2Lab)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]

        # Limita o tamanho da amostra de cada imagem em 5 cliques apenas.
        if (len(amostraImgAtual) < 5):
            amostraImgAtual.append([bgr, lab])
            amostraTotal.append([bgr, lab])

    if event == cv2.EVENT_MOUSEMOVE:
        # get the value of pixel from the location of mouse in (x,y)
        bgr = img[y,x]


        # Convert the BGR pixel into other colro formats
        ycb = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2YCrCb)[0][0]
        lab = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2Lab)[0][0]
        hsv = cv2.cvtColor(np.uint8([[bgr]]),cv2.COLOR_BGR2HSV)[0][0]

        # Create an empty placeholder for displaying the values
        placeholder = np.zeros((img.shape[0],400,3),dtype=np.uint8)

        # fill the placeholder with the values of color spaces
        cv2.putText(placeholder, "BGR {}".format(bgr), (20, 70), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "HSV {}".format(hsv), (20, 140), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "YCrCb {}".format(ycb), (20, 210), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)
        cv2.putText(placeholder, "LAB {}".format(lab), (20, 280), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)

        # Combine the two results to show side by side in a single image
        combinedResult = np.hstack([img,placeholder])

        cv2.imshow(msg, combinedResult)


if __name__ == '__main__' :

    # load the image and setup the mouse callback function
    global img, msg, amostraImgAtual, amostraTotal
    amostraImgAtual = list()
    amostraTotal = list()

    files = glob.glob('images/rub*.jpg')
    files.sort()
    img = cv2.imread(files[0])
    img = cv2.resize(img,(400,400))
    msg = "Pressione  A para Anterior, P para proxima imagem, X pra computar"
    cv2.imshow(msg, img)

    # Create an empty window
    cv2.namedWindow(msg)
    # Create a callback function for any event on the mouse
    cv2.setMouseCallback(msg, showPixelValue)
    i = 0
    while(1):
        k = cv2.waitKey(1) & 0xFF
        # check next image in the folder
        if k == ord('p'):
            i += 1
            print("Stats da imagem {} que possui {} amostras".format(i, len(amostraImgAtual)))
            calculaStats(amostraImgAtual)
            amostraImgAtual = list()
            img = cv2.imread(files[i%len(files)])
            img = cv2.resize(img,(400,400))
            cv2.imshow(msg, img)

        # check previous image in folder
        elif k == ord('a'):
            i -= 1
            print("Stats da imagem {} que possui {} amostras".format(i, len(amostraImgAtual)))
            calculaStats(amostraImgAtual)
            amostraImgAtual = list()
            img = cv2.imread(files[i%len(files)])
            img = cv2.resize(img,(400,400))
            cv2.imshow(msg, img)

        elif k == ord('x'):
            print("Stats do total da amostra de {} cores.".format(len(amostraTotal)))
            calculaStats(amostraTotal)

        elif k == 27:
            cv2.destroyAllWindows()
            break
