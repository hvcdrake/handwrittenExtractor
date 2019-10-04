# import libraries
import cv2
import os
from PIL import Image
import datetime
import numpy as np
import datetime
import math


WORK_DIRECTORY = os.getcwd()
INPUT = True
# ruta_input = 'C:\\git\\cuponesWong\\CuponesWong\\notebooks_flow\\input'
# ruta_campania = 'C:\\git\\cuponesWong\\CuponesWong\\data\\escaneos\\marzo_compras_2018'
ruta_input = WORK_DIRECTORY + '\\input'
ruta_campania = 'C:\\git\\cuponesWong\\CuponesWong\\data\\escaneos\\marzo_compras_2018'

# SIFT initialization
detector = cv2.xfeatures2d.SIFT_create()
FLANN_INDEX_KDITREE = 0
flannParam = dict(algorithm=FLANN_INDEX_KDITREE, tree=5)
flann = cv2.FlannBasedMatcher(flannParam, {})
trainImg = cv2.imread("sift_img/dni5.jpg", 0)
trainKP, trainDesc = detector.detectAndCompute(trainImg, None)
MIN_MATCH_COUNT = 20


if INPUT:
    # 1 Getting files from inputh
    # ruta = 'C:\\git\\cuponesWong\\CuponesWong\\data\\muestra1800'
    files = [arch.name for arch in os.scandir(ruta_input) if arch.is_file()]
    files_lista = []

    for i in files:
        files_lista.append(ruta_input+'\\'+i)

    print("Terminó. {}".format(len(files_lista)))
else:
    # 1 Getting files in multiple paths
    rutas_personas  = [arch.name for arch in os.scandir(ruta_campania) if arch.is_dir()]
    total = 0

    files_lista = []

    for persona in rutas_personas:
        bolsas  = [arch.name for arch in os.scandir(ruta_campania + "\\" + persona) if arch.is_dir()]
        for bolsa in bolsas:
            files = [arch.name for arch in os.scandir(ruta_campania + "\\" + persona + "\\" + bolsa) if arch.is_file()]
            for file in files:
                files_lista.append(ruta_campania + "\\" + persona + "\\" + bolsa + "\\" + file)
    print("Terminó. {}".format(len(files_lista)))


# 1 Extracción del area de forma estática
y_min = 120
y_max = 213
x_min = 0
x_max = 833

falses = 0

# array's
h_lista = []
w_lista = []
p_lista = []
c_x_lista = []
c_y_lista = []
file_lista = []
paths_lista = []

print("Inicio {}".format(datetime.datetime.now()))
for filepath in files_lista:
    # Initializing a boolean variable with False
    dni_founded = False

    # Preprocessing section
    image = cv2.imread(filepath)
    print('Reading {}'.format(filepath))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift_roi = gray[y_min:y_max, x_min:x_max]
    sift_roi = cv2.adaptiveThreshold(sift_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 11)

    # SIFT DETECTION
    queryKP, queryDesc = detector.detectAndCompute(sift_roi, None)
    matches = flann.knnMatch(queryDesc, trainDesc, k=2)
    detection = False

    goodMatch = []
    for m, n in matches:
        if(m.distance < 0.75 * n.distance):
            goodMatch.append(m)
    if(len(goodMatch) > MIN_MATCH_COUNT):
        tp = []
        qp = []
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp, qp = np.float32((tp, qp))
        H, status = cv2.findHomography(tp, qp, cv2.RANSAC, 3.0)
        h, w = trainImg.shape
        trainBorder = np.float32([[[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]])
        queryBorder = cv2.perspectiveTransform(trainBorder, H)
        cv2.polylines(sift_roi, [np.int32(queryBorder)], True, (0, 255, 0), 3)
        dni_founded = True

    if dni_founded:
        # -------Comparacion de Hmax y Wmax-----
        # restringinedo valores a 0
        queryBorder = np.where(queryBorder < 0, 0, queryBorder)

        # Coordenadas "query"
        x0 = queryBorder[[[0],0]][[0],0]
        y0 = queryBorder[[[0],0]][[0],1]

        x1 = queryBorder[[[0],1]][[0],0]
        y1 = queryBorder[[[0],1]][[0],1]

        x2 = queryBorder[[[0],2]][[0],0]
        y2 = queryBorder[[[0],2]][[0],1]

        x3 = queryBorder[[[0],3]][[0],0]
        y3 = queryBorder[[[0],3]][[0],1]

        arr_x = np.array([x0, x1, x2, x3])
        x_maximo = arr_x.max()
        x_minimo = arr_x.min()
        c_x = int(arr_x.sum() / arr_x.size)

        arr_y = np.array([y0, y1, y2, y3])
        y_maximo = arr_y.max()
        y_minimo = arr_y.min()
        c_y = int(arr_y.sum() / arr_y.size)

        h_lista.append(y_maximo - y_minimo)
        w_lista.append(x_maximo - x_minimo)
        p_lista.append([(x0,y0), (x1,y1), (x2,y2), (x3,y3)])
        c_x_lista.append(c_x)
        c_y_lista.append(c_y)
        paths_lista.append(filepath)
    else:
        falses += 1
        h_lista.append(9999)
        w_lista.append(9999)
        p_lista.append([(0,0,),(0,0,),(0,0,),(0,0,)])
        c_x_lista.append(9999)
        c_y_lista.append(9999)
        paths_lista.append(filepath)
    # cv2.imshow("Sift_Roi", sift_roi)
    # cv2.imshow("Gray", gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
print("Terminó :D. Fin:{}. Total:{}".format(datetime.datetime.now(), falses))

# convirtiendo a nunmpy arrays
n_widths = np.array(w_lista)
n_heights = np.array(h_lista)
n_points = np.array(p_lista)
n_cxs = np.array(c_x_lista)
n_cys = np.array(c_y_lista)
n_paths = np.array(paths_lista)

# Saving nunmpy arrays
np.savetxt('areas_result/widths.txt', n_widths)
np.savetxt('areas_result/heights.txt', n_heights)
np.save('areas_result/points', n_points)
np.savetxt('areas_result/cxs.txt', n_cxs)
np.savetxt('areas_result/cys.txt', n_cys)
np.save('areas_result/paths', n_paths)
