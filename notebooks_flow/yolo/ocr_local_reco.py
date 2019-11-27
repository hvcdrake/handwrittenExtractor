# Importing libraries
import sys
import argparse
from PIL import Image
import numpy as np
import datetime
import cv2
from scipy.spatial import distance
# from scipy import stats
import pandas as pd
from yolo import YOLO
import json
from os import getcwd
from os import path
from os import pardir

# Importing second model libraries
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop

# Importing db libraries
import pandas as pd
import sqlalchemy as sa
import pyodbc
import time

sys.path.append("..")
import general_utils


# construct the argument parser the unique param is the campaign id
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--campaign", required=True, help="Identificador de la campaña a procesar")
ap.add_argument("-db", "--database", required=True, help="Identificador de la campaña")
args = vars(ap.parse_args())

# Info about the params file
p = '../params/campaigns'
camp_file = general_utils.get_conf_file_of_camp(p, str(args['campaign']))
print("[INFO] Cargando archivo de parámetros: {}".format(camp_file))
dni_area = general_utils.get_param_from_file('../' + camp_file, 'dni_search_area')
traslation_dict = general_utils.get_param_from_file('../' + camp_file, 'traslation_dict')
# print('{}'.format(dni_area))

'''
 ## MARZO COMPRAS
traslation_dict = {
    'dni':{'tras_x':27,'tras_y':-44,'width':340,'height':75},
    'tel':{'tras_x':27+455,'tras_y':-44,'width':290,'height':75},
    'mail':{'tras_x':60,'tras_y':185,'width':800,'height':65},
    'nomb_ape1':{'tras_x':240,'tras_y':-180,'width':540,'height':80},
    'nomb_ape2':{'tras_x':-40,'tras_y':-105,'width':820,'height':65},
    'dir1':{'tras_x':100,'tras_y':20,'width':670,'height':65},
    'dir2':{'tras_x':-40,'tras_y':72,'width':820,'height':65},
    'distrito':{'tras_x':70,'tras_y':127,'width':800,'height':65},
    'cc':{'tras_x':200,'tras_y':240,'width':800,'height':65}
}
## CAMIONETA TOYOTA 2018
traslation_dict = {
    'nomb_ape1':{'tras_x':210,'tras_y':-140,'width':540,'height':60},
    'nomb_ape2':{'tras_x':-40,'tras_y':-82,'width':820,'height':50},
    'dni':{'tras_x':27,'tras_y':-40,'width':310,'height':55},
    'tel':{'tras_x':27+400,'tras_y':-40,'width':290,'height':55},
    'dir1':{'tras_x':100,'tras_y':7,'width':670,'height':50},
    'dir2':{'tras_x':-40,'tras_y':57,'width':820,'height':50},
    'distrito':{'tras_x':70,'tras_y':102,'width':800,'height':50},
    'mail':{'tras_x':60,'tras_y':152,'width':800,'height':50},
    'cc':{'tras_x':200,'tras_y':198,'width':800,'height':50}
}
# MUNDIAL MAMA
traslation_dict = {
    'dni':{'tras_x':27,'tras_y':-44,'width':340,'height':75},
    'tel':{'tras_x':27+455,'tras_y':-44,'width':290,'height':75},
    'mail':{'tras_x':60,'tras_y':185,'width':800,'height':65},
    'nomb_ape1':{'tras_x':240,'tras_y':-180,'width':540,'height':80},
    'nomb_ape2':{'tras_x':-40,'tras_y':-105,'width':820,'height':65},
    'dir1':{'tras_x':100,'tras_y':20,'width':670,'height':65},
    'dir2':{'tras_x':-40,'tras_y':72,'width':820,'height':65},
    'distrito':{'tras_x':70,'tras_y':127,'width':800,'height':65},
    'cc':{'tras_x':200,'tras_y':240,'width':800,'height':65}
}
'''

# Folder params
WORK_DIRECTORY = getcwd()
PARENT_DIRECTORY = path.abspath(path.join(WORK_DIRECTORY, pardir))
RUTA_AREAS = PARENT_DIRECTORY + '\\areas_result\\'
RUTA_CSV = PARENT_DIRECTORY + '\\local_result\\'
# BD params
BD_SAVE_FLAG = True
BD_USERNAME = 'sa'
BD_PASSWORD = 'Admin123'
# BD_DATABASE_NAME = 'ClienteCupon'
BD_DATABASE_NAME = args['database']
BD_HOST = '13.82.178.179,2701'
# Threshold params
ADMITED_THRESHOLD = 84.00
IMPROVEMENT_TRESHHOLD = 94.00
# Executions params
ID_CAMPANIA = str(args['campaign'])
ID_USUARIO = 1


# Setting the batch id
now = datetime.datetime.now()
dt_string = now.strftime("%Y%m%d%H%M%S")
logg = open('log_idbatch_{}.txt'.format(dt_string), "w+")


def today_date():
    '''
    utils:
    get the datetime of today
    '''
    date = datetime.datetime.now()
    date = pd.to_datetime(date)
    return date


def fix(x):
    # x = x.replace('E:\\git\\cupones_wong\\data\\escaneos\\marzo_compras_2018\\','')
    x = '\\'.join(x.split('\\')[-3:])
    x = x.replace('\\','/')
    return '/cupones/'+str(camp_file)+'/'+x


def ordering_cx(boxes_in, scores_in, classes_in):
    boxes = boxes_in.copy()
    scores = scores_in.copy()
    classes = classes_in.copy()

    cys = (boxes[:, 0] + boxes[:, 2]) / 2
    cxs = (boxes[:, 1] + boxes[:, 3]) / 2
    for i in range(boxes.shape[0]):
        for j in range(i + 1, boxes.shape[0]):
            if cxs[j] < cxs[i]:
                # Ordering x axis of centroid
                aux = cxs[j]
                cxs[j] = cxs[i]
                cxs[i] = aux
                # Ordering y axis of centroid
                aux = cys[j]
                cys[j] = cys[i]
                cys[i] = aux
                # Ordering scores
                aux = scores[j]
                scores[j] = scores[i]
                scores[i] = aux
                # Ordering classes
                aux = classes[j]
                classes[j] = classes[i]
                classes[i] = aux
                # Ordering boxes
                aux = boxes[j,:].copy()
                boxes[j,:] = boxes[i,:].copy()
                boxes[i,:] = aux.copy()
    return boxes, scores, classes, cxs, cys


def filter_near_cxs(cxs, scores):
    i=0
    total_idx = []
    delta = 9

    while i < cxs.size:
        # print('{}'.format(cxs[i]))
        j = i + 1
        acum_xs = [cxs[i]]
        acum_sco = [scores[i]]
        acum_is = [i]

        while j<cxs.size:
            if (cxs[j] - cxs[i])<=delta:
                acum_xs.append(cxs[j])
                acum_sco.append(scores[j])
                acum_is.append(j)
                i+=1
                j=i+1
            else:
                break
        i=j
        np_acum_sco = np.array(acum_sco)
        max_sco = np_acum_sco.max()
        max_xs = acum_xs[np.argmax(np_acum_sco)]
        max_is = acum_is[np.argmax(np_acum_sco)]
        # print('{}, {}, {} - Sco.Sel.={} Xs.Sel={} It. Sel={}'.format(acum_xs, acum_sco, acum_is, max_sco, max_xs, max_is))
        total_idx.append(max_is)
    # print('{}'.format(total_idx))
    return total_idx


def filter_far_from_cys(valid_ixs, cys, std, mean):
    re_valid = []
    delta = 2*(std)
    delta = 20
    for idx in valid_ixs:
        if abs(cys[idx]-mean)<delta:
            re_valid.append(idx)
    return re_valid


def dni_rect_from_ce(x, y, cv2_v, img, areas_dict, lines_dict):
    w = img.shape[1]
    h = img.shape[0]

    color = (0, 0, 255)
    thicknes = 2

    fields = []
    res_dict = {}

    for key in areas_dict.keys():
        tras_vector = np.array([areas_dict[key]['tras_x'], areas_dict[key]['tras_y']])
        point = np.array([x, y]) + tras_vector
        point_2 = point + np.array([areas_dict[key]['width'], areas_dict[key]['height']])

        point = np.where(point<=0,0,point)
        point_2[0] = w if point_2[0] >= w else point_2[0]
        point_2[1] = h if point_2[1] >= h else point_2[1]

        cv2_v.rectangle(img,(point[0],point[1]),(point_2[0],point_2[1]),color,thicknes)
        res_dict[key] = (point, point_2)
        # print('{} {}'.format(point,point_2))

    for key in lines_dict.keys():
        tras_vector = np.array([lines_dict[key]['tras_x'], lines_dict[key]['tras_y']])
        point = np.array([x, y]) + tras_vector
        point_2 = point + np.array([lines_dict[key]['width'], lines_dict[key]['height']])

        point = np.where(point<=0,0,point)
        point_2[0] = w if point_2[0]>= w else point_2[0]
        point_2[1] = h if point_2[1]>= h else point_2[1]

        # cv2_v.rectangle(img,(point[0],point[1]),(point_2[0],point_2[1]),color,thicknes)
        res_dict[key] = (point, point_2)

    return res_dict['dni'][0], res_dict['dni'][1], res_dict['tel'][0], res_dict['tel'][1]
    # return res_dict


def draw_detection(cv2_v, img, boxes, scores, classes, color, thick):
    for i in range(boxes.shape[0]):
        # print('{},{},{},{}'.format(dni_boxes[i][0],boxes[i][1],boxes[i][2],boxes[i][3]))
        # print('{}>class={},score={},box=[{},{},{},{}]'.format(i,classes[i],round(scores[i],2),int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])))
        cv2_v.rectangle(img,(int(boxes[i][1]),int(boxes[i][0])),(int(boxes[i][3]),int(boxes[i][2])),color,thick)
        cv2_v.putText(img,'{}'.format(classes[i]), (int(boxes[i][1])+2, int(boxes[i][0])+7), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

def deleting_horizontal_lines(cv2_v, frame):
    mean_c = cv2_v.adaptiveThreshold(frame, 255, cv2_v.ADAPTIVE_THRESH_MEAN_C, cv2_v.THRESH_BINARY_INV, 15, 12)
    # cv2_v.imshow("dniGrayLinesSec", dniGrayLinesSec)
    # cv2_v.imshow("mean_c", mean_c)

    ver = mean_c.copy()
    # ver_struc3 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 3))
    ver_struc4 = cv2_v.getStructuringElement(cv2_v.MORPH_RECT, (1, 4))
    # ver_struc5 = cv2.getStructuringElement(cv2.MORPH_RECT,(1, 5))

    '''
    ver33 = cv2.erode(ver.copy(),ver_struc3,iterations = 1)
    ver33 = cv2.dilate(ver33,ver_struc3,iterations = 1)
    cv2.imshow("ver33", ver33)
    '''
    ver44 = cv2_v.erode(ver.copy(), ver_struc4, iterations=1)
    ver44 = cv2_v.dilate(ver44, ver_struc4, iterations=1)
    # cv2_v.imshow("ver44", ver44)

    hor_struc4 = cv2_v.getStructuringElement(cv2_v.MORPH_RECT,(4, 1))
    ver44_close = cv2_v.morphologyEx(ver44.copy(), cv2_v.MORPH_CLOSE, hor_struc4)
    # cv2_v.imshow("ver44_close", ver44_close)
    return ver44_close

def stringfy(array):
    res = ""
    for i in range(array.size):
        res += str(array[i])
    return res

def clasification_by_cnn(frame, boxes, cv2_v):
    images = []
    boxes = np.where(boxes < 0, 0, boxes).astype(int)
    # print('Shhhappee {}'.format(frame.shape))
    delta_x = 5
    delta_y = 5

    for i in range(boxes.shape[0]):
        y_min = boxes[i][0] if (boxes[i][0])>0 else 0
        y_max = boxes[i][2]+delta_y if (boxes[i][2]+delta_y)<frame.shape[0] else frame.shape[0]
        x_min = boxes[i][1] if (boxes[i][1])>0 else 0
        x_max = boxes[i][3]+delta_x if (boxes[i][3]+delta_x)<frame.shape[1] else frame.shape[1]
        
        i_img = frame[y_min:y_max,x_min:x_max]
        i_img = cv2.resize(i_img, (28, 28), interpolation=cv2.INTER_LINEAR)
        i_img = np.stack((i_img, i_img), axis=2)
        i_img = i_img[:,:,:1]
        images.append(i_img)
        # cv2_v.imshow('i_img',i_img)
        # print('xmin:{} xmax:{} shape:{}'.format(boxes[i][1],boxes[i][3],i_img.shape))
        # print('after shape:{}'.format(i_img.shape))
        # cv2_v.waitKey(0)
        # cv2_v.destroyAllWindows()
    pre = model.predict_classes(np.array(images), verbose=1)
    # print('Shape del array: {}'.format(np.array(images).shape))
    # print('Preds: {}'.format(pre))
    return stringfy(pre)


def distance(str1, str2):
    d=dict()
    for i in range(len(str1)+1):
        d[i]=dict()
        d[i][0]=i
    for i in range(len(str2)+1):
        d[0][i] = i
    for i in range(1, len(str1)+1):
        for j in range(1, len(str2)+1):
            d[i][j] = min(d[i][j-1]+1, d[i-1][j]+1, d[i-1][j-1]+(not str1[i-1] == str2[j-1]))
    return d[len(str1)][len(str2)]


def improve_num(classes, scores, cnn_preds, threshold):
    res = ""
    for i in range(classes.shape[0]):
        if scores[i] >= threshold:
            res += str(classes[i])
        else:
            res += str(cnn_preds[i])
    return res


# Initiating YOLO model class
d = {'image': True}
yolo = YOLO(**d)

# Loading second model
CLASSES = 10
CHANNELS = 1
IMAGE_SIZE = 28
IMAGE_WIDTH, IMAGE_HEIGHT = IMAGE_SIZE, IMAGE_SIZE

# Model (0.995)
model = Sequential()

model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='Same', 
                 activation='relu',
                 input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, CHANNELS)))
model.add(Conv2D(filters=32,
                 kernel_size=(5,5),
                 padding='Same', 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.5))

model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 
                 activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3,3),padding='Same', 
                 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2), strides=(2,2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(8192, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(CLASSES, activation="softmax"))

# Model lading weights
model.load_weights("digit_clasification_model/model_backup.h5")

# compile
model.compile(optimizer=RMSprop(lr=0.0001,
                                rho=0.9,
                                epsilon=1e-08,
                                decay=0.00001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

print("-- {} : Modelos cargados correctamente".format(datetime.datetime.now()))
logg.write("-- {} : Modelos cargados correctamente".format(datetime.datetime.now())+'\n')

# Loading muestra numpy arrays
m_widths = np.genfromtxt(RUTA_AREAS + 'widths.txt')
m_heights = np.genfromtxt(RUTA_AREAS + 'heights.txt')
m_cxs = np.genfromtxt(RUTA_AREAS + 'cxs.txt')
m_cys = np.genfromtxt(RUTA_AREAS + 'cys.txt')
m_files = np.load(RUTA_AREAS + 'paths.npy')

# Loading muestra numpy arrays
areas_r = pd.read_csv(RUTA_AREAS + 'areas.csv')
areas_r = areas_r.fillna('')

print("-- {} : Areas leidas y cargadas".format(datetime.datetime.now()))
logg.write("-- {} : Areas leidas y cargadas".format(datetime.datetime.now())+'\n')

# 1 PROCESSING
# y_min = 120
# y_max = 213
# x_min = 0
# x_max = 833
y_min = dni_area['y_min']
y_max = dni_area['y_max']
x_min = dni_area['x_min']
x_max = dni_area['x_max']

res_filenames = []
res_rutas = []

res_dni_digs = []
res_dni_scores = []
res_dni_cnn = []
res_dni_boxes = []
dni_areas = []

res_cel_digs = []
res_cel_scores = []
res_cel_cnn = []
res_cel_boxes = []
cel_areas = []

print("-- {} : Empezó PROCESSING".format(datetime.datetime.now()))
logg.write("-- {} : Empezó PROCESSING".format(datetime.datetime.now()) + '\n')
# Calculando el arreglo de arhivos correctos
# checked_files = np.delete(n_paths, (np.where(n_paths==failed_files))[1])
# sample = np.take(checked_files,np.random.randint(0,checked_files.size,200))

# for i in range(m_files.size):
for i in range(areas_r.path.values.size):
    # Grabbing the file
    '''
    filename = failed_files[i][0]
    c_x = int(failed_cxs[i][0])
    c_y = int(failed_cys[i][0])
    '''

    # filename = m_files[i]
    # filename = areas_r.iloc[i, 3]
    filename = areas_r.path.values[i]
    name = filename.split('\\')
    name = name[len(name) - 1]
    name = name.split('.')[0]

    # c_x = int(m_cxs[i])
    # c_y = int(m_cys[i])
    # c_x = int(areas_r.iloc[i, 0])
    # c_y = int(areas_r.iloc[i, 1])
    c_x = int(areas_r.cx.values[i])
    c_y = int(areas_r.cy.values[i])

    # wi = m_widths[i]
    # hi = m_heights[i]
    # wi = areas_r.iloc[i, 4]
    # hi = areas_r.iloc[i, 2]
    wi = areas_r.width.values[i]
    hi = areas_r.height.values[i]

    x_minimo = int(c_x - (wi / 2))
    x_maximo = int(c_x + (wi / 2))
    y_minimo = int(c_y - (hi / 2))
    y_maximo = int(c_y + (hi / 2))

    image = cv2.imread(filename)
    frame = image.copy()
    found = False

    res_filenames.append(name)
    # res_rutas.append(areas_r.path.values[i].replace('E:\\git\\cupones_wong\\data\\escaneos\\',''))
    res_rutas.append(fix(filename))

    if (c_x == 9999 or c_y == 9999):
        # cv2.putText(image,'{}'.format('Area no detectada'), (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        print('{},{}'.format(filename, 'Ancla no encontrada'))
        found = False

        res_dni_digs.append(np.array([]))
        res_dni_scores.append(np.array([]))
        res_dni_cnn.append("")
        res_dni_boxes.append(np.array([]))
        dni_areas.append([])

        res_cel_digs.append(np.array([]))
        res_cel_scores.append(np.array([]))
        res_cel_cnn.append("")
        res_cel_boxes.append(np.array([]))
        cel_areas.append([])
    else:
        found = True
        print('{} . {}'.format(i, name), end=' / ')
        cv2.rectangle(image,(x_minimo, y_minimo + int(y_min)), (x_maximo,y_maximo+int(y_min)), (0,0,255), 1)
        dni1, dni2, cel1, cel2 = dni_rect_from_ce(c_x, c_y + y_min, cv2, image, traslation_dict, {})

        try:
            # Getting DNI area
            dniGray = cv2.cvtColor(frame[dni1[1]:dni2[1],dni1[0]:dni2[0]], cv2.COLOR_BGR2GRAY)

            # Preprocessing dni area
            dniGrayTresh = deleting_horizontal_lines(cv2, dniGray.copy())

            # Getting the model output
            pil_dni = Image.fromarray(dniGrayTresh)
            dni_img, dni_boxes, dni_scores, dni_classes = yolo.detect_image(pil_dni)

            # Ordering boxes by axis x
            boxes, scores, classes, cxs, cys = ordering_cx(dni_boxes, dni_scores, dni_classes)

            # Filtering erronues boxes
            valid_ixs = filter_near_cxs(cxs, scores)
            valid_cys = np.take(cys, valid_ixs)

            prom = valid_cys.sum()/valid_cys.size
            desv = (((valid_cys - prom)**2).sum() / valid_cys.size) ** (1/2)
            valid_ixs = filter_far_from_cys(valid_ixs,cys,desv, prom)

            b = []
            for i in valid_ixs:
                b.append(boxes[i,:].copy())
            n_boxes = np.array(b)
            n_boxes = (np.where(n_boxes < 0, 0, n_boxes)).astype(int)

            # Getting info from cnn model
            dni_est_2 = clasification_by_cnn(dniGrayTresh, n_boxes, cv2)
            dni_est_2 = improve_num(np.take(classes,valid_ixs), np.take(scores,valid_ixs), dni_est_2, IMPROVEMENT_TRESHHOLD/100)

            # Appending into arrays
            res_dni_digs.append(np.take(classes, valid_ixs))
            res_dni_scores.append(np.take(scores, valid_ixs))
            res_dni_cnn.append(dni_est_2)
            res_dni_boxes.append(n_boxes)
            dni_areas.append([dni1[0], dni2[0], dni1[1], dni2[1]])
        except Exception as e:
            print('Error DNI: {}'.format(e))
            res_dni_digs.append(np.array([]))
            res_dni_scores.append(np.array([]))
            res_dni_cnn.append("")
            res_dni_boxes.append(np.array([]))
            dni_areas.append([])

        try:
            # Getting Cel area
            celGray = cv2.cvtColor(frame[cel1[1]:cel2[1],cel1[0]:cel2[0]], cv2.COLOR_BGR2GRAY)

            # Preprocessing cel area
            celGrayTresh = deleting_horizontal_lines(cv2, celGray.copy())

            # Getting the model output
            pil_cel = Image.fromarray(celGrayTresh)
            cel_img, cel_boxes, cel_scores, cel_classes = yolo.detect_image(pil_cel)

            # Ordering boxes by axis x
            boxes, scores, classes, cxs, cys = ordering_cx(cel_boxes, cel_scores, cel_classes)

            # Filtering erronues boxes
            valid_ixs = filter_near_cxs(cxs, scores)
            valid_cys = np.take(cys, valid_ixs)
            prom = valid_cys.sum() / valid_cys.size
            desv = (((valid_cys - prom) ** 2).sum() / valid_cys.size) ** (1 / 2)

            valid_ixs = filter_far_from_cys(valid_ixs, cys, desv, prom)

            b = []
            for i in valid_ixs:
                b.append(boxes[i, :].copy())
            n_boxes = np.array(b)
            n_boxes = (np.where(n_boxes < 0, 0, n_boxes)).astype(int)

            # Getting info from cnn model
            cel_est_2 = clasification_by_cnn(celGrayTresh, n_boxes, cv2)
            cel_est_2 = improve_num(np.take(classes,valid_ixs), np.take(scores,valid_ixs), cel_est_2, IMPROVEMENT_TRESHHOLD/100)

            # Appending into arrays
            res_cel_digs.append(np.take(classes, valid_ixs))
            res_cel_scores.append(np.take(scores, valid_ixs))
            res_cel_cnn.append(cel_est_2)
            res_cel_boxes.append(n_boxes)
            cel_areas.append([cel1[0], cel2[0], cel1[1], cel2[1]])

        except Exception as e:
            print('Error CEL: {}'.format(e))
            res_cel_digs.append(np.array([]))
            res_cel_scores.append(np.array([]))
            res_cel_cnn.append("")
            res_cel_boxes.append(np.array([]))
            cel_areas.append([])

# logg.write("    -{}Procesó {}".format(datetime.datetime.now(), name) + '\n')
print("-- {} : Terminó :D. Fin".format(datetime.datetime.now()))

# Setting arrays
np_res_filenames = np.array(res_filenames)
np_res_rutas = np.array(res_rutas)

np_res_dni_digs = np.array(res_dni_digs)
np_res_dni_scores = np.array(res_dni_scores)
np_res_dni_cnn = np.array(res_dni_cnn)
np_res_dni_boxes = np.array(res_dni_boxes)

np_res_cel_digs = np.array(res_cel_digs)
np_res_cel_scores = np.array(res_cel_scores)
np_res_cel_cnn = np.array(res_cel_cnn)
np_res_cel_boxes = np.array(res_cel_boxes)


# Formatting data
acers_d = []
acers_c = []
local_jsons = []

for i in range(np_res_dni_scores.size):
    # print('{}'.format(i))
    sc_dni = np.round(np_res_dni_scores[i] * 10000).astype(int)
    sc_cel = np.round(np_res_cel_scores[i] * 10000).astype(int)

    # acer_dni = 0 if len(np_res_dni_cnn[i]) != 8 else np_res_dni_scores[i].sum()/np_res_dni_scores[i].size
    acer_dni = 0 if len(np_res_dni_cnn[i]) != 8 else np_res_dni_scores[i].min()
    # acer_cel = 0 if (len(np_res_cel_cnn[i]) > 9 or len(np_res_cel_cnn[i])<6) else np_res_cel_scores[i].sum()/np_res_cel_scores[i].size
    acer_cel = 0 if (len(np_res_cel_cnn[i]) > 9 or len(np_res_cel_cnn[i]) < 6) else np_res_cel_scores[i].min()

    acers_d.append(round(acer_dni * 100, 2))
    acers_c.append(round(acer_cel * 100, 2))

    # print('T:{}, V:{}'.format(type(dni_areas[i]), dni_areas[i]))
    dni_area = np.array(dni_areas[i]).astype(int)
    cel_area = np.array(cel_areas[i]).astype(int)

    local_dict = {
                    'dni_area': dni_area.tolist(),
                    'telefono_area': cel_area.tolist(),
                    'dni_yolo':
                        {
                        'values': np_res_dni_digs[i].tolist(),
                        'scores': sc_dni.tolist(),
                        'boxes': np_res_dni_boxes[i].tolist()},
                    'telefono_yolo':
                        {
                        'values': np_res_cel_digs[i].tolist(),
                        'scores': sc_cel.tolist(),
                        'boxes': np_res_cel_boxes[i].tolist()},
                    'threshold': int(IMPROVEMENT_TRESHHOLD * 100),
                    'dni_final':{'value': np_res_dni_cnn[i]},
                    'telefono_final':{'value': np_res_cel_cnn[i]}
                 }
    local_jsons.append(json.dumps(local_dict))


# Building data frame
bd_cupones = pd.DataFrame({'NombreArchivo':np_res_filenames})
# bd_cupones['idCupon'] = np.arange(1000,1000+np_res_filenames.size,1)
bd_cupones['DNI'] = np_res_dni_cnn
bd_cupones['AcertividadDNI'] = np.array(acers_d)
bd_cupones['Telefono'] = np_res_cel_cnn
bd_cupones['AcertividadTelefono'] = np.array(acers_c)
bd_cupones['idCampania'] = ID_CAMPANIA
bd_cupones['idUsuario'] = ID_USUARIO
bd_cupones['idEstado'] = 1
bd_cupones['idBatch'] = int(dt_string)
bd_cupones['LocalJsonOCR'] = np.array(local_jsons)
bd_cupones['Ruta'] = np.array(np_res_rutas)

# Corrección de tamaño de campos
bd_cupones['DNI'] = bd_cupones['DNI'].apply(lambda x: x[:20])
bd_cupones['Telefono'] = bd_cupones['Telefono'].apply(lambda x: x[:20])
bd_cupones['LocalJsonOCR'] = bd_cupones['LocalJsonOCR'].apply(lambda x: x[:8000])


# engine = sa.create_engine('mssql+pyodbc://usercupon:123456789@192.168.2.55/ClienteCupon?driver=SQL+Server+Native+Client+11.0')
conn_str = 'mssql+pyodbc://' + BD_USERNAME + ':' + BD_PASSWORD + '@'
conn_str += BD_HOST + '/' + BD_DATABASE_NAME
conn_str += '?driver=SQL+Server+Native+Client+11.0'
engine = sa.create_engine(conn_str)

if BD_SAVE_FLAG:
    # Inserción a la tabla cupon
    t0 = time.time()
    bd_cupones.to_sql('Cupon', engine, if_exists='append', index=False, chunksize=200)
    # print(f"Inserción Cupon finalizada en {time.time() - t0:.1f} seconds")
    print("Inserción Cupon finalizada en {:.1f} seconds".format(time.time()-t0))
    logg.write("-- {} : Fin escritura en base de datos".format(datetime.datetime.now(), name)+'\n')
    # Getting the ids from the database
    time.sleep(1)
    sql = 'select c.idCupon,c.NombreArchivo from ' + BD_DATABASE_NAME + '.dbo.Cupon c where c.idBatch={}'.format(dt_string)
    ids_cupon = pd.read_sql_query(sql, engine)

    bd_cupones = pd.merge(ids_cupon, bd_cupones, how='left', on='NombreArchivo')
else:
    print("BD_SAVE_FLAG set in False para Cupon")


# Setting azure flag for th next process
bd_cupones['FechaHora'] = today_date()
bd_cupones['Azure'] = np.where(
    np.logical_and(
        np.logical_and(bd_cupones['DNI'].apply(len) == 8, bd_cupones.AcertividadDNI>=ADMITED_THRESHOLD),
        np.logical_and(
            np.logical_and(bd_cupones['Telefono'].apply(len)<=9,bd_cupones['Telefono'].apply(len)>=6),
            bd_cupones.AcertividadTelefono>=ADMITED_THRESHOLD
            )
        ),
    1,
    1)


bd_cupones.to_csv(RUTA_CSV + 'result.csv', encoding='utf-8', index=False)
np.save(RUTA_AREAS + 'azure_flags', bd_cupones['Azure'].values)
print("-- {} : Fin escritura del archivo".format(datetime.datetime.now()))

if BD_SAVE_FLAG:
    # Inserción a la tabla logcupon
    bd_cupones = bd_cupones[bd_cupones.Azure == 1].iloc[:, :-1]
    t0 = time.time()
    bd_cupones.to_sql('Logcupon', engine, if_exists='append', index=False, chunksize=200)
    # print(f'Inserción Logcupon finalizada en {time.time() - t0:.1f} seconds')
    print("Inserción Logcupon finalizada en {:.1f} seconds".format(time.time() - t0))
else:
    print("BD_SAVE_FLAG set in False para Logcupon")

logg.write("-- {} : Fin escritura del archivo".format(datetime.datetime.now(), name) + '\n')
logg.close()
