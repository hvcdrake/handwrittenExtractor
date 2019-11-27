# Import section
import cv2
import numpy as np
import json
import pandas as pd
import sqlalchemy as sa
import pyodbc
import time
from os import getcwd
import datetime

import argparse
import general_utils
from paralel_db_send import multi_db_send


# construct the argument parser the unique param is the campaign id
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--campaign", required=True, help="Identificador de la campaña")
args = vars(ap.parse_args())

# Info about the params file
p = 'params/campaigns'
camp_file = general_utils.get_conf_file_of_camp(p, str(args['campaign']))
print("[INFO] Cargando archivo de parámetros: {}".format(camp_file))
dni_area = general_utils.get_param_from_file(camp_file, 'dni_search_area')
traslation_dict = general_utils.get_param_from_file(camp_file, 'traslation_dict')
lines_dict = general_utils.get_param_from_file(camp_file, 'lines_dict')

# Folder params
WORK_DIRECTORY = getcwd()
ARRAYS_PATH = WORK_DIRECTORY + '\\areas_result\\'
TMP_PATH = WORK_DIRECTORY + '\\temps\\'
LOCAL_PATH = WORK_DIRECTORY + '\\local_result\\'
TMP_FAILED_PATH = TMP_PATH + '\\reproc\\'
# Execution params
ID_CAMPANIA = int(args['campaign'])
ID_USUARIO = 1
# BD params
BD_SAVE_FLAG = True
BD_USERNAME = 'usercupon'
BD_PASSWORD = '123456789'
BD_DATABASE_NAME = 'ClienteCupon'
# BD_DATABASE_NAME = 'DevClienteCupon'
BD_HOST = '192.168.2.55'

eqs = {
    'Normal': 90.00,
    'Low': 60.00
}

banned_field_dict = ['DNI:','DNI','DN']+['Telefono:','Telefono']+['E-mail:']+['Nombres y Apellidos:']+['Direccion:']+['Distrito:']+['Centro Comercial:']
n_eqs = {
    '%': '90',
    ')': '1',
    ',': '',
    'S': '5',
    '$':'5','f':'7','&':'6','+':'7','g':'9','l':'1','o':'0','O':'0','y':'4','/':'1','.':'','-':''
    }


def center_of(bounding_box):
    xs = np.array([bounding_box[0],bounding_box[2],bounding_box[4],bounding_box[6]])
    ys = np.array([bounding_box[1],bounding_box[3],bounding_box[5],bounding_box[7]])
    return int(xs.sum() / xs.size) , int(ys.sum() / ys.size)


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
        point_2[0] = w if point_2[0]>= w else point_2[0]
        point_2[1] = h if point_2[1]>= h else point_2[1]

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

    # return res_dict['dni'][0], res_dict['dni'][1], res_dict['tel'][0], res_dict['tel'][1]
    return res_dict


def is_in_box(center, top_left_point, bottom_right_point):
    x = center[0] > top_left_point[0] and center[0] < bottom_right_point[0]
    y = center[1] > top_left_point[1] and center[1] < bottom_right_point[1]
    return (x and y)


def get_fields_from_json(azure_json, areas_dict):
    fields = {}
    scores = {}
    j=0

    # Setting empty arrays in result
    for key in areas_dict.keys():
        fields[key] = []
        scores[key] = []

    for line in azure_json['recognitionResult']['lines']:
        j += 1
        # print('Line {}: {},{}'.format(j,line['text'],line['boundingBox']))

        # Drawing areas
        cv2.rectangle(image,(line['boundingBox'][0],line['boundingBox'][1]),(line['boundingBox'][2],line['boundingBox'][5]),(0,0,0),1)
        cv2.circle(image,center_of(line['boundingBox']), 4, (0,255,255), -1)

        # Searching in areas_dict
        c = center_of(line['boundingBox'])
        t = line['text']
        ws = []
        scos = []
        for w in line['words']:
            ws.append(w['text'])
            try:
                scos.append(w['confidence'])
            except:
                scos.append('Normal')

        for key in areas_dict.keys():
            # print('C {} está en {}?'.format(c, areas_dict[key]))
            if is_in_box(c, areas_dict[key][0], areas_dict[key][1]):
                # print('String: {} --> {}'.format(t,key))
                fields[key].append(ws)
                scores[key].append(scos)

    return fields, scores


def basic_digit_clean(string, num_eqs):
    for k in num_eqs.keys():
        string = string.replace(k, num_eqs[k])
    return string


def basic_field_clean(string):
    a = string.find(':')
    if a == len(string)-1:
        return string[:-1]
    else:
        if a >= 0:
            return string[a + 1:]
        return string


def post_num_field(l_field, l_score, banned, num_eqs={}, separator=""):
    value = ""
    score = 0.0
    ct = 0

    for i in range(len(l_field)):
        for j in range(len(l_field[i])):
            # print(j)
            if l_field[i][j] not in banned:
                ct += 1
                value += separator + l_field[i][j]
                # print('{}'.format(l_score[i][j]))
                score += eqs[l_score[i][j]]
    value = basic_digit_clean(value, num_eqs)
    return (value.strip(), score / ct) if ct > 0 else (value, 0.0)


def get_dni_from_line(string, num_eqs):
    a = string.find('DNI:')
    b = string.find('Telefono:')
    if a >= 0 and b >= 0:
        return basic_digit_clean(string[a+len('DNI:'):b],num_eqs)
    a = string.find('DNI')
    b = string.find('Telefono')
    if a >= 0 and b >= 0:
        return basic_digit_clean(string[a+len('DNI'):b],num_eqs)
    else:
        return ""


def get_tel_from_line(string, num_eqs):
    b = string.find('Telefono:')
    if b >= 0:
        return basic_digit_clean(string[b+len('Telefono:'):],num_eqs)
    b = string.find('Telefono')
    if b >= 0:
        return basic_digit_clean(string[b+len('Telefono'):],num_eqs)
    else:
        return ""


def clean_letters(word):
    ret = ""
    for c in word:
        if '0' <= c <= '9':
            ret += str(c)
    return ret


def has_letters(word):
    ret = True
    for c in word:
        if not '0' <= c and c <= '9':
            return not False
    return not True

def digits(word):
    for c in word:
        if not c.isdigit():
            return False
    return True


# Loading muestra numpy arrays
areas_r = pd.read_csv(ARRAYS_PATH + 'areas.csv')
areas_r = areas_r.fillna('')
print('Total cupones area: {}'.format(areas_r.path.values.size))

# Getting all the processed files
import os
filenames = [arch.name for arch in os.scandir(TMP_FAILED_PATH) if arch.is_file()]
files = []
jsons = []

for f in filenames:
    if 'n_files' in f:
        file = np.load(TMP_FAILED_PATH + f, allow_pickle=True)
        files.append(file)
    elif 'n_jsons' in f:
        azu = np.load(TMP_FAILED_PATH + f, allow_pickle=True)
        jsons.append(azu)
    else:
        print('->sin caso')

# Getting reproc
files2 = []
jsons2 = []
for i in range(10):
    file = np.load(TMP_PATH + 'n_files_{}.npy'.format(i), allow_pickle=True)
    azu = np.load(TMP_PATH + 'n_jsons_{}.npy'.format(i), allow_pickle=True)
    if file.size > 0 and azu.size > 0:
        files2.append(file)
        jsons2.append(azu)


azure_files = np.concatenate(files2[:]+files[:], axis=0)
azure_jsons = np.concatenate(jsons2[:]+jsons[:], axis=0)
print('Resultado de azure: {}  {}'.format(azure_files.size, azure_jsons.size))

# Merging the areas result with the azure result
proc = pd.DataFrame({'path': azure_files, 'azure_json': azure_jsons})
proc = pd.merge(areas_r, proc, how='left', on='path')
print('Total cupones merge: {}'.format(proc.path.values.size))

# Processing
y_min = dni_area['y_min']
y_max = dni_area['y_max']
x_min = dni_area['x_min']
x_max = dni_area['x_max']

filenames = []

dnis = []
tels = []
mails = []
nombres = []
dirs = []
distris = []
ccs = []

sco_dnis = []
sco_tels = []
sco_mails = []
sco_nombres = []
sco_dirs = []
sco_distris = []
sco_ccs = []

azure_str_jsons = []

# for i in range(azure_files.size):
for i in range(proc.path.values.size):
    # Grabbing the file
    # filename = m_files[i]
    # filename = azure_files[i]
    filename = proc.path.values[i]
    name = filename.split('\\')
    name = name[len(name) - 1]
    name = name.split('.')[0]

    c_x = int(proc.cx.values[i])
    c_y = int(proc.cy.values[i])

    wi = proc.width.values[i]
    hi = proc.height.values[i]

    x_minimo = int(c_x - (wi / 2))
    x_maximo = int(c_x + (wi / 2))
    y_minimo = int(c_y - (hi / 2))
    y_maximo = int(c_y + (hi / 2))

    image = cv2.imread(filename)
    filenames.append(name)

    if (c_x == 9999 or c_y == 9999):
        dnis.append("")
        tels.append("")
        mails.append("")
        nombres.append("")
        dirs.append("")
        distris.append("")
        ccs.append("")

        sco_dnis.append(0.0)
        sco_tels.append(0.0)
        sco_mails.append(0.0)
        sco_nombres.append(0.0)
        sco_dirs.append(0.0)
        sco_distris.append(0.0)
        sco_ccs.append(0.0)

        azure_str_jsons.append("")
    else:
        print('{} . {}'.format(i, name))

        cupon_areas = dni_rect_from_ce(c_x, c_y+y_min, cv2, image, traslation_dict, lines_dict)
        # print('{}'.format(cupon_areas))
        # fields, scores = get_fields_from_json(azure_jsons[i], cupon_areas)
        fields, scores = get_fields_from_json(proc.azure_json.values[i], cupon_areas)

        # Parsing numeric fields
        dni, sdni = post_num_field(fields['dni'],scores['dni'],banned_field_dict,n_eqs)
        tel, stel = post_num_field(fields['tel'],scores['tel'],banned_field_dict,n_eqs)
        dni, tel = basic_field_clean(dni).strip(), basic_field_clean(tel).strip()

        # Parsing mail
        mail, smail = post_num_field(fields['mail'],scores['mail'],banned_field_dict)
        mail = basic_field_clean(mail).lower().strip()

        # Parsing other fields
        nomb1, snomb1 = post_num_field(fields['nomb_ape1'],scores['nomb_ape1'],banned_field_dict,separator=" ")
        nomb2, snomb2 = post_num_field(fields['nomb_ape2'],scores['nomb_ape2'],banned_field_dict,separator=" ")
        dire1, sdire1 = post_num_field(fields['dir1'],scores['dir1'],banned_field_dict,separator=" ")
        dire2, sdire2 = post_num_field(fields['dir2'],scores['dir2'],banned_field_dict,separator=" ")
        dist, sdist = post_num_field(fields['distrito'],scores['distrito'],banned_field_dict,separator=" ")
        cc, scc = post_num_field(fields['cc'],scores['cc'],banned_field_dict,separator=" ")

        nomb1 = basic_field_clean(nomb1).strip()
        nomb2 = basic_field_clean(nomb2).strip()
        dire1 = basic_field_clean(dire1).strip()
        dire2 = basic_field_clean(dire2).strip()
        dist = basic_field_clean(dist).strip()
        cc = basic_field_clean(cc).strip()

        # Concatenating two lines fields
        nomb_ape, snomb_ape = nomb1+" "+nomb2, (snomb1+snomb2)/2
        direccion, sdireccion = dire1+" "+dire2, (sdire1+sdire2)/2

        # Last trying to get DNI
        if sdni==0:
            # print('{}'.format(post_num_field(fields['dni_tel'],scores['dni_tel'],{},separator="")))
            st, sc = post_num_field(fields['dni_tel'],scores['dni_tel'],{},separator=" ")
            dni = get_dni_from_line(st, n_eqs)
            sdni = 0.0 if dni=="" else sc
        # Last trying to get Tel
        if stel==0:
            # print('{}'.format(post_num_field(fields['dni_tel'],scores['dni_tel'],{},separator="")))
            st, sc = post_num_field(fields['dni_tel'],scores['dni_tel'],{},separator=" ")
            tel = get_tel_from_line(st, n_eqs)
            stel = 0.0 if tel=="" else sc

        dni = dni.replace(" ","")
        tel = tel.replace(" ","")
        # Showing fields
        # print('DNI {} {}'.format(dni, sdni))
        # print('TEL {} {}'.format(tel, stel))
        # print('MAIL {} {}'.format(mail, smail))
        # print('NOMB {} {}'.format(nomb_ape, snomb_ape))
        # print('NOMB2 {} {}'.format(nomb2, snomb2))
        # print('DIR {} {}'.format(direccion, sdireccion))
        # print('DIR2 {} {}'.format(dire2, sdire2))
        # print('DIST {} {}'.format(dist, sdist))
        # print('CC {} {}'.format(cc, scc))

        # Appending into arrays
        dnis.append(dni)
        tels.append(tel)
        mails.append(mail)
        nombres.append(nomb_ape)
        dirs.append(direccion)
        distris.append(dist)
        ccs.append(cc)

        sco_dnis.append(sdni)
        sco_tels.append(stel)
        sco_mails.append(smail)
        sco_nombres.append(snomb_ape)
        sco_dirs.append(sdireccion)
        sco_distris.append(sdist)
        sco_ccs.append(scc)

        azure_str_jsons.append(json.dumps(proc.azure_json.values[i]))

print('Termino :D ')


bd_azure = pd.DataFrame({'NombreArchivo': np.array(filenames)})
# bd_azure['idCupon'] = np.arange(1000, 1000+np.array(dnis).size,1)
bd_azure['DNI'] = np.array(dnis)
bd_azure['DNI'] = np.where(bd_azure['DNI'].apply(len) >= 14,
    bd_azure['DNI'].apply(lambda x : clean_letters(x)[:8]),
    bd_azure['DNI']
    )

bd_azure['AcertividadDNI'] = np.array(sco_dnis)
bd_azure['AcertividadDNI'] = np.where(np.logical_and(bd_azure['DNI'].apply(len) <= 9,
                                                     np.logical_and(bd_azure['DNI'].apply(len) >= 8,
                                                                    bd_azure['DNI'].apply(digits)
                                                                    )
                                                     ),
                                      bd_azure['AcertividadDNI'],
                                      0.0
                                      )

bd_azure['Telefono'] = np.array(tels)
bd_azure['AcertividadTelefono'] = np.array(sco_tels)
bd_azure['AcertividadTelefono'] = np.where(np.logical_and(np.logical_or(bd_azure['Telefono'].apply(len) == 9,
                                                                   np.logical_or(bd_azure['Telefono'].apply(len) == 7,
                                                                                 bd_azure['Telefono'].apply(len) == 6)
                                                                   ),
                                                     bd_azure['Telefono'].apply(digits)
                                                     ),
                                           bd_azure['AcertividadTelefono'],
                                           0.0
                                           )

bd_azure['NombreCompleto'] = np.array(nombres)
bd_azure['AcertividadNombreCompleto'] = np.array(sco_nombres)

bd_azure['Direccion'] = np.array(dirs)
bd_azure['AcertividadDireccion'] = np.array(sco_dirs)

bd_azure['Distrito'] = np.array(distris)
bd_azure['AcertividadDistrito'] = np.array(sco_distris)

bd_azure['Correo'] = np.array(mails)
# bd_azure['Correo'] = bd_azure['Correo'].apply(str.lower)

bd_azure['AcertividadCorreo'] = np.array(sco_mails)

bd_azure['AzureJsonOCR'] = np.array(azure_str_jsons)

bd_azure['idCampania'] = ID_CAMPANIA
bd_azure['idUsuario'] = ID_USUARIO
bd_azure['idEstado'] = 2
# bd_azure['idBatch'] = int(dt_string)

bd_azure.to_csv('azure_result/azure_result.csv', encoding='utf-8', index=False)
print('Termino escritura')

# dataframe
dataPrueba = bd_azure.fillna('')

# Corrección de tamaño de campos
dataPrueba['DNI'] = dataPrueba['DNI'].apply(lambda x: x[:20])
dataPrueba['Telefono'] = dataPrueba['Telefono'].apply(lambda x: x[:20])
dataPrueba['NombreCompleto'] = dataPrueba['NombreCompleto'].apply(lambda x: x[:150])
dataPrueba['Direccion'] = dataPrueba['Direccion'].apply(lambda x: x[:150])
dataPrueba['Distrito'] = dataPrueba['Distrito'].apply(lambda x: x[:50])
dataPrueba['Correo'] = dataPrueba['Correo'].apply(lambda x: x[:150])
dataPrueba['AzureJsonOCR'] = dataPrueba['AzureJsonOCR'].apply(lambda x: x[:8000])

# Getting the local result
local_result = pd.read_csv(LOCAL_PATH+'result.csv', dtype={'DNI':str, 'Telefono':str})

# Merge for updating in database
dataPrueba = pd.merge(dataPrueba, local_result[['idCupon','NombreArchivo','DNI','AcertividadDNI','Telefono','AcertividadTelefono']], how='left', on='NombreArchivo',)

dataPrueba['DNI_def'] = np.where(dataPrueba['AcertividadDNI_y']>=89.00,
                                 dataPrueba['DNI_y'],
                                 np.where(np.logical_and(dataPrueba['AcertividadDNI_x'] == 0.0, dataPrueba['AcertividadDNI_y'] != 0.0),
                                          dataPrueba['DNI_y'],
                                          dataPrueba['DNI_x']
                                         )
                                )
dataPrueba['AcertDNI_def'] = np.where(dataPrueba['AcertividadDNI_y']>=89.00,
                                 dataPrueba['AcertividadDNI_y'],
                                 np.where(np.logical_and(dataPrueba['AcertividadDNI_x'] == 0.0, dataPrueba['AcertividadDNI_y'] != 0.0),
                                          dataPrueba['AcertividadDNI_y'],
                                          dataPrueba['AcertividadDNI_x']
                                         )
                                )
dataPrueba['Telefono_def'] = np.where(dataPrueba['AcertividadTelefono_y']>=90.00,
                                 dataPrueba['Telefono_y'],
                                 np.where(np.logical_and(dataPrueba['AcertividadTelefono_x'] == 0.0, dataPrueba['AcertividadTelefono_y'] != 0.0),
                                          dataPrueba['Telefono_y'],
                                          dataPrueba['Telefono_x']
                                         )
                                )
dataPrueba['AcertTelefono_def'] = np.where(dataPrueba['AcertividadTelefono_y']>=90.00,
                                 dataPrueba['AcertividadTelefono_y'],
                                 np.where(np.logical_and(dataPrueba['AcertividadTelefono_x'] == 0.0, dataPrueba['AcertividadTelefono_y'] != 0.0),
                                          dataPrueba['AcertividadTelefono_y'],
                                          dataPrueba['AcertividadTelefono_x']
                                         )
                                )


if BD_SAVE_FLAG:
    '''
    cnxn = pyodbc.connect('Driver={SQL Server}; Server=192.168.2.55; Database=ClienteCupon; UID=usercupon;PWD=123456789', autocommit=True)
    conn_str = 'Driver={SQL Server}; Server=' + BD_HOST + '; Database='
    conn_str += BD_DATABASE_NAME + '; UID=' + BD_USERNAME + ';PWD=' + BD_PASSWORD
    cnxn = pyodbc.connect(conn_str, autocommit=True)
    crsr = cnxn.cursor()
    crsr.fast_executemany = False
    sql = "UPDATE Cupon SET [DNI]=?, [AcertividadDNI]=?, [Telefono]=?, [AcertividadTelefono]=?, [NombreCompleto]=?, [AcertividadNombreCompleto]=?, [Direccion]=?, [AcertividadDireccion]=?, [Distrito]=?, [AcertividadDistrito]=?, [Correo]=?, [AcertividadCorreo]=?, [idCampania]=?, [idUsuario]=?, [idEstado]=?, [AzureJsonOCR]=? WHERE [idCupon]=?;"
    '''
    params = [(dataPrueba.at[i,'DNI_def'],
        dataPrueba.iloc[i]['AcertDNI_def'],
        dataPrueba.at[i,'Telefono_def'],
        dataPrueba.iloc[i]['AcertTelefono_def'],
        dataPrueba.iloc[i]['NombreCompleto'],
        dataPrueba.iloc[i]['AcertividadNombreCompleto'],
        dataPrueba.iloc[i]['Direccion'],
        dataPrueba.iloc[i]['AcertividadDireccion'],
        dataPrueba.iloc[i]['Distrito'],
        dataPrueba.iloc[i]['AcertividadDistrito'],
        dataPrueba.iloc[i]['Correo'],
        dataPrueba.iloc[i]['AcertividadCorreo'],
        int(dataPrueba.iloc[i]['idCampania']),
        int(dataPrueba.iloc[i]['idUsuario']),
        int(dataPrueba.iloc[i]['idEstado']),
        dataPrueba.iloc[i]['AzureJsonOCR'],
        int(dataPrueba.at[i, 'idCupon'])) for i in range(dataPrueba.shape[0])]

    multi_db_send(params, 10)
    '''
    t0 = time.time()
    crsr.executemany(sql, params)
    print(f'{time.time() - t0:.1f} seconds')
    '''
# Merge
local_result = pd.read_csv(LOCAL_PATH+'result.csv', dtype={'DNI':str, 'Telefono':str})
res =pd.merge(local_result, bd_azure, how='left', on='NombreArchivo',)
res.rename(columns={
                        'DNI_x':'DNI_local',
                        'AcertividadDNI_x':'AcertividadDNI_local',
                        'DNI_y':'DNI_azure',
                        'AcertividadDNI_y':'AcertividadDNI_azure',
                        'Telefono_x':'Telefono_local',
                        'AcertividadTelefono_x':'AcertividadTelefono_local',
                        'Telefono_y':'Telefono_azure',
                        'AcertividadTelefono_y':'AcertividadTelefono_azure',
                    },
                 inplace=True)

print('Termino actualizacion en base de datos')

# Defining pandas engine for sql
conn_str = 'mssql+pyodbc://' + BD_USERNAME + ':' + BD_PASSWORD + '@'
conn_str += BD_HOST + '/' + BD_DATABASE_NAME
conn_str += '?driver=SQL+Server+Native+Client+11.0'
engine = sa.create_engine(conn_str)

# Get the value of the params
sql_params = 'select * from ' + BD_DATABASE_NAME + '.dbo.Campo'
params = pd.read_sql_query(sql_params, engine)
dni_param = params[params['Denominacion']=='DNI'].iloc[0,2]
tel_param = params[params['Denominacion']=='Telefono'].iloc[0,2]
mail_param = params[params['Denominacion']=='Email'].iloc[0,2]

# Subset of DNI
sub_dni = res[res.AcertividadDNI_azure<dni_param]
sub_dni = sub_dni[['idCupon']]
sub_dni['idCampo'] = int(1)

# Subset of Telefono
sub_tel = res[res.AcertividadTelefono_azure<tel_param]
sub_tel = sub_tel[['idCupon']]
sub_tel['idCampo'] = int(2)

# Subset of Email
sub_mail = res[res.AcertividadCorreo<mail_param]
sub_mail = sub_mail[['idCupon']]
sub_mail['idCampo'] = int(3)

if BD_SAVE_FLAG:
    # Saving in DB
    # Inserción del subset DNI
    t0 = time.time()
    # sub_dni.to_sql('CuponUsuario', engine, if_exists='append', index=False, chunksize=200)
    print("Inserción de {} rows en CuponUsuario DNI finalizada en {:.1f} seconds".format(time.time()-t0, sub_dni['idCupon'].values.size))

    # Inserción del subset Telefono
    t0 = time.time()
    # sub_tel.to_sql('CuponUsuario', engine, if_exists='append', index=False, chunksize=200)
    print("Inserción de {} rows en CuponUsuario DNI finalizada en {:.1f} seconds".format(time.time()-t0, sub_tel['idCupon'].values.size))

    # Inserción del subset Email
    t0 = time.time()
    # sub_mail.to_sql('CuponUsuario', engine, if_exists='append', index=False, chunksize=200)
    print("Inserción de {} rows en CuponUsuario DNI finalizada en {:.1f} seconds".format(time.time()-t0, sub_mail['idCupon'].values.size))

# Writing summary
res = res[['NombreArchivo','DNI_local','AcertividadDNI_local','DNI_azure','AcertividadDNI_azure','Telefono_local','AcertividadTelefono_local','Telefono_azure','AcertividadTelefono_azure']]
res.to_csv('azure_result/summary.csv', encoding='utf-8', index = False)
