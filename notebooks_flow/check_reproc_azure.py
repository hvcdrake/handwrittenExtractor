# Import section
import numpy as np
import pandas as pd
from paralel_send import multi_send
from os import getcwd
import os
import shutil

# Folder params
WORK_DIRECTORY = getcwd()
ARRAYS_PATH = WORK_DIRECTORY + '\\areas_result\\'
TMP_PATH = WORK_DIRECTORY + '\\temps\\'
LOCAL_PATH = WORK_DIRECTORY + '\\local_result\\'
TMP_FAILED_PATH = TMP_PATH + '\\reproc\\'

# Loading muestra numpy arrays
areas_r = pd.read_csv(ARRAYS_PATH + 'areas.csv')
areas_r = areas_r.fillna('')
print('Total cupones area: {}'.format(areas_r.path.values.size))

# Moving temp files into reproc
filenames = [arch.name for arch in os.scandir(TMP_PATH) if arch.is_file()]

for f in filenames:
    shutil.move(TMP_PATH+f, TMP_FAILED_PATH+f)

# Getting all the processed files
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

azure_files = np.concatenate(files[:], axis=0)
azure_jsons = np.concatenate(jsons[:], axis=0)
print('Resultado de azure: {}  {}'.format(azure_files.size, azure_jsons.size))

# Merging the areas result with the azure result
proc = pd.DataFrame({'path': azure_files, 'azure_json': azure_jsons})
proc = pd.merge(areas_r, proc, how='left', on='path')
print('Total cupones merge: {}'.format(proc.path.values.size))

NUM_WORKERS = 10
print('Empezó multi send con:{}'.format(len(proc[proc['azure_json'].isnull()]['path'].values.tolist())))
multi_send(proc[proc['azure_json'].isnull()]['path'].values.tolist(), NUM_WORKERS, 1, 1)
print('Terminó multi send')
