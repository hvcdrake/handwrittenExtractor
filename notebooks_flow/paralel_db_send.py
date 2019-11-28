import threading
import time
import datetime
import pyodbc
import sqlalchemy as sa


# BD params
BD_USERNAME = 'sa'
BD_PASSWORD = 'Admin123'
BD_DATABASE_NAME = 'ClienteCupon'
# BD_DATABASE_NAME = 'DevClienteCupon'
BD_HOST = '13.82.178.179,2701'
'''
BD_USERNAME = 'usercupon'
BD_PASSWORD = '123456789'
# BD_DATABASE_NAME = 'ClienteCupon'
BD_DATABASE_NAME = 'DevClienteCupon'
BD_HOST = '192.168.2.55'
'''

def multi_db_insert(df_full, workers, table_name):
    fs = df_full[:]
    num_of_workers = workers

    size_per_thread = 1 if len(fs) <= 10 else ((len(fs) // num_of_workers) + 1)

    loads_per_thread = []
    li = 0

    print('Tamaño por worker: {} de total {}'.format(size_per_thread, len(fs)))
    for i in range(num_of_workers):
        loads_per_thread.append(fs.iloc[li:li + size_per_thread, :].copy())
        li += size_per_thread

    # Initializing all the threads
    threads = []
    i = 0
    for load in loads_per_thread:
        hilo = threading.Thread(target=insert_in_db, args=(load[:], i, table_name),)
        threads.append(hilo)
        # print(load.index)
        i += 1

    # Starting all the threads
    for t in threads:
        t.start()

    # Waiting for all the threads
    for t in threads:
        t.join()

    print('End process')


def multi_db_send(params_full, workers):
    fs = params_full[:]
    num_of_workers = workers
    # size_per_thread = 1 if len(fs) <= 10 else (len(fs) // num_of_workers) + (1 if (len(fs) // num_of_workers) != 0 else 0)
    size_per_thread = 1 if len(fs) <= 10 else (len(fs) // num_of_workers) + (1 if False != 0 else 0)

    loads_per_thread = []
    li = 0

    print('Tamaño por worker: {} de total {}'.format(size_per_thread, len(fs)))
    for i in range(num_of_workers):
        loads_per_thread.append(fs[li:li + size_per_thread].copy())
        li += size_per_thread

    print('Existe un extra de {} a repartir'.format(len(fs) - li))
    if li < len(fs):
        k = 0
        for res in fs[li:]:
            loads_per_thread[k].append(res)
            k += 1
    '''
    i = 0
    for load in loads_per_thread:
        print('{}. Tamaño: {} lista: {}'.format(i, len(load), load))
        i += 1
    '''

    # Initializing all the threads
    threads = []
    i = 0
    for load in loads_per_thread:
        hilo = threading.Thread(target=save_in_db, args=(load[:], i),)
        threads.append(hilo)
        i += 1

    # Starting all the threads
    for t in threads:
        t.start()

    # Waiting for all the threads
    for t in threads:
        t.join()

    print('End process')


def insert_in_db(df, worker_id, table_name):
    '''
     params: array containing the params to save in db
     worker_id : worker incremental identificaction
    '''

    try:
        print("Worker {}: Starting save into DB".format(worker_id))

        conn_str = 'mssql+pyodbc://' + BD_USERNAME + ':' + BD_PASSWORD + '@'
        conn_str += BD_HOST + '/' + BD_DATABASE_NAME
        conn_str += '?driver=SQL+Server+Native+Client+11.0'
        engine = sa.create_engine(conn_str)

        # Inserción a la tabla
        t0 = time.time()
        df.to_sql(table_name, engine, if_exists='append', index=False, chunksize=200)
        print("Worker {}: Inserción finalizada en {:.1f} seconds".format(worker_id, time.time()-t0))

    except Exception as e:
        print('Worker {}: .Error: {}'.format(worker_id, e))
    finally:
        # Close Close
        # image.close()
        print('End Worker {} in {}'.format(worker_id, datetime.datetime.now()))



def save_in_db(params, worker_id):
    '''
     params: array containing the params to save in db
     worker_id : worker incremental identificaction
    '''

    try:
        print("Worker {}: Starting save into DB".format(worker_id))
        conn_str = 'Driver={SQL Server}; Server=' + BD_HOST + '; Database='
        conn_str += BD_DATABASE_NAME + '; UID=' + BD_USERNAME + ';PWD=' + BD_PASSWORD
        cnxn = pyodbc.connect(conn_str, autocommit=True)
        crsr = cnxn.cursor()
        crsr.fast_executemany = False

        sql = "UPDATE Cupon SET [DNI]=?, [AcertividadDNI]=?, [Telefono]=?, [AcertividadTelefono]=?, [NombreCompleto]=?, [AcertividadNombreCompleto]=?, [Direccion]=?, [AcertividadDireccion]=?, [Distrito]=?, [AcertividadDistrito]=?, [Correo]=?, [AcertividadCorreo]=?, [idCampania]=?, [idUsuario]=?, [idEstado]=?, [AzureJsonOCR]=?, [DNI_Original]=?, [Telefono_Original]=?, [Correo_Original]=? WHERE [idCupon]=?;"

        t0 = time.time()
        crsr.executemany(sql, params)
        print("Worker {}: Finishing in {:.1f} seconds".format(worker_id, time.time() - t0))
        crsr.close()
    except Exception as e:
        print('Worker {}: .Error: {}'.format(worker_id, e))
    finally:
        # Close Close
        # image.close()
        print('End Worker {} in {}'.format(worker_id, datetime.datetime.now()))

