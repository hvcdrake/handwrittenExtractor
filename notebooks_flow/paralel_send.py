import threading
import requests
import time
import json
import numpy as np
import datetime
import os


# URL variables
# vision_base_url = "https://eastus2.api.cognitive.microsoft.com/vision/v2.0/"
# vision_base_url = "https://southcentralus.api.cognitive.microsoft.com/vision/v2.0/"
# subscription_key = "db6ddba1087844a58b0329c27b44ff5e"
# subscription_key = "e3d29d8cac4748328976033879af3216"
# WONG CREDENTIALS
vision_base_url = "https://ewongcomputervisionapi.cognitiveservices.azure.com/vision/v2.0/"
subscription_key = "5073681f50e64c1ea2618089f673b00b"

# 2ND WONG CREDENTIALS
vision_base_url2 = "https://ewongcomputervision02.cognitiveservices.azure.com/vision/v2.0/"
subscription_key2 = "6a5396cd65ed49f99d75a86f0fdd192c"

ocr_url = vision_base_url + "recognizeText?mode=Handwritten"
ocr_url2 = vision_base_url2 + "recognizeText?mode=Handwritten"

headers = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key
}

headers2 = {
    'Content-Type': 'application/octet-stream',
    'Ocp-Apim-Subscription-Key': subscription_key2
}


def multi_send(paths, workers, sleep_after, await_before):
    fs = paths[:]
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
        hilo = threading.Thread(target=send_azure, args=(load[:], sleep_after, await_before, i),)
        threads.append(hilo)
        i += 1

    # Starting all the threads
    for t in threads:
        t.start()

    # Waiting for all the threads
    for t in threads:
        t.join()

    print('End process')


def send_azure(paths, sleep_after_time, await_before_ask_time, worker_id):
    '''
     paths: numpy array containing the paths of the files to send azure
    '''

    files = []
    jsons = []

    sec_account = True if worker_id > 9 else False
    str_debug_acc = "2nd" if sec_account else "1st"

    try:
        # Fetching in paths
        for i, path in enumerate(paths):
            print("Acc: {}. Worker {}: Reading {}.{}".format(str_debug_acc, worker_id, i, path.split('\\')[-1]))
            image = open(path, "rb")
            d = image.read()

            if sec_account:
                response = requests.post(ocr_url, headers=headers, data=d)
            else:
                response = requests.post(ocr_url2, headers=headers2, data=d)

            # Holds the URI used to retrieve the recognized text.
            operation_url = response.headers["Operation-Location"]

            # The recognized text isn't immediately available, so poll to wait for completion.
            analysis = {}
            poll = True

            while (poll):
                time.sleep(await_before_ask_time)

                if sec_account:
                    response_final = requests.get(operation_url, headers=headers2)
                else:
                    response_final = requests.get(operation_url, headers=headers)

                analysis = response_final.json()
                debug_str = str(analysis)[:50] if len(str(analysis)) >= 50 else str(analysis)
                print('Acc: {}. Worker {}: Status:{}'.format(str_debug_acc, worker_id, debug_str))

                if ('recognitionResult' in analysis):
                    poll = False
                if ("status" in analysis and analysis['status'] == 'Failed'):
                    poll = False
            obj = json.loads(response_final.text)

            files.append(path)
            jsons.append(obj)
            image.close()
            print("Acc: {}. Worker {}: Finished {}.{} and waiting for {}".format(str_debug_acc, worker_id, i, path.split('\\')[-1], sleep_after_time))

            if (i % 300) == 0:
                # print("Worker {}: Pre-saving")
                # convirtiendo a nunmpy arrays
                n_files = np.array(files)
                n_jsons = np.array(jsons)
                # Saving nunmpy arrays
                np.save('temps/n_files_{}_nf'.format(worker_id), n_files)
                np.save('temps/n_jsons_{}_nf'.format(worker_id), n_jsons)
            time.sleep(sleep_after_time)

        # convirtiendo a nunmpy arrays
        n_files = np.array(files)
        n_jsons = np.array(jsons)
        # Saving numpy arrays
        np.save('temps/n_files_{}'.format(worker_id), n_files)
        np.save('temps/n_jsons_{}'.format(worker_id), n_jsons)

        tries = 1
        while not os.path.exists('temps/n_files_{}.npy'.format(worker_id)):
            print('Trying to save. Try:{}'.format(tries))
            np.save('temps/n_files_{}'.format(worker_id), n_files)
            tries += 1
            if tries >= 10:
                break

        if tries >= 10:
            print("Acc: {}. Worker {}: Numero de intentos superado n_files".format(str_debug_acc, worker_id))
        else:
            print("Acc: {}. Worker {}: Guardado nfiles exitoso".format(str_debug_acc, worker_id))

        tries = 1
        while not os.path.exists('temps/n_jsons_{}.npy'.format(worker_id)):
            print('Trying to save. Try:{}'.format(tries))
            np.save('temps/n_jsons_{}'.format(worker_id), n_jsons)
            tries += 1
            if tries >= 10:
                break

        if tries >= 10:
            print("Acc: {}. Worker {}: Numero de intentos superado n_jsons_".format(str_debug_acc, worker_id))
        else:
            print("Acc: {}. Worker {}: Guardado n_jsons_ exitoso".format(str_debug_acc, worker_id))

        if os.path.exists('temps/n_files_{}.npy'.format(worker_id)) and os.path.exists('temps/n_files_{}_nf.npy'.format(worker_id)):
            print("Acc: {}. Worker {}: Final f deleting".format(str_debug_acc, worker_id))
            os.remove('temps/n_files_{}_nf.npy'.format(worker_id))
        if os.path.exists('temps/n_jsons_{}.npy'.format(worker_id)) and os.path.exists('temps/n_jsons_{}_nf.npy'.format(worker_id)):
            os.remove('temps/n_jsons_{}_nf.npy'.format(worker_id))
            print("Acc: {}. Worker {}: Final js deleting".format(str_debug_acc, worker_id))

    except Exception as e:
        print('Worker {}: .Error: {}'.format(worker_id, e))
    finally:
        # Close Close
        # image.close()
        print('End Worker {} in {}'.format(worker_id, datetime.datetime.now()))

# print('Start multi send')
# multi_send(files[:11], 20, 60, 60)
# print("End")
