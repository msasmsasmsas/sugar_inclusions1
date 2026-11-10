import torch
import numpy as np
from PIL import Image

import cv2
import requests

import os

# Імпортуємо централізовану конфігурацію
from env_config import (
    ENV_LEVEL, USE_DB, MODEL_VERSION,
    SAVE_RESULTS_FOR_MARKINGS, SAVE_DETECTED_IMAGES, SAVE_MASKED_IMAGES,
    SHOW_RESULTS, SAVE_RESULTS, TELEGRAM_TOKEN,
    TELEGRAM_CHANNEL_ID, TELEGRAM_CHANNEL_ID_KOVNIR,
    TELEGRAM_CHANNEL_ID_JSZ, PERIOD_ROTATION, CSV_ROTATION,
    NUM_DETECTION_THREADS, CAMERAS_ACTIVE_STATUS, LOG_LEVEL
)

from ultralytics import YOLO
from yolov5.utils.plots import Annotator, colors
import shutil
from torchvision.ops import nms
#import glob
#from send_mail import init_mail
#import json
# from picamera2 import Picamera2
# from IPython.display import display

import hikvisionapi
import time
from datetime import datetime, timezone
import pytz

import csv
from threading import Thread
from dir_files import work_with_files


# import pygsheets
import logging
from logging.handlers import RotatingFileHandler

from src.migrations.database import SessionLocal, engine
from src.migrations.models import Base, detection_sugar_inclusions
from dto import detectionSugarInclusions
from sqlalchemy.orm import Session
import paramiko
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue, Empty
import json

# Ініціалізація БД буде пізніше після перевірки use_db_connection
# Base.metadata.create_all(bind=engine, checkfirst=True)

# Встановлюємо prod_release на основі ENV_LEVEL
prod_release = True if ENV_LEVEL == 'prod' else False
if ENV_LEVEL == 'prod':
    cameras = {
        "JDN": {
            "Location": "Жданівка",
            "type_cam": "dahua",
            #"camera_ip": "10.20.60.99:10084",
            "camera_ip": "10.112.0.143:10084",
            "username": "admin",
            "password": "Admin12345",
            "is_active": True
        },
        "NRK": {
            "Location": "Наркевичі",
            "type_cam": "dahua",
            #"camera_ip": "10.20.188.200:10085",
            "camera_ip": "10.112.0.143:10085",
            "username": "admin",
            "password": "Admin12345",
            "is_active": True
        },
        "JCZ": {
            "Location": "Яреськи",
            "type_cam": "hikvision",
            #"camera_ip": "10.20.2.250:10086",
            "camera_ip": "10.112.0.143:10086",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        },
        "GLO": {
            "Location": "Глобине",
            "type_cam": "hikvision",
            #"camera_ip": "10.20.40.250:10087",
            "camera_ip": "10.112.0.143:10087",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        },
        "NVO": {
            "Location": "Новооржиця",
            "type_cam": "hikvision",
            #"camera_ip": "10.20.105.250:10088",
            "camera_ip": "10.112.0.143:10088",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        }
    }
elif ENV_LEVEL == 'dev':
    cameras = {
        "JDN": {
            "Location": "Жданівка",
            "type_cam": "dahua",
            "camera_ip": "10.20.60.99",
            "username": "admin",
            "password": "Admin12345",
            "is_active": True
        },
        "NRK": {
            "Location": "Наркевичі",
            "type_cam": "dahua",
            "camera_ip": "10.20.188.200",
            "username": "admin",
            "password": "Admin12345",
            "is_active": True
        },
        "JCZ": {
            "Location": "Яреськи",
            "type_cam": "hikvision",
            "camera_ip": "10.20.2.250",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        },
        "GLO": {
            "Location": "Глобине",
            "type_cam": "hikvision",
            "camera_ip": "10.20.40.251",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        },
        "NVO": {
            "Location": "Новооржиця",
            "type_cam": "hikvision",
            "camera_ip": "10.20.105.250",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        }
    }    
elif ENV_LEVEL == 'local':
    cameras = {
        "JDN": {
            "Location": "Жданівка 8MP",
            "type_cam": "dahua",
            "camera_ip": "192.168.1.164",
            "username": "admin",
            "password": "Admin12345",
            "is_active": True
        },
        "NRK": {
            "Location": "Наркевичі 4MP",
            "type_cam": "dahua",
            "camera_ip": "192.168.1.165",
            "username": "admin",
            "password": "Admin12345",
            "is_active": True
        },
        "JCZ": {
            "Location": "Яреськи",
            "type_cam": "hikvision",
            "camera_ip": "192.168.1.167",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        },
        "NVO": {
            "Location": "Новооржиця",
            "type_cam": "hikvision",
            "camera_ip": "192.168.1.168",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        },
        "GLO": {
            "Location": "Глобине",
            "type_cam": "hikvision",
            "camera_ip": "192.168.1.169",
            "username": "admin",
            "password": "Astarta_2025",
            "is_active": True
        }
    }
    
print(f"prod_release = {prod_release}")
print(f"Using cameras: {cameras}")

# Обробка змінної середовища для налаштування активності камер
if CAMERAS_ACTIVE_STATUS:
    try:
        # Парсимо формат: JDN=true,NRK=false,...
        for cam_status in CAMERAS_ACTIVE_STATUS.split(','):
            if '=' in cam_status:
                cam_key, status = cam_status.strip().split('=')
                if cam_key in cameras:
                    cameras[cam_key]['is_active'] = status.lower() == 'true'
                    print(f"Camera {cam_key} active status set to: {cameras[cam_key]['is_active']}")
    except Exception as e:
        print(f"Error parsing CAMERAS_ACTIVE_STATUS: {e}")

# Виведення інформації про активні та неактивні камери
active_cameras = [k for k, v in cameras.items() if v.get('is_active', True)]
inactive_cameras = [k for k, v in cameras.items() if not v.get('is_active', True)]
print(f"Active cameras: {active_cameras}")
if inactive_cameras:
    print(f"Inactive cameras: {inactive_cameras}")
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def send_text_telegram(token, channel_id, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendMessage"

    r = requests.post(method, data={
        "chat_id": channel_id,
        "text": text
    })

    if r.status_code != 200:
        raise Exception("post_text error")

def send_photo_telegram(token, channel_id, path_photo: str, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendPhoto"

    files = {'photo': open(path_photo, 'rb')}

    r = requests.post(method, files=files, data={
        "chat_id": channel_id,
        "text": text
    })
    
    if r.status_code != 200:
        logger.error(f"[Telegram] [ERROR] Failed to send photo: status={r.status_code}, response={r.text}")
        raise Exception(f"post_text error: {r.status_code} - {r.text}")

def send_document_telegram(token, channel_id, path_csv: str, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendDocument"

    files = {'document': open(path_csv, 'rb')}

    r = requests.post(method, files=files, data={
        "chat_id": channel_id,
        "text": text
    })
    print('Отправляем файл %s' %(path_csv))
    #print(r.json())

    if r.status_code != 200:
        raise Exception("post_text error")

def send_document_telegram1(token, channel_id, path_csv: str, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendDocument"

    with open(path_csv, "rb") as csvfile:
        files = {"document": csvfile}
        title = "данные.csv"
    print('Отправляем файл %s' %(path_csv))
    try:
        r = requests.post(method, data={"chat_id": channel_id, "caption": text}, files=files)
    except OSError as e:
        logger.error("Ошибка: %s" % (e.strerror))
        #print("Ошибка: %s" % (e.strerror))
    if r.status_code != 200:
       raise Exception("send error")

def send_group_img(token, channel_id, output_path, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendMediaGroup"

    temp_files_list = list()
    media = list()
    files = dict()
    for filename in os.listdir(output_path):

        file_path = f'{os.getcwd()}/{output_path}/{filename}'
        file_path = file_path.replace('/', '\\')
        if os.path.isfile(file_path):
            temp_files_list.append(file_path)

    for f in enumerate(temp_files_list):
        files[f"random-name-{f[0]}"] = open(f[1], "rb")
        if f[0] == 0:
            media.append({"type": "photo",
                          "media": f"attach://random-name-{f[0]}",
                          "caption": text}
                         )
        else:
            media.append({"type": "photo",
                          "media": f"attach://random-name-{f[0]}"})
    params = {
        "chat_id": channel_id, "media": str(media).replace("'", '"')}
    result = requests.post(method, params=params, files=files)
    if result.status_code == 200:
        return True
    else:
        return False

def draw_text(img, point, text, fontScale, fontFace, text_thickness, bg_thickness, bg_color, drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param fontScale = 0.4
    :param fontFace = cv2.FONT_HERSHEY_SIMPLEX
    :param text_thickness = 1
    :param bg_thickness = 5
    :param bg_color = (255, 0, 0)
    :param drawType: custom or simple
   :return:
    '''
    # fontFace=cv2.FONT_HERSHEY_SIMPLEX
    if drawType == "custom":
        text_size, baseline = cv2.getTextSize(str(text), fontFace, fontScale, bg_thickness)
        text_loc = (point[0], point[1] + text_size[1])
        cv2.rectangle(img, (text_loc[0] - 2 // 2, text_loc[1] - 2 - baseline),
                      (text_loc[0] + text_size[0], text_loc[1] + text_size[1]), bg_color, -1)
        # draw score value
        cv2.putText(img, str(text), (text_loc[0], text_loc[1] + baseline), fontFace, fontScale,
                    (255, 255, 255), text_thickness, 8)
    elif drawType == "simple":
        cv2.putText(img, (text), point, fontFace, 0.5, (255, 255, 0))
    return img

def draw_text_line(img, point, text_line: str, fontScale, fontFace, text_thickness, bg_thickness, bg_color,
                   drawType="custom"):
    '''
    :param img:
    :param point:
    :param text:
    :param fontScale = 0.4
    :param fontFace = cv2.FONT_HERSHEY_SIMPLEX
    :param text_thickness = 1
    :param thickness = 5
    :param bg_color = (255, 0, 0)
    :param drawType: custom or simple
    :return:
    '''
    text_line = text_line.split("\n")
    # text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, thickness)
    text_size, baseline = cv2.getTextSize(str(text_line), fontFace, fontScale, bg_thickness)
    for i, text in enumerate(text_line):
        if text:
            draw_point = [point[0], point[1] + (text_size[1] + 2 + baseline) * i]
            img = draw_text(img, draw_point, text, fontScale, fontFace, text_thickness, bg_thickness, bg_color,
                            drawType)
    return img

def detect_trash_v5(camera_key: str, img_path: str, num_thread,detection_start):
    if os.path.isfile(img_path):
        file_name = img_path.split('/')[-1] # images/source/screen20221202_16_48_46_088831.jpg ---> screen20221202_16_48_46_088831.jpg
        file_name = file_name.split('.')[0] # screen20221202_16_48_46_088831.jpg ---> screen20221202_16_48_46_088831
        file_date_time = file_name[6:] # screen20221202_16_48_46_088831 ---> 20221202_16_48_46_088831
        file_date = file_date_time[0:8] # 20221202_16_48_46_088831 ---> 20221202
        file_date = file_date[0:4]+'-'+file_date[4:6]+'-'+file_date[6:8]
        file_time = file_date_time[9:] # 20221202_16_48_46_088831 ---> 16_48_46_088831
        file_time = file_time.replace("_",":",2) # 16_48_46_088831 ---> 16:48:46_088831
        file_time = file_time.replace("_",".",) # 16:48:46_088831  ---> 16:48:46.088831


    try:
        img = cv2.imread(img_path)
    except:
        exit()
    h, w = img.shape[:2]
    #diff_time_detect = time.time() - detection_start
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))

    # cv2.imshow('Harris Corners', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    mask = np.zeros(img.shape, dtype=np.uint8)

    # Poligon that limits the area in the picture
    #roi_corners = np.array([[(450, 325), (1480, 160),
    #                         (2340, 90), (2640, 710),
    #                         (2590, 750), (2630, 840),
    #                         (1710, 1515), (960, 1515),
    #                         (890, 1360),
    #                         ]],
    #                       dtype=np.int32)
    roi_corners = np.array([[(0, 0), (w, 0),
                             (w, h), (0, h)
                             ]],
                           dtype=np.int32)

    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    #cv2.imwrite('images/for_detection/image_masked.png', masked_image)
    cv2.imwrite(img_path, masked_image)
    if os.path.isfile(img_path):
        try:
            img = cv2.imread(img_path)
        except OSError as e:
            logger.error("detection Ошибка чтения файла: %s : %s" % (img_path, e.strerror))
            exit(-1)
    else:
        exit(-1)
     # Inference
    logger.warn('Camera %s: Start detection %d %d' %(camera_key, h,w))
    #print('Start detection %d %d' %(h,w))
    #diff_time_detect = time.time() - detection_start
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))
    try:
        results = model(img)
    except OSError as e:
        logger.error("detection Ошибка работы модели: %s : %s" % (img_path, e.strerror))
        exit(-1)

    labels = results.names
    #diff_time_detect = time.time() - detection_start
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))
    # tekList = results.xyxy[0].tolist()
    # tekLen = len(results.xyxy[0].tolist())

    # tekresults = results.tolist()
    # print(tekresults)

    s = ''
    isTrash = False
    for i, (im, pred) in enumerate(zip(results.ims, results.pred)):
         s += f'изображение {i + 1}/{len(results.pred)}: {im.shape[0]}x{im.shape[1]}, '  # string
         dict = {a: 0 for a in range(3)}
         if pred.shape[0]:
             for c in pred[:, -1].unique():
                 n = (pred[:, -1] == c).sum()  # detections per class
                 s += f"{n} - {labels_ru[results.names[int(c)]]}{'s' * (n > 1)}, "  # add to string
                 s = s.replace('s,', ',')
                 dict[int(c)] = int(f"{n}")
                 if int(c) == 0: # Только для мусора
                     isTrash = True
             s = s.rstrip(', ')
             save_to_csv(camera_key, path_csv,dict,file_date,file_time)
             if use_db_connection:
                 save_to_db(dict, camera_key, file_date + ' ' + file_time)
             #save_to_google_sheets(dict)
             annotator = Annotator(img, example=str(results.names))
             for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                 label = f' {conf:.2f}'
                 # {results.names[int(cls)]}
                 annotator.box_label(box, label if labels else '', color=colors(cls))
             img = annotator.im
             #cv2.imshow('только рамки', img)
             #cv2.waitKey(0)
             # cv2.imwrite('runs/detect/exp/images_%03d_1.jpg' %(i), img)
             # pl.imshow(img)
             # pl.show()
    #save_to_google_sheets(dict)
    text = s.replace(', ', '\n')
    img = draw_text_line(img, (5, 20), text, FONT_SCALE, FONT_FACE, FONT_THICKNESS, LINE_THICKNESS, LINE_COLOR,
                         "custom")
    cv2.imwrite(output_path + '/images_%03d.jpg' % (i), im)

    #    display(im) if is_notebook() else im.show(results.files[i])
    #    img.show()

    # Results
    # print('Print results ')
    # results.print() # or .s
    # print('Print results files ')
    # print(results.files)
    # print(results.ims)
    # print(results.pred)

    # print('Try result tensor ')
    # results.xyxy[0]
    # print('Print Pandas tensor ')
    # results.pandas().xyxy[0]
    # print('Crops ')
    # crops = results.crop(save=True)
    #diff_time_detect = time.time() - detection_start
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))
    if saveResults_ForMarkings:
        cv2.imwrite("images/for_marking/" + file_name, masked_image)
    if isTrash:
        extension = img_path.split('.')[-1]
        file_name = img_path.split('/')[-1]

        if showResults:
            results.show()
        if saveResults:
            logger.info('detection: Save analize ')
            #results.save(True, path_csv)
            try:
                results.save(True, )
            except OSError as e:
                logger.error("detection Ошибка сохранения результатов модели: %s : %s" % (img_path, e.strerror))
                #print("detection Ошибка сохранения результатов модели: %s : %s" % (img_path, e.strerror))

        # send_text_telegram(token, channel_id, text)
        # send_photo_telegram(token, channel_id, output_path+'/image0.jpg',text)
        try:
            # sended = send_group_img(token, channel_id, output_path, text)
            # sended = send_group_img(token, channel_id_kovnir, output_path, text)
            sended = send_photo_telegram(token, channel_id, output_path+'/images_%03d.jpg' % (i),text)
            if prod_release:

                sended = send_photo_telegram(token, channel_id_kovnir, output_path+'/images_%03d.jpg' % (i),text)
                sended = send_photo_telegram(token, channel_id_jsz, output_path+'/images_%03d.jpg' % (i),text)

            print('Отправляем фото %s :: %s' % (output_path, sended))

            #sended = send_group_img(token, channel_id_kovnir, output_path, text)
        except OSError as e:
            logger.error("detection: Ошибка отправки в бот: %s : %s" % (output_path, e.strerror))
            #print("detection: Ошибка отправки в бот: %s : %s" % (output_path, e.strerror))
        addr_to = "oleksandr.maslianchuk@agrichain.ua"  # Получатель
        files = [
            '../yolov5_train_detect/runs/exp/']  # Если нужно отправить все файлы из заданной папки, нужно указать её
        # init_mail.send_email(addr_to, "sugar_trash Привет, тебе сообщение от малинки", "Первіе файлы", files)
    else:
        logger.info("srvs_threads %s: detect_trash_v5: не нашли мусор" % (output_path, sended))

def detect_trash_v7(camera_key: str, img_path: str, num_thread,detection_start):
    #diff_time_detect = time.time() - detection_start
    #print('-3 Разница %s '% (diff_time_detect))
    if os.path.isfile(img_path):
        file_name = img_path.split('/')[-1] # images/source/screen20221202_16_48_46_088831.jpg ---> screen20221202_16_48_46_088831.jpg
        file_name = file_name.split('.')[0] # screen20221202_16_48_46_088831.jpg ---> screen20221202_16_48_46_088831
        file_date_time = file_name[6:] # screen20221202_16_48_46_088831 ---> 20221202_16_48_46_088831
        file_date = file_date_time[0:8] # 20221202_16_48_46_088831 ---> 20221202
        file_date = file_date[0:4]+'-'+file_date[4:6]+'-'+file_date[6:8]
        file_time = file_date_time[9:] # 20221202_16_48_46_088831 ---> 16_48_46_088831
        file_time = file_time.replace("_",":",2) # 16_48_46_088831 ---> 16:48:46_088831
        file_time = file_time.replace("_",".",) # 16:48:46_088831  ---> 16:48:46.088831


    try:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
    except OSError as e:
        logger.error("srvs_threads %s: detect_trash_v7: Ошибка чтения файла изображения: %s : %s" % (
        num_thread, img_path, e.strerror))
        exit()
    #diff_time_detect = time.time() - detection_start
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))

    # cv2.imshow('Harris Corners', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    #diff_time_detect = time.time() - detection_start
    #print('-2 Разница %s '% (diff_time_detect))

    mask = np.zeros(img.shape, dtype=np.uint8)

    # Poligon that limits the area in the picture
    roi_corners = np.array([[(585, 995), (645, 1515),
                             (1210, 1351), (1615, 1120),
                             (1380, 770),
                             ]],
                           dtype=np.int32)
    #roi_corners = np.array([[(0, 0), (w, 0),
    #                         (w, h), (0, h)
    #                         ]],
    #                       dtype=np.int32)

    channel_count = img.shape[2]
    ignore_mask_color = (255,) * channel_count
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    #diff_time_detect = time.time() - detection_start
    #print('-1 Разница %s '% (diff_time_detect))

    #cv2.imwrite('images/for_detection/image_masked.png', masked_image)
    cv2.imwrite(img_path, masked_image)
    if os.path.isfile(img_path):
        try:
            img = cv2.imread(img_path)
            #img.show()
        except OSError as e:
            logger.error("srvs_threads %s: detect_trash_v7: Ошибка чтения файла изображения: %s : %s" % (num_thread, img_path, e.strerror))
            exit(-1)
    else:
        exit(-1)
     # Inference
    logger.info('')
    logger.info('srvs_threads %s: detect_trash_v7: Camera %s: Start detection %d %d' %(num_thread, camera_key, h,w))
    #print('Start detection %d %d' %(h,w))
    #diff_time_detect = time.time() - detection_start
    #print('1 Разница %s '% (diff_time_detect))
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))

    try:
        results = model(img)
    except OSError as e:
        logger.error("srvs_threads %s: detect_trash_v7: detection Ошибка работы модели: %s : %s" % (num_thread, img_path, e.strerror))
        exit(-1)

    #diff_time_detect = time.time() - detection_start
    #print('2 Разница %s '% (diff_time_detect))
    isTrash = False
    # Вывод размера изображения
    s = f'изображение {1}: {w}x{h}, '
    #results_n = results.numpy()
    #results_j = results.cpu().tojson()
    #results.cpu().save_txt("images/for_marking/1.txt", save_conf=False)
    im = img
    #diff_time_detect = time.time() - detection_start
    #print('3 Разница %s '% (diff_time_detect))
    for result in results:
        boxes = result.boxes.cpu().numpy()  # get boxes on cpu in numpy
        for box in boxes:  # iterate boxes
            coord = box.xyxy[0].astype(int)  # get corner points as int
            lable = result.names[int(box.cls[0])]
            print(f'{lable}:{int(box.cls[0])} {coord} ')
            if int(box.cls[0]) == 1:
                isTrash = True
                #s += f"{labels_ru[results.names[int(box.cls[0])]]}{'s' * (n > 1)},'
                print(f'{lable}:{int(box.cls[0])} {coord} ')  # print boxes
                cv2.rectangle(im, coord[:2], coord[2:], (255, 0, 0), 6) # draw boxes on img

    n = (boxes.cls == 1).sum()  # detections per class
    dict = {a: 0 for a in range(2)}
    dict[int(0)] = int(f"{n}")
    dict[int(1)] = int(f"{0}")
    s += f"{n} - {labels_ru[results[0].names[int(1)]]}{'s' * (n > 1)}, "  # add to string
    if isTrash:
        save_to_csv(camera_key, path_csv, dict, file_date, file_time)
        if use_db_connection:
            save_to_db(dict, camera_key,  file_date+' '+file_time, num_thread)
    #diff_time_detect = time.time() - detection_start
    #print('4 Разница %s '% (diff_time_detect))

    #im_array = result.plot()  # plot a BGR numpy array of predictions
    #im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
    #im.show()  # show image
    #im.save('results.jpg')  # save image

    text = s.replace(', ', '\n')
    img = draw_text_line(img, (5, 20), text, FONT_SCALE, FONT_FACE, FONT_THICKNESS, LINE_THICKNESS, LINE_COLOR,
                         "custom")
    i = 0
    cv2.imwrite(output_path + '/images_%03d.jpg' % (i), img)

    if isTrash:

        im = Image.fromarray(img[..., ::-1])  # RGB PIL image
        im.show()
    #diff_time_detect = time.time() - detection_start
    #print('5 Разница %s '% (diff_time_detect))

    #    display(im) if is_notebook() else im.show(results.files[i])
    #    img.show()

    # Results
    # print('Print results ')
    # results.print() # or .s
    # print('Print results files ')
    # print(results.files)
    # print(results.ims)
    # print(results.pred)

    # print('Try result tensor ')
    # results.xyxy[0]
    # print('Print Pandas tensor ')
    # results.pandas().xyxy[0]
    # print('Crops ')
    # crops = results.crop(save=True)
    #diff_time_detect = time.time() - detection_start
    #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (
    #num_thread, detection_start, diff_time_detect))

    if isTrash:
        if saveResults_ForMarkings:
            cv2.imwrite("images/for_marking/" + file_name + '.jpg', masked_image)

        extension = img_path.split('.')[-1]
        file_name = img_path.split('/')[-1]

        #if showResults:
        #    results.show()
        if saveResults:
            logger.info('srvs_threads %s: detect_trash_v7: Save analized results ' % (num_thread))
            #results.save(True, path_csv)
            try:
                results.save(True, )
            except OSError as e:
                logger.error("srvs_threads %s: detect_trash_v7: Ошибка сохранения результатов модели: %s : %s" % (num_thread, img_path, e.strerror))
                #print("detection Ошибка сохранения результатов модели: %s : %s" % (img_path, e.strerror))

        # send_text_telegram(token, channel_id, text)
        # send_photo_telegram(token, channel_id, output_path+'/image0.jpg',text)
        try:
            # sended = send_group_img(token, channel_id, output_path, text)
            # sended = send_group_img(token, channel_id_kovnir, output_path, text)
            sended = send_photo_telegram(token, channel_id, output_path+'/images_%03d.jpg' % (i),text)
            if prod_release:
                sended = send_photo_telegram(token, channel_id_kovnir, output_path+'/images_%03d.jpg' % (i),text)
                sended = send_photo_telegram(token, channel_id_jsz, output_path+'/images_%03d.jpg' % (i),text)

            print('Отправляем фото %s :: %s' % (output_path, sended))

            #sended = send_group_img(token, channel_id_kovnir, output_path, text)
        except OSError as e:
            logger.error("srvs_threads %s: detect_trash_v7: Ошибка отправки в бот: %s : %s" % (num_thread, output_path, e.strerror))
            #print("srvs_threads detect_trash_v7: Ошибка отправки в бот: %s : %s" % (output_path, e.strerror))
        addr_to = "oleksandr.maslianchuk@agrichain.ua"  # Получатель
        files = [
            '../yolov5_train_detect/runs/exp/']  # Если нужно отправить все файлы из заданной папки, нужно указать её
        # init_mail.send_email(addr_to, "sugar_trash Привет, тебе сообщение от малинки", "Первіе файлы", files)
    else:
        logger.info("srvs_threads %s: detect_trash_v7: не нашли мусор" % (num_thread))

def apply_nms(boxes, labels, scores, iou_threshold=0.3):
    """Застосовує Non-Maximum Suppression для видалення дублюючих детекцій"""
    if len(boxes) == 0:
        return [], [], []
    
    # Перетворюємо бокси в numpy масив для NMS
    boxes_np = np.array([[box[0][0], box[0][1], box[1][0], box[1][1]] for box in boxes])
    scores_np = np.array(scores)
    
    # Застосовуємо NMS для кожного класу окремо
    unique_labels = list(set(labels))
    keep_indices = []
    
    for label in unique_labels:
        # Знаходимо індекси для поточного класу
        label_indices = [i for i, l in enumerate(labels) if l == label]
        if len(label_indices) == 0:
            continue
            
        # Відбираємо бокси та скори для поточного класу
        label_indices_np = np.array(label_indices)
        label_boxes = boxes_np[label_indices_np]
        label_scores = scores_np[label_indices_np]
        
        # Застосовуємо NMS
        boxes_tensor = torch.from_numpy(label_boxes).float()
        scores_tensor = torch.from_numpy(label_scores).float()
        
        keep = nms(boxes_tensor, scores_tensor, iou_threshold)
        keep = keep.cpu().numpy()
        
        # Додаємо відфільтровані індекси
        for idx in keep:
            keep_indices.append(label_indices[idx])
    
    # Сортуємо індекси
    keep_indices = sorted(keep_indices)
    
    # Повертаємо відфільтровані результати
    filtered_boxes = [boxes[i] for i in keep_indices]
    filtered_labels = [labels[i] for i in keep_indices]
    # Перевіряємо тип scores і правильно індексуємо
    if isinstance(scores, np.ndarray):
        filtered_scores = scores[np.array(keep_indices)].tolist() if len(scores) > 0 else []
    elif isinstance(scores, list):
        filtered_scores = [scores[i] for i in keep_indices] if scores else []
    else:
        filtered_scores = []
    
    return filtered_boxes, filtered_labels, filtered_scores

def get_outputs_v9(image, model, num_thread, camera_key=None, threshold=0.5):
    outputs = model.predict(image, imgsz=1024, conf=threshold, iou=0.3, max_det=3000)

    scores = outputs[0].boxes.conf.detach().cpu().numpy()
    thresholded_indices = [idx for idx, score in enumerate(scores) if score > threshold]
    
    # Перетворюємо список індексів в NumPy масив для правильної індексації
    thresholded_indices_np = np.array(thresholded_indices, dtype=np.int64) if thresholded_indices else np.array([], dtype=np.int64)

    print(f"Total detections: {len(scores)}, Passed threshold: {len(thresholded_indices)}")
    masks, boxes, labels = [], [], []
    try:
        if len(thresholded_indices) > 0 and thresholded_indices_np.size > 0:
            if hasattr(outputs[0], 'masks') and outputs[0].masks is not None:
                masks = [outputs[0].masks.xy[idx] for idx in thresholded_indices]
            
            boxes = outputs[0].boxes.xyxy.detach().cpu().numpy()[thresholded_indices_np]
            boxes = [[(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))] for box in boxes]
            labels = [outputs[0].names[int(outputs[0].boxes.cls[idx])] for idx in thresholded_indices]
            camera_info = f"Camera {camera_key}: " if camera_key else ""
            print(f"srvs_threads {num_thread}: detect_trash_v9: {camera_info}{labels}")
            print(f"srvs_threads {num_thread}: detect_trash_v9: {camera_info}{boxes}")

            # Сортировка результатів тільки якщо всі списки мають однакову довжину
            if masks and len(masks) == len(boxes) == len(labels):
                combined = sorted(zip(masks, boxes, labels), key=lambda x: (x[1][0][1], x[1][0][0]))
                if combined:
                    masks, boxes, labels = zip(*combined)
                    masks, boxes, labels = list(masks), list(boxes), list(labels)
            elif boxes and labels:  # Якщо немає масок, але є бокси та мітки
                # Сортуємо тільки бокси та мітки
                combined = sorted(zip(boxes, labels), key=lambda x: (x[0][0][1], x[0][0][0]))
                if combined:
                    sorted_results = list(zip(*combined))
                    boxes, labels = list(sorted_results[0]), list(sorted_results[1])
            
            # NMS ВІДКЛЮЧЕНО - працює без нього
            # # Застосовуємо додатковий NMS для видалення дублюючих детекцій
            # if boxes and labels:
            #     logger.info(f"Before NMS: {len(boxes)} boxes, {len(labels)} labels, {len(scores[thresholded_indices_np])} scores")
            #     print(f"[DEBUG] Before NMS: boxes={len(boxes)}, labels={len(labels)}, scores shape={scores[thresholded_indices_np].shape}")
            #     try:
            #         boxes, labels, _ = apply_nms(boxes, labels, scores[thresholded_indices_np], iou_threshold=0.3)
            #         logger.info(f"After NMS: {len(boxes)} detections remaining")
            #         print(f"[DEBUG] After NMS: {len(boxes)} detections")
            #     except Exception as nms_error:
            #         logger.error(f"NMS error: {str(nms_error)}")
            #         print(f"[ERROR] NMS failed: {str(nms_error)}")
            #         import traceback
            #         logger.error(f"NMS Traceback: {traceback.format_exc()}")
            #         print(f"NMS Traceback: {traceback.format_exc()}")
            #         # Продовжуємо без NMS якщо він не спрацював
                
    except Exception as e:  # Исправил тип исключения на более общий
        import traceback
        print(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка получения данных: {str(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка получения данных: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Логування перед поверненням
    logger.info(f"[DEBUG get_outputs_v9] Returning: masks={len(masks) if masks else 0}, boxes={len(boxes) if boxes else 0}, labels={len(labels) if labels else 0}")
    return masks, boxes, labels


def detect_trash_v9(camera_key: str, img_path: str, num_thread, threshold=0.5):
    detection_start = time.time()
    # Получение имени файла и даты из пути
    if os.path.isfile(img_path):
        file_name = os.path.basename(img_path).split('.')[0]
        file_date_time = file_name[11:]
        logger.info(f"srvs_threads {num_thread}: file_date_time: {file_date_time}")
        try:
            # Проверяем, что длина строки соответствует формату даты
            if len(file_date_time) >= 15:  # Ожидаем формат YYYYMMDD_HH_MM_SS
                # Определение даты и времени файла на основе имени
                file_date = f"{file_date_time[:4]}-{file_date_time[4:6]}-{file_date_time[6:8]}"  # Формат YYYY-MM-DD
                file_time = file_date_time[9:].replace("_", ":", 2).replace("_", ".")  # Формат HH:MM:SS.milliseconds
                logger.info(f"srvs_threads {num_thread}: file_date_time: {file_date_time} {file_date} {file_time}")
            else:
                raise ValueError(f"Некорректная длина строки для даты и времени: {file_date_time} ")

        except IndexError as e:
            logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка обработки имени файла для даты и времени: {file_name}: {str(e)}")
            return
    else:
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Файл не найден: {img_path}")
        return

    try:
        img = cv2.imread(img_path)
        if img is None:
            logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка чтения изображения: {img_path}")
            return
        h, w = img.shape[:2]
    except OSError as e:
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка чтения изображения: {img_path} : {e.strerror}")
        return

    # Маскирование изображения
    if camera_key=='JDN':
        # Створюємо ЧОРНУ маску (прибираємо все спочатку)
        mask = np.zeros(img.shape, dtype=np.uint8)
        
        # Малюємо БІЛИЙ полігон для транспортерної стрічки (ЗАЛИШАЄМО область з цукром для аналізу)
        # Координати охоплюють всю білу область в центрі фото (розмір 2048x1536)
        # Формат: (x, y) де x - горизонталь (0-2048), y - вертикаль (0-1536)
        roi_corners = np.array([
            [(200, 1190), (190, 325), (1537, 225), (1820, 833)]  # Прямокутник навколо білої транспортерної стрічки
        ], dtype=np.int32)
        roi_mask_color = (255,) * img.shape[2]  # Білий колір - залишаємо для аналізу
        cv2.fillPoly(mask, roi_corners, roi_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Applied mask for sugar conveyor belt')
        
        # Збереження маскованого зображення для перевірки (окремий параметр)
        if SAVE_MASKED_IMAGES:
            debug_mask_path = f"images/for_marking_mask/{file_name}.jpg"
            cv2.imwrite(debug_mask_path, masked_image)
            logger.info(f'srvs_threads {num_thread}: DEBUG: Saved masked image to {debug_mask_path}')
    elif camera_key=='JCZ':
        # Створюємо ЧОРНУ маску (прибираємо все спочатку)
        mask = np.zeros(img.shape, dtype=np.uint8)
        
        # Малюємо БІЛИЙ полігон для транспортерної стрічки (ЗАЛИШАЄМО область з цукром для аналізу)
        # Координати для камери JCZ
        roi_corners = np.array([[(1280, 1920), (1400, 225), (2420, 225), (2545, 1920)]], dtype=np.int32)
        roi_mask_color = (255,) * img.shape[2]  # Білий колір - залишаємо для аналізу
        cv2.fillPoly(mask, roi_corners, roi_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Applied mask for sugar conveyor belt')
        
        # Збереження маскованого зображення для перевірки (окремий параметр)
        if SAVE_MASKED_IMAGES:
            debug_mask_path = f"images/for_marking_mask/{file_name}.jpg"
            cv2.imwrite(debug_mask_path, masked_image)
            logger.info(f'srvs_threads {num_thread}: DEBUG: Saved masked image to {debug_mask_path}')
    elif camera_key=='NRK':
        # Створюємо ЧОРНУ маску (прибираємо все спочатку)
        mask = np.zeros(img.shape, dtype=np.uint8)
        
        # Малюємо БІЛИЙ полігон для транспортерної стрічки (ЗАЛИШАЄМО область з цукром для аналізу)
        # Координати для камери NRK
        roi_corners = np.array([[(940, 2140), (1500, 700), (2400, 700), (3000, 2150)]], dtype=np.int32)
        roi_mask_color = (255,) * img.shape[2]  # Білий колір - залишаємо для аналізу
        cv2.fillPoly(mask, roi_corners, roi_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Applied mask for sugar conveyor belt')
        
        # Збереження маскованого зображення для перевірки (окремий параметр)
        if SAVE_MASKED_IMAGES:
            debug_mask_path = f"images/for_marking_mask/{file_name}.jpg"
            cv2.imwrite(debug_mask_path, masked_image)
            logger.info(f'srvs_threads {num_thread}: DEBUG: Saved masked image to {debug_mask_path}')
    elif camera_key=='NVO':
        # Створюємо ЧОРНУ маску (прибираємо все спочатку)
        mask = np.zeros(img.shape, dtype=np.uint8)
        
        # Малюємо БІЛИЙ полігон для транспортерної стрічки (ЗАЛИШАЄМО область з цукром для аналізу)
        # Координати для камери NVO
        roi_corners = np.array([[(540, 1070), (600, 570), (1500, 570), (1570, 1070)]], dtype=np.int32)
        roi_mask_color = (255,) * img.shape[2]  # Білий колір - залишаємо для аналізу
        cv2.fillPoly(mask, roi_corners, roi_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Applied mask for sugar conveyor belt')
        
        # Збереження маскованого зображення для перевірки (окремий параметр)
        if SAVE_MASKED_IMAGES:
            debug_mask_path = f"images/for_marking_mask/{file_name}.jpg"
            cv2.imwrite(debug_mask_path, masked_image)
            logger.info(f'srvs_threads {num_thread}: DEBUG: Saved masked image to {debug_mask_path}')
    elif camera_key=='GLO':
        # Створюємо ЧОРНУ маску (прибираємо все спочатку)
        mask = np.zeros(img.shape, dtype=np.uint8)
        
        # Малюємо БІЛИЙ полігон для транспортерної стрічки (ЗАЛИШАЄМО область з цукром для аналізу)
        # Координати для камери GLO
        roi_corners = np.array([[(220, 1435), (465, 500), (1930, 455), (2100, 1435)]], dtype=np.int32)
        roi_mask_color = (255,) * img.shape[2]  # Білий колір - залишаємо для аналізу
        cv2.fillPoly(mask, roi_corners, roi_mask_color)

        masked_image = cv2.bitwise_and(img, mask)
        logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Applied mask for sugar conveyor belt')
        
        # Збереження маскованого зображення для перевірки (окремий параметр)
        if SAVE_MASKED_IMAGES:
            debug_mask_path = f"images/for_marking_mask/{file_name}.jpg"
            cv2.imwrite(debug_mask_path, masked_image)
            logger.info(f'srvs_threads {num_thread}: DEBUG: Saved masked image to {debug_mask_path}')
    else:
        masked_image = img
    #masked_image = img
    detection_current = time.time()
    # Работайте с маскированным изображением в памяти, без повторного сохранения/чтения
    logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Start detection {h}x{w}  current {detection_current-detection_start:.2f}  сек ')


    try:
        # Выполнение модели на маскированном изображении
        logger.info(f"[DEBUG detect_trash_v9] Thread {num_thread}: BEFORE get_outputs_v9 call")
        masks, boxes, labels = get_outputs_v9(masked_image, model, num_thread, camera_key=camera_key, threshold=threshold)
        logger.info(f"[DEBUG detect_trash_v9] Thread {num_thread}: AFTER get_outputs_v9 call: got {len(boxes) if boxes else 0} boxes, {len(labels) if labels else 0} labels")
        
        detection_current = time.time()
        logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Camera {camera_key}: Результаты модели boxes={len(boxes)}, labels={len(labels)}   current  {detection_current-detection_start:.2f}  сек ' )

        if not boxes or not labels:
            logger.info(f"[DEBUG detect_trash_v9] Thread {num_thread}: No boxes or labels, returning")
            logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Модель не нашла ни одного объекта")
            return

        if len(boxes) != len(labels):
            logger.info(f"[DEBUG detect_trash_v9] Thread {num_thread}: boxes/labels length mismatch, returning")
            logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Некорректные данные от модели, количество объектов и меток не совпадает")
            return
        
        logger.info(f"[DEBUG detect_trash_v9] Thread {num_thread}: Proceeding to process results...")
    except Exception as e:
        logger.error(f"[DEBUG detect_trash_v9] Thread {num_thread}: Exception caught: {str(e)}")
        import traceback
        logger.error(f"[DEBUG detect_trash_v9] Thread {num_thread}: Traceback: {traceback.format_exc()}")
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка работы модели: {img_path} : {str(e)}")
        return

    # Обработка результатов
    isTrash = False  # За замовчуванням False
    s = f'{camera_key} зображення {1}: {w}x{h}, '
    im = masked_image.copy()

    logger.info(f"[{camera_key}] Processing {len(boxes)} detections: {labels}")
    
    for box, label in zip(boxes, labels):
        if label == "trash":
            isTrash = True
            logger.info(f"[{camera_key}] [OK] TRASH detected at {box}")
            cv2.rectangle(im, box[0], box[1], (255, 0, 0), 6)
        elif label == "arm":  # Також позначаємо руки
            cv2.rectangle(im, box[0], box[1], (0, 255, 0), 4)  # Зелений прямокутник для руки

    n_trash = labels.count("trash")
    n_arm = labels.count("arm")
    s += f"домишки: {n_trash}"
    
    logger.info(f"[{camera_key}] Detection summary: trash={n_trash}, arm={n_arm}, isTrash={isTrash}")

    if isTrash:
        try:
            save_to_csv(camera_key, path_csv, {0: n_trash, 1: 0}, file_date, file_time)
            logger.info(f"srvs_threads {num_thread}: CSV saved successfully")
        except Exception as e:
            logger.error(f"srvs_threads {num_thread}: Error saving to CSV: {e}")
            import traceback
            logger.error(f"srvs_threads {num_thread}: CSV Traceback: {traceback.format_exc()}")
        
        if use_db_connection:
            try:
                save_to_db({ 0: n_trash, 1: 0}, camera_key, f'{file_date} {file_time}', num_thread)
                logger.info(f"srvs_threads {num_thread}: DB saved successfully")
            except Exception as e:
                logger.error(f"srvs_threads {num_thread}: Error saving to DB: {e}")

    # Добавление текста на изображение
    text = s.replace(', ', '\n')
    img_with_text = draw_text_line(im, (5, 20), text, FONT_SCALE, FONT_FACE, FONT_THICKNESS, LINE_THICKNESS, LINE_COLOR, "custom")

    # Сохранение изображений только один раз
    try:
        # Створюємо папку якщо не існує
        os.makedirs(output_path, exist_ok=True)
        output_image_path = os.path.join(output_path, f'images_{num_thread:03d}.jpg')
        cv2.imwrite(output_image_path, img_with_text)
        logger.info(f"srvs_threads {num_thread}: Saved output image: {output_image_path}")
    except Exception as e:
        logger.error(f"srvs_threads {num_thread}: ERROR saving output image: {e}")
        import traceback
        logger.error(f"srvs_threads {num_thread}: Traceback: {traceback.format_exc()}")
    
    # Зберігаємо фото у всі папки якщо знайдено мусор
    if isTrash:  # Тільки для trash
        # Зберігаємо розмічене фото в папку detected
        if saveDetectedImages:
            try:
                os.makedirs("images/detected", exist_ok=True)
                detected_path = f"images/detected/{file_name}.jpg"
                cv2.imwrite(detected_path, img_with_text)
                logger.info(f"Saved detected image: {detected_path}")
            except Exception as e:
                logger.error(f"Error saving detected image: {e}")
        
        # Зберігаємо оригінальне фото в папку for_marking
        if saveResults_ForMarkings:
            try:
                os.makedirs("images/for_marking", exist_ok=True)
                save_path = f"images/for_marking/{file_name}.jpg"
                cv2.imwrite(save_path, img)  # Зберігаємо оригінальне зображення
                logger.info(f"Saved for_marking: {file_name}.jpg (trash: {n_trash}, arm: {n_arm})")
                if not os.path.exists(save_path):
                    logger.error(f"Failed to save file to {save_path}")
            except Exception as e:
                logger.error(f"Error saving for_marking: {e}")
        
        # Зберігаємо масковане фото в папку for_marking_mask
        if SAVE_MASKED_IMAGES:
            try:
                os.makedirs("images/for_marking_mask", exist_ok=True)
                mask_path = f"images/for_marking_mask/{file_name}.jpg"
                cv2.imwrite(mask_path, masked_image)
                logger.info(f"Saved for_marking_mask: {file_name}.jpg")
            except Exception as e:
                logger.error(f"Error saving for_marking_mask: {e}")
    else:
        logger.info(f"NOT saving images: isTrash={isTrash}, saveResults_ForMarkings={saveResults_ForMarkings}, saveDetectedImages={saveDetectedImages}")

    # Отправка результатов только при наличии мусора
    logger.info(f"[{camera_key}] srvs_threads {num_thread}: isTrash={isTrash}, n_trash={n_trash}, n_arm={n_arm}")
    
    if isTrash:
        logger.info(f"[{camera_key}] srvs_threads {num_thread}: [OK] TRASH DETECTED! Starting notifications...")
        
        # Step 1: Telegram notifications (independent, failures won't block PBI)
        logger.info(f"[{camera_key}] srvs_threads {num_thread}: Step 1/3: Sending to Telegram channels...")
        telegram_success_count = 0
        
        # Main channel
        try:
            if not os.path.exists(output_image_path):
                logger.error(f"[{camera_key}] [ERROR] Output image file not found: {output_image_path}")
                raise FileNotFoundError(f"Output image file not found: {output_image_path}")
            logger.info(f"[{camera_key}] Sending to main channel: {channel_id}, file: {output_image_path}")
            send_photo_telegram(token, channel_id, output_image_path, text)
            logger.info(f"[{camera_key}] [OK] Main channel sent")
            telegram_success_count += 1
        except Exception as e:
            logger.error(f"[{camera_key}] [ERROR] Failed to send to main channel: {str(e)}")
        
        # Kovnir channel
        try:
            logger.info(f"[{camera_key}] Sending to Kovnir channel: {channel_id_kovnir}")
            send_photo_telegram(token, channel_id_kovnir, output_image_path, text)
            logger.info(f"[{camera_key}] [OK] Kovnir channel sent")
            telegram_success_count += 1
        except Exception as e:
            logger.error(f"[{camera_key}] [ERROR] Failed to send to Kovnir channel: {str(e)}")
        
        # JSZ channel (JDN only)
        if camera_key == 'JDN':
            try:
                logger.info(f"[{camera_key}] Sending to JSZ channel: {channel_id_jsz}")
                send_photo_telegram(token, channel_id_jsz, output_image_path, text)
                logger.info(f"[{camera_key}] [OK] JSZ channel sent")
                telegram_success_count += 1
            except Exception as e:
                logger.error(f"[{camera_key}] [ERROR] Failed to send to JSZ channel: {str(e)}")
        
        logger.info(f"[{camera_key}] srvs_threads {num_thread}: Step 1/3 Complete: {telegram_success_count} Telegram channel(s) sent successfully")
        
        # Step 2: Power BI push (ALWAYS attempt, even if Telegram failed)
        try:
            logger.info(f"[{camera_key}] srvs_threads {num_thread}: Step 2/3: Sending data to Power BI...")
            logger.info(f"[{camera_key}] PBI data: camera={camera_key}, trash_count={n_trash}, datetime={file_date} {file_time}")
            send_to_PBI(camera_key, n_trash, f'{file_date} {file_time}')
            logger.info(f"[{camera_key}] srvs_threads {num_thread}: [OK] Step 2/3 Complete: Power BI sent")
        except Exception as e:
            logger.error(f"[{camera_key}] srvs_threads {num_thread}: [ERROR] ERROR sending to Power BI: {str(e)}")
            import traceback
            logger.error(f"[{camera_key}] PBI Traceback: {traceback.format_exc()}")
        
        # Final log
        logger.info(f"[{camera_key}] srvs_threads {num_thread}: [OK] Step 3/3 Complete: Notification process finished for {output_image_path}")

    if not isTrash:
        detection_current = time.time()
        logger.info(f"srvs_threads {num_thread}: Не нашли мусор, end {detection_current-detection_start:.2f}  сек " )
    isTrash = False

def send_to_PBI(camera_key, n_trash, current_datetime_str):
    logger.info(f"[PBI] Starting Power BI data send for camera {camera_key}")
    print(f"[PBI] Preparing to send data to Power BI: {camera_key}, trash={n_trash}")
    
    target_datetime_str = repl_to_local_timezone(current_datetime_str)
    logger.info(f"[PBI] Converted timestamp: {current_datetime_str} -> {target_datetime_str}")

    url = "https://api.powerbi.com/beta/3e7ffad5-67c6-4cca-9d21-51e043f024dc/datasets/8c758886-05f5-48a9-98e9-ac9d22d4d397/rows?experience=power-bi&key=1g%2F82De%2FQ7oHLK0TdpZZPWTWAD88OFBSkhs8b2SQ%2Fe6ZmNm7is9Y%2BRDb8%2FiNVDnnUd26LXCWoOY3kh4%2FCMW5XA%3D%3D"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiIsIng1dCI6InJ0c0ZULWItN0x1WTdEVlllU05LY0lKN1ZuYyIsImtpZCI6InJ0c0ZULWItN0x1WTdEVlllU05LY0lKN1ZuYyJ9.eyJhdWQiOiJodHRwczovL2FuYWx5c2lzLndpbmRvd3MubmV0L3Bvd2VyYmkvYXBpIiwiaXNzIjoiaHR0cHM6Ly9zdHMud2luZG93cy5uZXQvM2U3ZmZhZDUtNjdjNi00Y2NhLTlkMjEtNTFlMDQzZjAyNGRjLyIsImlhdCI6MTc2MzA0MTcyNCwibmJmIjoxNzYzMDQxNzI0LCJleHAiOjE3NjMwNDU2MjQsImFpbyI6ImsySmdZSmowb3YzakpLZlR1Njh5YzIrYzh2VlJFQUE9IiwiYXBwaWQiOiI4NTE5ZWE3Yi1jMWRkLTQ1YzQtOGJmMS0xYTQ4MWIyMmNjNjgiLCJhcHBpZGFjciI6IjEiLCJpZHAiOiJodHRwczovL3N0cy53aW5kb3dzLm5ldC8zZTdmZmFkNS02N2M2LTRjY2EtOWQyMS01MWUwNDNmMDI0ZGMvIiwiaWR0eXAiOiJhcHAiLCJvaWQiOiI4MGY1Nzg3MC02N2E5LTRkYTktYWU0NC0yNzFhNWU1NjA0NjAiLCJyaCI6IjEuQVhNQTFmcF9Qc1pueWt5ZElWSGdRX0FrM0FrQUFBQUFBQUFBd0FBQUFBQUFBQUJ6QUFCekFBLiIsInJvbGVzIjpbIlRlbmFudC5SZWFkV3JpdGUuQWxsIiwiVGVuYW50LlJlYWQuQWxsIl0sInN1YiI6IjgwZjU3ODcwLTY3YTktNGRhOS1hZTQ0LTI3MWE1ZTU2MDQ2MCIsInRpZCI6IjNlN2ZmYWQ1LTY3YzYtNGNjYS05ZDIxLTUxZTA0M2YwMjRkYyIsInV0aSI6IlRaSGc3dVE1NDBtSzNNNVk0ZkVRQUEiLCJ2ZXIiOiIxLjAiLCJ4bXNfYWN0X2ZjdCI6IjMgOSIsInhtc19mdGQiOiJxMEtWRzQ4YlZPREhHVjdzTFZnSkdRYVhPd3k0RndTZHZmc25KdFpuYXJRQlpuSmhibU5sWXkxa2MyMXoiLCJ4bXNfaWRyZWwiOiI3IDE4IiwieG1zX3JkIjoiMC40MkxsWUJKaUxCWVM0V0FYRXJpeUxfaFI1dEc1dmt2NnNuZVhpOGo4QjRweUNnbFlkdGF4aUh4YzdOYTc0ZUNhUmZxYnRJQ2lIRUlDbkF3UWNBQktBd0EiLCJ4bXNfc3ViX2ZjdCI6IjMgOSJ9.Wyr40bbpZjs00uZ9brbdvbgJ5pKMtPlDoxdilQ9vf9N4dl4c3GLM4N-Av9c2N28MRba93z4-U-IHBad4rf8i9b_TmhdI9AybVAkF5RjnKAFq4ExeKTRSEKDAlt6jOV5A19DokDw0kTGmdgiYwwFjWe31-RyAFQl6_A7Et-Sf00pHjNJ9900UQPUaqQA1L8nlBIJMX1IhSMG9GQTBzSReSbG_A3GKB-Qyma8Bc27Bc360v5ZUbJb9wmQsx-GE1j2BZy91mDOC3YFykHNrkNNYZprLcq9io1XM-dkZBZSBe9NAmLOHW1FUOyOxsBM8APPX4YD_cEeGjkubEqITKmPrRw"
    }
    
    factory_location = cameras.get(camera_key, {}).get("Location", "Неизвестно")
    data = {
        "trash": n_trash,
        "arm": 0,
        "timestamp": target_datetime_str,
        "factory_id": factory_location
    }
    
    logger.info(f"[PBI] Payload: camera={camera_key}, location={factory_location}, trash={n_trash}, timestamp={target_datetime_str}")
    print(f"[PBI] Data payload: {data}")
    #if camera_key=='NRK':
    #    exit()
    retries = 5
    for attempt in range(retries):
        logger.info(f"[PBI] Attempt {attempt + 1}/{retries}: Sending POST request to Power BI...")
        print(f"[PBI] Attempt {attempt + 1}/{retries}: Sending data...")
        try:
            response = requests.post(url, headers=headers, data=json.dumps([data]))
            logger.info(f"[PBI] Response received: status_code={response.status_code}")
            print(f"[PBI] Response: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"[PBI] [OK] Data successfully sent to Power BI for camera {camera_key}")
                print(f"[PBI] [OK] Successfully sent: {camera_key} - {factory_location}, trash={n_trash}")
                return
            elif response.status_code == 429:
                wait_time = 2 ** attempt
                logger.warning(f"[PBI] Rate limit exceeded (429). Retry {attempt + 1}/{retries}, waiting {wait_time}s...")
                print(f"[PBI] Rate limit hit, waiting {wait_time}s before retry...")
                time.sleep(wait_time)  # Экспоненциальная задержка
            else:
                logger.error(f"[PBI] [ERROR] Error sending data: status={response.status_code}, response={response.text}")
                print(f"[PBI] [ERROR] Error {response.status_code}: {response.text[:200]}")
                break
        except requests.exceptions.RequestException as e:
            logger.error(f"[PBI] [ERROR] Network error on attempt {attempt + 1}: {str(e)}")
            print(f"[PBI] [ERROR] Network error: {str(e)}")
            if attempt < retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"[PBI] Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
    
    logger.error(f"[PBI] [ERROR] FAILED to send data to Power BI after {retries} attempts for camera {camera_key}")
    print(f"[PBI] [ERROR] FAILED after {retries} attempts: {camera_key}")

def repl_to_local_timezone(current_datetime_str, tz_original='UTC', tz_target="Europe/Kyiv"):
    original_timezone = pytz.timezone(tz_original)  # Замените на вашу исходную временную зону

    # Целевая временная зона (например, Europe/Moscow или другая)
    target_timezone = pytz.timezone(tz_target)
    
    # Значення за замовчуванням на випадок помилки
    target_datetime_str = current_datetime_str

    try:
        # Парсим строку в объект datetime
        # Спробуємо спочатку з мілісекундами
        try:
            naive_datetime = datetime.strptime(current_datetime_str, "%Y-%m-%d %H:%M:%S.%f")
        except ValueError:
            # Якщо не вдалось, спробуємо без мілісекунд
            naive_datetime = datetime.strptime(current_datetime_str, "%Y-%m-%d %H:%M:%S")

        # Привязываем исходную временную зону
        original_datetime = original_timezone.localize(naive_datetime)
        #print(f"Дата и время в оригинальной временной зоне: {original_datetime}")

        # Переводим в целевую временную зону
        target_datetime = original_datetime.astimezone(target_timezone)

        # Результат в строковом формате
        target_datetime_str = target_datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"Дата и время в целевой временной зоне: {target_datetime_str}")

    except Exception as e:
        print(f"Ошибка преобразования даты и времени: {e}")

    return target_datetime_str

def count_line_csv(filename):
    with open(filename) as f:
        return sum(1 for line in f)

def generate_version() -> int:
    return int(time.time() * 1000)

def save_to_db(data: dict, camera_key: str, timestamp: str, arm_id: int):
    db: Session = SessionLocal()
    try:
        version = generate_version()
        new_record = detection_sugar_inclusions(
            version=version,
            factory_id=cameras.get(camera_key, {}).get("Location", "Неизвестно"),
            created=timestamp,
            updated=timestamp,
            trash=data.get(0, 0),
            not_trash=0,
            arm=data.get(1, 0)
        )
        db.add(new_record)
        db.commit()
        logger.info(f"Data saved to DB: {version} {timestamp} {data.get(0, 0)} {data.get(1, 0)}")
        print(f"Data saved to DB: {version} {timestamp} {data.get(0, 0)} {data.get(1, 0)}")
    except Exception as e:
        logger.error(f"Error saving data to DB: {e}")
        print(f"Error saving data to DB: {e}")
        db.rollback()
    finally:
        db.close()

def save_to_csv(camera_key: str, path_csv: str,pred,file_date,file_time):
    file_name = camera_key+'_'+file_date.replace('-','_')+'.csv'
    # file_name = datetime.now().strftime('%Y_%m_%d')+'.csv'
    path_csv = path_csv + file_name
    if os.path.exists(path_csv):
        create = 'a'
    else:
        create = 'w'
    with open(path_csv, create, newline='') as csvfile:
        #fieldnames = ['date', 'time', 'red_trash','sugar_trash','metal_trash','hiden_trash','partlihiden_trash','rag_trash','rope_trash']
        #fieldnames = ['date', 'time', 'trash','not_trash','arm']
        if model_version == 5:
            fieldnames = ['date', 'time', 'trash','not_trash']
        elif model_version == 8:
            fieldnames = ['date', 'time', 'trash','nonTrash']
        else:
            fieldnames = ['date', 'time', 'trash', 'arm']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
       # writer = csv.writer(csvfile)
        if count_line_csv(path_csv) == 0:
            writer.writeheader()
        #writer.writerow({'date':file_date, 'time': file_time, 'red_trash': pred[0], 'sugar_trash': pred[1],
        #                 'metal_trash': pred[2], 'hiden_trash': pred[3], 'partlihiden_trash': pred[4], 'rag_trash':pred[5], 'rope_trash':pred[0]})
        #writer.writerow({'date':file_date, 'time': file_time, 'trash': pred[0], 'not_trash': pred[1],
        #                 'arm': pred[2]})
        if model_version == 5:
            writer.writerow({'date':file_date, 'time': file_time, 'trash': pred[0], 'not_trash': pred[1]})
        elif model_version == 8:
            writer.writerow({'date': file_date, 'time': file_time, 'trash': pred[0], 'nonTrash': pred[1]})
        else:
            writer.writerow({'date':file_date, 'time': file_time, 'trash': pred[0], 'arm': pred[1]})

def htmlColorToJSON(htmlColor):
    if htmlColor.startswith("#"):
        htmlColor = htmlColor[1:]
    return {"red": int(htmlColor[0:2], 16) / 255.0, "green": int(htmlColor[2:4], 16) / 255.0, "blue": int(htmlColor[4:6], 16) / 255.0}

def save_to_google_sheets(pred):
    docTitle = 'Документ по включениям мусора на ЖСЗ'
    docKEY = '1CCqz0dd0Zg6Hid2STravPP1Ca-NAYwKSZ9IBJBvOykU'
    sheetTitle = '' + datetime.now().strftime('%Y_%m_%d')
    values = [['date', 'time', 'trash', 'arm', 'low_production'],  # header row
              [datetime.now().strftime('%Y-%m_%d'), datetime.now().time(), pred[0], pred[1], pred[2], ]]
    rowCount = len(values) - 1
    colorsForCategories = {"trash": htmlColorToJSON("#FFCCCC"),
                           "arm": htmlColorToJSON("#CCFFCC"),
                           "nonTrash": htmlColorToJSON("#CCCCFF")}

    SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    # gc = pygsheets.authorize(service_file=GOOGLE_CREDENTIALS_FILE)

    ## Open spreadsheet and then worksheet
    ##sh = gc.open_by_key(docKEY)

    #sh = gc.open(docTitle)
    ##sh.share("oleksandr.maslianchuk@agrichain.ua")
    #try:
        #wks = sh.worksheet_by_title(sheetTitle)
    #except:
        #wks = sh.add_worksheet(sheetTitle)

    # share the sheet with your friend



class Main_Thread(Thread):

    def __init__(self, cam_img_path, period_rotation):
        Thread.__init__(self)
        self._cam_img_path = cam_img_path
        self._period_rotation = period_rotation

    def run(self):
        while True:
            logger.error("")

            cam_img_path = self._cam_img_path  + 'screen'+datetime.now().strftime('%Y%m%d_%H_%M_%S_%f')+'.jpg'
            sleep_start = time.time()
            try:
                if type_cam == 'hikvision':
                    response = cam.Streaming.channels[1].picture(method='get', type='opaque_data')
                    with open(cam_img_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=1024):
                            if chunk:
                                f.write(chunk)

                        f.close()
                else:
                    api_endpoint = "/cgi-bin/snapshot.cgi"
                    url = f"http://{camera_ip}{api_endpoint}"

                    auth = requests.auth.HTTPDigestAuth(username, password)

                    response = requests.get(url, auth=auth)

                    if response.status_code == 200:
                        with open(cam_img_path, "wb") as f:
                            f.write(response.content)
                        print("Снимок успешно сохранен")
                    else:
                        print(f"Ошибка: {response.status_code}")
            except OSError as e:
                logger.error("Main_Thread, Camera: Ошибка запроса get в камеру  %s " % (e.strerror))
                send_text_telegram(token, channel_id,"Camera не доступна:  %s " % (e.strerror))
                if prod_release:
                    send_text_telegram(token, channel_id_kovnir,"Camera не доступна:  %s " % (e.strerror))
                    send_text_telegram(token, channel_id_jsz, "Camera не доступна:  %s " % (e.strerror))
                continue
            time.sleep(self._period_rotation)
            sleep_end = time.time()
            diff_time_request = sleep_end - sleep_start
            if os.path.exists(cam_img_path):
                logger.info('Main_Thread, Camera: записали файл: %s' %(cam_img_path))
            logger.info('Main_Thread, Camera: плановый период:: %s затраченное время:: %s разница:: %s' % (
            self._period_rotation, diff_time_request, diff_time_request - self._period_rotation))

class Service_Thread(Thread):
    def __init__(self, cam_img_path, path_for_detections,num_thread):
        Thread.__init__(self)
        self._cam_img_path = cam_img_path
        self._path_for_detections = path_for_detections
        self._num_thread = num_thread

    def run(self):
        while True:

            detection_start = time.time()
            #diff_time_detect = time.time() - detection_start
            #logger.info('detection %s: время начала процедуры детектирования:: %s : %s' % (self._num_thread,detection_start, diff_time_detect))
            new_path_file = ''
            try:
                new_path_file = work_with_files.sort_files_by_date(self._cam_img_path, self._path_for_detections)
                #diff_time_detect = time.time() - detection_start
                #logger.info('detection %s: время после передачи файла между папками:: %s : %s' % (self._num_thread, detection_start, diff_time_detect))
            except OSError as e:
                i=0
                #print("detection prepare %s: не нашли самый старый файл: %s" % (self._num_thread, e.strerror))
                diff_time_detect = time.time() - detection_start
                logger.error('srvs_threads: detection %s: время после передачи файла между папками:: %s : %s  %s' % (self._num_thread, detection_start, diff_time_detect, e.strerror))
            if not new_path_file == '':
                if model_version == 5:
                    detect_trash_v5(new_path_file,self._num_thread, detection_start)
                elif model_version == 8:
                    detect_trash_v7(new_path_file, self._num_thread, detection_start)
                else:
                    # NOTE: This Service_Thread is deprecated, use continuous_capture_images instead
                    # Passing 'UNKNOWN' as camera_key since Service_Thread doesn't have it
                    detect_trash_v9('UNKNOWN', new_path_file, self._num_thread, 0.5)
                detection_end = time.time()
                diff_time_detect = detection_end-detection_start
                logger.info('srvs_threads: detection %s: затраченное время на распознавание:: %s' %(self._num_thread, diff_time_detect) )
                #print('detection %s: затраченное время на распознавание:: %s' %(self._num_thread, diff_time_detect) )

                try:
                    os.remove(new_path_file)
                    logger.error("srvs_threads: detection %s: Удалили файла: %s : " % (self._num_thread, new_path_file))
                except OSError as e:
                    logger.error("srvs_threads: detection %s: Ошибка удаления файла: %s : %s" % (self._num_thread, new_path_file, e.strerror))
                    #print("detection %s: Ошибкаудаления файла: %s : %s" % (self._num_thread, new_path_file, e.strerror))
                if os.path.exists(f'{detect_path}'):
                    try:
                        if model_version == 5:
                            shutil.rmtree(detect_path)
                            logger.error("detection")
                    except OSError as e:
                        logger.error("srvs_threads: detection %s: Ошибка удаления директории: %s : %s" % (self._num_thread, detect_path, e.strerror))
                        #print("detection %s: Ошибка удаления директории: %s : %s" % (self._num_thread, detect_path, e.strerror))

class SendResults_Thread(Thread):
    def __init__(self, path_csv, csv_rotation):
        Thread.__init__(self)
        self._path_csv = path_csv
        self._csv_rotation = csv_rotation

    def run(self):
        while True:

            file_name = datetime.now().strftime('%Y_%m_%d') + '.csv'
            _path_csv = self._path_csv + file_name
            if os.path.exists(_path_csv):
                send_document_telegram(token, channel_id, _path_csv, file_name)
                if prod_release:
                    send_document_telegram(token, channel_id_kovnir, _path_csv, file_name)
                    send_document_telegram(token, channel_id_jsz, _path_csv, file_name)

            else:
                send_text_telegram(token, channel_id,'за текущую дату отсутствует файл с данными')
            time.sleep(self._csv_rotation)


if torch.cuda.is_available():
    print("CUDA доступна")
else:
    print("CUDA не доступна")

# Налаштування логування
logger = logging.getLogger(__name__)
log_level = getattr(logging, LOG_LEVEL, logging.INFO)
logger.setLevel(log_level)

# Створюємо handler для виводу в консоль
console_handler = logging.StreamHandler()
console_handler.setLevel(log_level)

# Формат логування
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Додаємо handler до logger (якщо ще не доданий)
if not logger.handlers:
    logger.addHandler(console_handler)

logger.info(f"Logging configured with level: {LOG_LEVEL}")

# Контроль використання БД через змінну оточення
use_db_connection = USE_DB
logger.info(f"Database connection enabled: {use_db_connection}")

# Ініціалізація БД тільки якщо потрібно
if use_db_connection:
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        use_db_connection = False
        logger.warning("Continuing without database connection")

model_version = MODEL_VERSION
GOOGLE_CREDENTIALS_FILE = 'input/commonhelpua-a4e2da7af440.json'
hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)
cam_img_path = 'images/source/'
#cam_img_path = 'E:\\ai\\object detection\\sugarInclusionsClassifier\\images\\source\\'
period_rotation = PERIOD_ROTATION
csv_rotation = CSV_ROTATION

path_for_detections = 'images/for_detection/'
path_csv = 'output/csv/'
token = TELEGRAM_TOKEN
channel_id = TELEGRAM_CHANNEL_ID
channel_id_kovnir = TELEGRAM_CHANNEL_ID_KOVNIR
channel_id_jsz = TELEGRAM_CHANNEL_ID_JSZ

# "362669295" kovnir
# "406058615" maslianchuk
showResults = SHOW_RESULTS
saveResults = SAVE_RESULTS

if prod_release:
    saveResults_ForMarkings = False  # В production не зберігаємо для розмітки
    saveDetectedImages = SAVE_DETECTED_IMAGES
else:
    saveResults_ForMarkings = SAVE_RESULTS_FOR_MARKINGS
    saveDetectedImages = SAVE_DETECTED_IMAGES
img_paths = 'images/Start/test'  # or file, Path, PIL, OpenCV, numpy, list
output_path = 'output/marked'
detect_path = 'runs/detect'
jpg_list = []
#labels_ru = {'red_trash': 'кальцинированный мусор', 'shugar_trash': 'кристалы сахара', 'metal_trash': 'метал',
#             'hiden_trash': 'не определено', 'partlyhiden_trash': 'частично спрятан', 'rag_trash': 'тряпка',
#             'rope_trash': 'нитка'}
if model_version == 5:
    #labels_ru = {'arm': 'рука', 'trash': 'мусор'}
    labels_ru = {'trash': 'мусор', 'not_trash': 'не мусор', 'arm': 'рука'}
elif model_version == 8:
    labels_ru = {'nonTrash': 'не мусор', 'trash': 'мусор'}
else:
    labels_ru = {'arm': 'рука', 'trash': 'мусор'}

FONT_SCALE = 1.5
FONT_COLOR = (255, 255, 255)
FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
FONT_THICKNESS = 5
TEXT_START_POSITION = (10, 50)
LINE_COLOR = (0, 0, 0)
LINE_THICKNESS = 2



handler = RotatingFileHandler(f"{__name__}.log", mode='w', maxBytes=10000000, backupCount=10)
formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s  - %(funcName)s: %(lineno)d -%(message)s")

# add formatter to the handler
handler.setFormatter(formatter)
# add handler to the logger
logger.addHandler(handler)
#logging.handlers.RotatingFileHandler(f"{__name__}.log", 2000, 20)
logger.info(f"Testing the custom logger for module {__name__}...")


# Model
yolov5_model_path = f'{os.getcwd()}/images/shem_yolov5/sugar_tresh_3class.pt'
yolov8_model_path = f'{os.getcwd()}/images/shem_yolov8/sugar_tresh_2class.pt'
yolov9_model_path = f'{os.getcwd()}/images/shem_yolov9/sugar_tresh_2class.pt'



path = os.getcwd()
# Вывести путь к репозиторию
print(path)

if model_version == 5:
    model = torch.hub.load(yolov5_model_path, 'custom', path=yolov5_model_path, source='local')
elif model_version == 8:
    model = YOLO(yolov8_model_path)
else:
    model = YOLO(yolov9_model_path)


type_cam = 'dahua'
#type_cam = hikvision
# Настройка камер



if type_cam == 'hikvision':
    # cam = hikvisionapi.Client('http://192.168.1.68', 'admin', 'AstartA_$ec')
    if prod_release:
        cam = hikvisionapi.Client('http://10.112.0.143:10084', 'admin', 'Admin12345')
    else:
        cam = hikvisionapi.Client('http://10.20.60.99', 'admin', 'Admin12345')
    # cam = hikvisionapi.Client('http://10.20.60.194', 'admin', 'AstartA_$ec')
    # cam = hikvisionapi.Client('http://10.112.0.143:10084', 'admin', 'AstartA_$ec')

    response = cam.System.deviceInfo(method='get')
    logger.info(f"Camera: {response}")
else:
    if prod_release:
        camera_ip = "10.112.0.143:10084"  # IP-адрес камеры
    else:
        camera_ip = "10.20.60.99"  # IP-адрес камеры
    username = "admin"  # Имя пользователя
    password = "Admin12345"  # Пароль
    api_endpoint = "/cgi-bin/snapshot.cgi"

    # Формирование URL-адреса для запроса
    url = f"http://{camera_ip}{api_endpoint}"

    # Создание HTTP-заголовков для аутентификации
    auth = requests.auth.HTTPDigestAuth(username, password)

    # Выполнение GET запроса для получения снимка
    try:
        response = requests.get(url, auth=auth)
        # Проверка статуса ответа
        if response.status_code == 200:
            # Сохранение полученного изображения
            with open("snapshot.jpg", "wb") as f:
                f.write(response.content)
            print("Снимок успешно сохранен")
        else:
            print(f"Ошибка: {response.status_code}")
            logger.error("Camera: Ошибка первичного ответа камеры: %s " % (response.status_code))

    except OSError as e:
        logger.error("Camera: Ошибка первичного запроса get в камеру: %s" % (e.strerror))

'''
cur_Main_Thread = Main_Thread(cam_img_path, period_rotation)
cur_Main_Thread.start()
cur_SendResults_Thread = SendResults_Thread(path_csv, csv_rotation)
cur_SendResults_Thread.start()

Service_Treads_list=[]
num_threads = 0

while True:
    if not cur_Main_Thread.is_alive():
        cur_Main_Thread = Main_Thread(cam_img_path,period_rotation)
        cur_Main_Thread.start()
        print('Пересоздали поток камеры')
        logger.error("main_module: Пересоздали поток камеры" )
    if not cur_SendResults_Thread.is_alive():
        cur_SendResults_Thread = SendResults_Thread(path_csv, csv_rotation)
        cur_SendResults_Thread.start()
        print('Пересоздали поток отчетов')
        logger.error("main_module: Пересоздали поток отчетов" )

    file_list = os.listdir(cam_img_path)
    if len(file_list)>20:
        for current_treads in Service_Treads_list:
            if not current_treads.is_alive():
                Service_Treads_list.remove(current_treads)
        num_active_service_treads = len(Service_Treads_list)
        num_needs_service_treads = 3 - len(Service_Treads_list)
        logger.info('main_module: количество активных сервисных потоков:: %s' % (num_active_service_treads))
        print('main_module: количество активных сервисных потоков:: %s' % (num_active_service_treads))
        index = num_needs_service_treads
        while index > 0:
            cur_Service_Thread = Service_Thread(cam_img_path, path_for_detections, num_threads)
            cur_Service_Thread.start()
            Service_Treads_list.append(cur_Service_Thread)
            num_threads = num_threads + 1
            index = index - 1
    if len(file_list) > 100:
        send_text_telegram(token, channel_id, 'Большое количество файлов на определение:: %s Количество активных потоков:: %s ::: %s  ' %(len(file_list),num_active_service_treads,num_threads))
        #print('Большое количество файлов на определение:: %s Количество активных потоков:: %s ::: %s  ' % (len(file_list), num_active_service_treads, num_threads))
        logger.info('main_module: Большое количество файлов на определение:: %s Количество активных потоков:: %s ::: %s  ' % (len(file_list), num_active_service_treads, num_threads))

    time.sleep(20)
'''

image_queue = Queue()

# Function to check camera availability
def is_camera_available(camera_ip, timeout=5):
    try:
        response = requests.get(f"http://{camera_ip}/status", timeout=timeout)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

# Image capture function
def capture_image(camera_key, cam_img_path, camera_config, timeout=10):
    cam_img_path = os.path.join(cam_img_path, camera_key+'_screen_' + datetime.now().strftime('%Y%m%d_%H_%M_%S') + '.jpg')
    logger.info(f"[{camera_key}] Starting image capture from {camera_config['camera_ip']} (type: {camera_config['type_cam']})")
    print(f"[{camera_key}] Attempting to capture from {camera_config['camera_ip']}")
    try:
        if camera_config['type_cam'] == 'hikvision':
            # Використовуємо бібліотеку hikvisionapi
            try:
                cam = hikvisionapi.Client(f"http://{camera_config['camera_ip']}", camera_config['username'], camera_config['password'])
                # Отримуємо знімок з камери
                response = cam.Streaming.channels[1].picture(method='get', type='opaque_data')
                
                with open(cam_img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)
                            
                logger.info(f"[{camera_key}] [OK] Image saved using hikvisionapi: {cam_img_path}")
                print(f"[{camera_key}] [OK] Successfully captured via hikvisionapi")
                return cam_img_path
            except Exception as e:
                logger.error(f"[{camera_key}] [ERROR] hikvisionapi error: {e}")
                print(f"[{camera_key}] [ERROR] hikvisionapi failed: {e}")
                
                # Спробуємо простий RTSP
                try:
                    rtsp_url = f"rtsp://{camera_config['username']}:{camera_config['password']}@{camera_config['camera_ip']}:554/h264/ch1/main/av_stream"
                    logger.info(f"[{camera_key}] Trying RTSP fallback: rtsp://{camera_config['camera_ip']}:554/...")
                    print(f"[{camera_key}] Trying RTSP stream...")
                    
                    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                    # Встановлюємо timeout для відео потоку (в мілісекундах)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)  # 5 секунд
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)  # 5 секунд
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    if cap.isOpened():
                        # Пропускаємо кілька кадрів
                        for i in range(5):
                            cap.grab()
                        
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret and frame is not None:
                            cv2.imwrite(cam_img_path, frame)
                            logger.info(f"[{camera_key}] [OK] Image saved from RTSP: {cam_img_path}")
                            print(f"[{camera_key}] [OK] Successfully captured via RTSP")
                            return cam_img_path
                        else:
                            logger.warning(f"[{camera_key}] RTSP capture failed: ret={ret}, frame={'None' if frame is None else 'OK'}")
                except Exception as e:
                    logger.error(f"[{camera_key}] [ERROR] RTSP error: {e}")
                    print(f"[{camera_key}] [ERROR] RTSP failed: {e}")
                    
            return None
        else:  # dahua
            url = f"http://{camera_config['camera_ip']}/cgi-bin/snapshot.cgi"
            logger.info(f"[{camera_key}] Trying Dahua HTTP snapshot: {url}")
            print(f"[{camera_key}] Trying Dahua HTTP snapshot...")
            auth = requests.auth.HTTPDigestAuth(camera_config['username'], camera_config['password'])
            response = requests.get(url, auth=auth, timeout=timeout)

        if response and response.status_code == 200:
            with open(cam_img_path, 'wb') as f:
                f.write(response.content)
            logger.info(f"[{camera_key}] [OK] Image saved: {cam_img_path}")
            print(f"[{camera_key}] [OK] Successfully captured")
            return cam_img_path
        else:
            status = response.status_code if response else "No response"
            logger.error(f"[{camera_key}] [ERROR] Failed to get image: {status}")
            print(f"[{camera_key}] [ERROR] Capture failed: {status}")
            return None
    except Exception as e:
        logger.error(f"[{camera_key}] [ERROR] Exception capturing image: {e}")
        print(f"[{camera_key}] [ERROR] Exception: {e}")
        return None

# Continuous image capture
def continuous_capture_images(camera_key, cam_img_path, camera_config, interval=2.0):
    logger.info(f"Starting continuous capture for camera {camera_key} with interval {interval}s")
    
    # Додаткова перевірка активності камери
    if not camera_config.get('is_active', True):
        logger.warning(f"Camera {camera_key} is marked as inactive, stopping capture thread")
        return
    
    # Лічильник помилок для тимчасової паузи (але не зупинки потоку)
    consecutive_errors = 0
    max_consecutive_errors = 10
    pause_after_errors = 60  # Пауза 60 секунд після багатьох помилок
    
    while True:
        try:
            img_path = capture_image(camera_key, cam_img_path, camera_config)
            if img_path:
                image_queue.put((camera_key, img_path))
                consecutive_errors = 0  # Скидаємо лічильник при успіху
                logger.info(f"[{camera_key}] [OK] Image captured and queued successfully")
            else:
                consecutive_errors += 1
                logger.warning(f"Camera {camera_key}: Failed to capture image ({consecutive_errors}/{max_consecutive_errors})")
                
                if consecutive_errors >= max_consecutive_errors:
                    logger.error(f"Camera {camera_key}: Too many consecutive errors, pausing for {pause_after_errors}s")
                    send_text_telegram(TELEGRAM_TOKEN, TELEGRAM_CHANNEL_ID, 
                                     f"[WARNING] Камера {camera_key} ({camera_config['Location']}) недоступна після {max_consecutive_errors} спроб. Пауза {pause_after_errors}с")
                    time.sleep(pause_after_errors)
                    consecutive_errors = 0  # Скидаємо лічильник після паузи
                    logger.info(f"Camera {camera_key}: Resuming capture attempts after pause")
                    continue
                    
            time.sleep(interval)
        except Exception as e:
            consecutive_errors += 1
            logger.error(f"Camera {camera_key}: Exception in capture loop: {e}")
            import traceback
            logger.error(f"Camera {camera_key}: Traceback: {traceback.format_exc()}")
            
            if consecutive_errors >= max_consecutive_errors:
                logger.error(f"Camera {camera_key}: Too many errors, pausing for {pause_after_errors}s")
                time.sleep(pause_after_errors)
                consecutive_errors = 0
                logger.info(f"Camera {camera_key}: Resuming after error pause")
                continue
                
            time.sleep(interval * 2)  # Подвійна пауза при помилці

# Image processing function
def process_images_from_queue(camera_key, path_for_detections, num_thread):
    while True:
        try:
            camera_key, img_path = image_queue.get(timeout=10)
            logger.info(f"Thread-{num_thread}: Processing image from {camera_key} -> {img_path}")

            if img_path:
                logger.info(f"Поток {num_thread}: Получено изображение {img_path}, начинаем обработку")
                process_detection(camera_key, img_path, path_for_detections, num_thread)
                image_queue.task_done()  # Сообщаем, что задача обработки завершена
        except Empty:
            logger.info(f"Поток {num_thread}: Нет новых изображений в очереди, продолжаем ожидание...")
        except Exception as e:
            logger.error(f"Поток {num_thread}: Ошибка обработки изображения: {e}")

            time.sleep(0.5)
            os.remove(img_path)
            image_queue.task_done()
        except Empty:
            logger.info(f"Thread-{num_thread}: No new images, waiting...")
        except Exception as e:
            logger.error(f"Thread-{num_thread}: Error processing image: {e}")


'''

# Функция для постоянного захвата изображений
def continuous_capture_images(cam_img_path, camera_ip, type_cam, interval=0.9, max_retries=5):
    retries = 0
    while True:
        #if not is_camera_available(camera_ip):
        #    logger.warning(f"Камера {camera_ip} недоступна. Ожидание...")
        #    time.sleep(interval)
        #    continue
        try:
            img_file_path = capture_image(cam_img_path, camera_ip, type_cam)
            if img_file_path:
                image_queue.put(img_file_path)  # Сохраняем путь к изображению в очередь
                retries = 0  # Сбрасываем счётчик при успешном захвате
            else:
                retries += 1
                logger.warning(f"Не удалось захватить изображение. Попытка {retries}/{max_retries}")
                if retries >= max_retries:
                    logger.error("Достигнуто максимальное количество попыток захвата. Переподключение камеры.")
                    retries = 0  # Сбрасываем счётчик после переподключения
        except Exception as e:
            logger.error(f"Ошибка захвата изображения: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Достигнуто максимальное количество попыток после ошибки. Переподключение камеры.")
                retries = 0
        time.sleep(interval)  # Задержка перед следующим захватом

# Функция для обработки изображений из очереди
def process_images_from_queue(path_for_detections, num_thread):
    while True:
        try:
            logger.info(f"Поток {num_thread}: Ожидание нового изображения для детектирования...")
            img_path = image_queue.get(timeout=10)  # Ожидаем изображение не более 10 секунд
            if img_path:
                logger.info(f"Поток {num_thread}: Получено изображение {img_path}, начинаем обработку")
                process_detection(img_path, path_for_detections,num_thread)
                image_queue.task_done()  # Сообщаем, что задача обработки завершена
        except Empty:
            logger.info(f"Поток {num_thread}: Нет новых изображений в очереди, продолжаем ожидание...")
        except Exception as e:
            logger.error(f"Поток {num_thread}: Ошибка обработки изображения: {e}")


# Функция для захвата изображения
def capture_image(cam_img_path, camera_ip, type_cam='dahua', timeout=5):
    cam_img_path = os.path.join(cam_img_path, 'screen' + datetime.now().strftime('%Y%m%d_%H_%M_%S_%f') + '.jpg')
    logger.info(f"Начало захвата изображения для {cam_img_path}")
    try:
        if type_cam == 'hikvision':
            # Пример получения изображения с камеры Hikvision
            response = cam.Streaming.channels[1].picture(method='get', type='opaque_data', timeout=timeout)
            with open(cam_img_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
        else:
            # Пример получения изображения с другой камеры через HTTP запрос с таймаутом
            api_endpoint = "/cgi-bin/snapshot.cgi"
            url = f"http://{camera_ip}{api_endpoint}"
            auth = requests.auth.HTTPDigestAuth(username, password)
            response = requests.get(url, auth=auth, timeout=timeout)
            if response.status_code == 200:
                with open(cam_img_path, "wb") as f:
                    f.write(response.content)
                logger.info(f"Снимок сохранён: {cam_img_path}")
            else:
                logger.error(f"Ошибка получения снимка: {response.status_code}")
                return None
    except requests.exceptions.Timeout:
        logger.error(f"Запрос к камере {camera_ip} превысил время ожидания.")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Ошибка запроса камеры {camera_ip}: {e}")
        return None
    except OSError as e:
        logger.error(f"Ошибка сохранения изображения {cam_img_path}: {e}")
        return None
    return cam_img_path

'''
# Функция для обработки изображений
def process_detection(camera_key, cam_img_file_path, path_for_detections,num_thread):
    logger.info(f"Thread-{num_thread}: Начало обработки изображения {cam_img_file_path}")
    detection_start = time.time()
    try:
        # Обработка изображения
        #new_path_file = work_with_files.sort_files_by_date(cam_img_path, path_for_detections)
        new_path_file = cam_img_file_path
        #logger.info(f"Файл перемещён в {new_path_file}")
        if not new_path_file == '':
            if model_version == 5:
                detect_trash_v5(new_path_file, num_thread, detection_start)
            elif model_version == 8:
                detect_trash_v7(new_path_file, num_thread, detection_start)
            else:
                detect_trash_v9(camera_key, new_path_file, num_thread, 0.5)
        os.remove(new_path_file)
        end_time = time.time()
        logger.info(f"Thread-{num_thread}: Обработка завершена за {end_time - detection_start:.2f} секунд")
        logger.info(f"Файл удалён: {new_path_file}")
    except OSError as e:
        logger.error(f"Ошибка обработки файла {cam_img_path}: {e}")

def send_csv_report(camera_key,path_csv):
    # Пример отправки CSV отчёта
    file_name = f"{camera_key}_{datetime.now().strftime('%Y_%m_%d')}.csv"
    _path_csv = os.path.join(path_csv, file_name)
    logger.info(f"Отправка CSV отчёта {file_name}")
    if os.path.exists(_path_csv):
        send_document_telegram(token, channel_id, _path_csv, file_name)
        logger.info(f"Отчёт {file_name} отправлен")
    else:
        logger.error(f"Файл отчёта {file_name} отсутствует")

# Функция для периодической отправки CSV отчётов
def continuous_csv_reports(camera_key, interval=1800):
    while True:
        send_csv_report(camera_key, path_csv)
        time.sleep(interval)



if __name__ == '__main__':
    logger.info("="*80)
    logger.info("STARTING APPLICATION")
    logger.info("="*80)
    
    num_detection_threads = NUM_DETECTION_THREADS

    # Підраховуємо кількість активних камер для max_workers
    active_camera_count = len([c for c in cameras.values() if c.get('is_active', True)])
    total_workers = active_camera_count + num_detection_threads
    
    logger.info(f"Configuration:")
    logger.info(f"  - Active cameras: {active_camera_count}")
    logger.info(f"  - Detection threads: {num_detection_threads}")
    logger.info(f"  - Total workers: {total_workers}")
    logger.info(f"Starting ThreadPoolExecutor with {total_workers} workers ({active_camera_count} cameras + {num_detection_threads} detection threads)")
    print(f"\n{'='*80}")
    print(f"Starting with {active_camera_count} cameras and {num_detection_threads} detection threads")
    print(f"{'='*80}\n")
    
    with ThreadPoolExecutor(max_workers=total_workers) as executor:
        # Start image capture for each camera
        for camera_key, camera_config in cameras.items():
            if camera_config.get('is_active', True):  # За замовчуванням True для зворотної сумісності
                logger.info(f"▶ Starting capture thread for camera {camera_key} ({camera_config['Location']}) at {camera_config['camera_ip']}")
                print(f"▶ Starting {camera_key} ({camera_config['Location']}) - {camera_config['camera_ip']}")
                executor.submit(continuous_capture_images, camera_key, cam_img_path, camera_config)
            else:
                logger.info(f"⏸ Skipping inactive camera {camera_key} at {camera_config['Location']}")
                print(f"⏸ Skipping {camera_key} (inactive)")

        # Start image processing threads
        for i in range(num_detection_threads):
            executor.submit(process_images_from_queue, camera_key, path_for_detections, i + 1)

        # Запуск периодической отправки отчётов каждые 1800 секунд (30 минут)
        for camera_key in cameras.items():
            executor.submit(continuous_csv_reports, camera_key, 1800)

        logger.info("Все задачи запущены")
        # Keep the main thread alive
        while True:
            time.sleep(1)
'''            
    with ThreadPoolExecutor(max_workers=3+num_detection_threads) as executor:
        # Запуск бесконечного цикла для захвата изображений каждые 0,6 секунд
        executor.submit(continuous_capture_images, cam_img_path, camera_ip, type_cam, period_rotation)

        # Запуск нескольких потоков для параллельной обработки изображений
        logger.info(f"Запуск процессов детектирования СТАРТ")
        for i in range(num_detection_threads):
            logger.info(f"Запуск процесса детектирования {i}")
            executor.submit(process_images_from_queue, path_for_detections, i + 1)

        # Запуск периодической отправки отчётов каждые 1800 секунд (30 минут)
        executor.submit(continuous_csv_reports, 1800)

        logger.info("Все задачи запущены")

        # Бесконечное ожидание завершения задач
        while True:
            time.sleep(1)
'''