import torch
import numpy as np
# from matplotlib import pyplot as pl
# import math
import cv2
import requests

import os
import yolov5
#from yolov5.utils.plots import Annotator, colors, save_one_box
import shutil
import glob
from send_mail import init_mail
import json
# from picamera2 import Picamera2
# from IPython.display import display

import hikvisionapi
import time
from datetime import datetime

import csv
from threading import Thread
from dir_files import work_with_files
import pygsheets
import logging
from logging.handlers import RotatingFileHandler

from oauth2client.service_account import ServiceAccountCredentials

# predict_class(40)

def send_msg_telebot(token, channel_id, text: str):
    bot = telebot.TeleBot(token)
    bot.send_message(channel_id, text)

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
    print(r.json())

    if r.status_code != 200:
        raise Exception("post_text error")


def send_document_telegram(token, channel_id, path_csv: str, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendDocument"

    files = {'document': open(path_csv, 'rb')}

    r = requests.post(method, files=files, data={
        "chat_id": channel_id,
        "text": text
    })
    print(r.json())

    if r.status_code != 200:
        raise Exception("post_text error")

def send_document_telegram1(token, channel_id, path_csv: str, text: str):
    url = "https://api.telegram.org/bot"
    url += token
    method = url + "/sendDocument"

    with open(path_csv, "rb") as csvfile:
        files = {"document": csvfile}
        title = "данные.csv"
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

def detect_trash(img_path: str):
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
    # Преобразовать изображение в оттенки серого и принудительно преобразовать его в значение с плавающей точкой, которое используется при определении угла.
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img_gray = np.float32(img_gray)

    # cv2.imshow('Harris Corners', img)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(450,325), (1480, 160),
                             (2340, 90), (2640, 710),
                             (2590, 750), (2630, 840),
                             (1710, 1515), (960, 1515),
                             (890, 1360),
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
    #cv2.imshow("images/for_detection/img1.jpg", img)
    #cv2.imshow("images/for_detection/image_masked.png", masked_image)
     # Inference
    logger.info('Start detection %d %d' %(h,w))
    try:
        results = model(img)
    except OSError as e:
        logger.error("detection Ошибка работы модели: %s : %s" % (img_path, e.strerror))
        #print("detection Ошибка работы модели: %s : %s" % (img_path, e.strerror))
        exit(-1)
    labels = results.names
    # tekList = results.xyxy[0].tolist()
    # tekLen = len(results.xyxy[0].tolist())

    # tekresults = results.tolist()
    # print(tekresults)

    s = ''
    isTrash = False
    for i, (im, pred) in enumerate(zip(results.ims, results.pred)):
        s += f'изображение {i + 1}/{len(results.pred)}: {im.shape[0]}x{im.shape[1]}, '  # string
        dict = {a: 0 for a in range(7)}
        if pred.shape[0]:
            for c in pred[:, -1].unique():
                n = (pred[:, -1] == c).sum()  # detections per class
                s += f"{n} - {labels_ru[results.names[int(c)]]}{'s' * (n > 1)}, "  # add to string
                s = s.replace('s,', ',')
                dict[int(c)] = int(f"{n}")
                isTrash = True
            s = s.rstrip(', ')
            save_to_csv(path_csv,dict,file_date,file_time)
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
    cv2.imwrite(output_path + '/images_%03d_3.jpg' % (i), im)

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
    if isTrash:
        extension = img_path.split('.')[-1]
        file_name = img_path.split('/')[-1]
        cv2.imwrite("images/for_marking/" + file_name, masked_image)

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
            sended = send_group_img(token, channel_id, output_path, text)
            #sended = send_group_img(token, channel_id_kovnir, output_path, text)
        except OSError as e:
            logger.error("detection: Ошибка отправки в бот: %s : %s" % (output_path, e.strerror))
            #print("detection: Ошибка отправки в бот: %s : %s" % (output_path, e.strerror))
        addr_to = "oleksandr.maslianchuk@agrichain.ua"  # Получатель
        files = [
            '../yolov5_train_detect/runs/exp/']  # Если нужно отправить все файлы из заданной папки, нужно указать её
        # init_mail.send_email(addr_to, "sugar_trash Привет, тебе сообщение от малинки", "Первіе файлы", files)
    else:
        logger.info("detection: не нашли мусор")
        #print("detection: не нашли мусор")

def count_line_csv(filename):
    with open(filename) as f:
        return sum(1 for line in f)

def save_to_csv(path_csv: str,pred,file_date,file_time):

#    file_name = datetime.now().strftime('%Y_%m_%d')+'.csv'
    file_name = file_date.replace('-','_')+'.csv'

    path_csv = path_csv + file_name
    if os.path.exists(path_csv):
        create = 'a'
    else:
        create = 'w'
    with open(path_csv, create, newline='') as csvfile:
        fieldnames = ['date', 'time', 'red_trash','sugar_trash','metal_trash','hiden_trash','partlihiden_trash','rag_trash','rope_trash']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#        writer = csv.writer(csvfile)
        if count_line_csv(path_csv) == 0:
            writer.writeheader()
        writer.writerow({'date':file_date, 'time': file_time, 'red_trash': pred[0], 'sugar_trash': pred[1],
                         'metal_trash': pred[2], 'hiden_trash': pred[3], 'partlihiden_trash': pred[4], 'rag_trash':pred[5], 'rope_trash':pred[0]})

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
                           "low_production": htmlColorToJSON("#CCCCFF")}

    SCOPES = ('https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive')
    gc = pygsheets.authorize(service_file=GOOGLE_CREDENTIALS_FILE)

    # Open spreadsheet and then worksheet
    #sh = gc.open_by_key(docKEY)
    sh = gc.open(docTitle)
    #sh.share("oleksandr.maslianchuk@agrichain.ua")
    try:
        wks = sh.worksheet_by_title(sheetTitle)
    except:
        wks = sh.add_worksheet(sheetTitle)

    # share the sheet with your friend



class Main_Thread(Thread):
    def __init__(self, cam_img_path, period_rotation):
        Thread.__init__(self)
        self._cam_img_path = cam_img_path
        self._period_rotation = period_rotation

    def run(self):
        while True:

            cam_img_path = self._cam_img_path + 'screen'+datetime.now().strftime('%Y%m%d_%H_%M_%S_%f')+'.jpg'
            sleep_start = time.time()
            try:
                response = cam.Streaming.channels[1].picture(method='get', type='opaque_data')
                with open(cam_img_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=1024):
                        if chunk:
                            f.write(chunk)

                    f.close()
            except OSError as e:
                с.error("Camera: Ошибка запроса get в камеру")

            time.sleep(self._period_rotation)
            sleep_end = time.time()
            diff_time_request = sleep_end - sleep_start
            if os.path.exists(cam_img_path):
                logger.info('Camera: записали файл: %s' %(cam_img_path))
            logger.info('Camera: плановый период:: %s затраченное время:: %s разница:: %s' % (
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
            new_path_file = ''
            try:
                new_path_file = work_with_files.sort_files_by_date(self._cam_img_path, self._path_for_detections)
            except OSError as e:
                i=0
                #print("detection prepare %s: не нашли самый старый файл: %s" % (self._num_thread, e.strerror))
            if not new_path_file == '':
                detect_trash(new_path_file)
                detection_end = time.time()
                diff_time_detect = detection_end-detection_start
                logger.info('detection %s: затраченное время на распознавание:: %s' %(self._num_thread, diff_time_detect) )
                #print('detection %s: затраченное время на распознавание:: %s' %(self._num_thread, diff_time_detect) )

                try:
                    os.remove(new_path_file)
                except OSError as e:
                    logger.error("detection %s: Ошибка удаления файла: %s : %s" % (self._num_thread, new_path_file, e.strerror))
                    #print("detection %s: Ошибкаудаления файла: %s : %s" % (self._num_thread, new_path_file, e.strerror))
                if os.path.exists(f'{detect_path}'):
                    try:
                        shutil.rmtree(detect_path)
                    except OSError as e:
                        logger.error("detection %s: Ошибка удаления директории: %s : %s" % (self._num_thread, detect_path, e.strerror))
                        #print("detection %s: Ошибка удаления директории: %s : %s" % (self._num_thread, detect_path, e.strerror))

class SendResults_Thread(Thread):
    def __init__(self, path_csv, csv_rotation):
        Thread.__init__(self)
        self._path_csv = path_csv
        self._csv_rotation = csv_rotation

    def run(self):
        while True:

            detection_start = time.time()
            file_name = datetime.now().strftime('%Y_%m_%d') + '.csv'
            _path_csv = self._path_csv + file_name
            if os.path.exists(_path_csv):
                send_document_telegram(token, channel_id, _path_csv, file_name)
                #send_photo_telegram(token, channel_id, self._path_csv, file_name)

            else:
                send_text_telegram(token, channel_id,'за текущую дату отсутствует файл с данными')
            time.sleep(self._csv_rotation)

GOOGLE_CREDENTIALS_FILE = 'input/commonhelpua-a4e2da7af440.json'
hsv_min = np.array((2, 28, 65), np.uint8)
hsv_max = np.array((26, 238, 255), np.uint8)
cam_img_path = 'images/source/'
#cam_img_path = 'E:\\ai\\object detection\\sugarInclusionsClassifier\\images\\source\\'
period_rotation = 0.8
csv_rotation = 3600

path_for_detections = 'images/for_detection/'
path_csv = 'output/csv/'
token = "5694476991:AAFBBqkT_b1I8cwCdoxL-Q8mdN6kXNl6BWU"
channel_id = "406058615"
channel_id_kovnir = '362669295'
# "362669295"
# "406058615"
showResults = False
saveResults = True
img_paths = 'images/Start/test'  # or file, Path, PIL, OpenCV, numpy, list
output_path = 'runs/detect/exp'
detect_path = 'runs/detect'
jpg_list = []
labels_ru = {'red_trash': 'кальцинированный мусор', 'shugar_trash': 'кристалы сахара', 'metal_trash': 'метал',
             'hiden_trash': 'не определено', 'partlyhiden_trash': 'частично спрятан', 'rag_trash': 'тряпка',
             'rope_trash': 'нитка'}
FONT_SCALE = 1.5
FONT_COLOR = (255, 255, 255)
FONT_FACE = cv2.FONT_HERSHEY_COMPLEX
FONT_THICKNESS = 5
TEXT_START_POSITION = (10, 50)
LINE_COLOR = (0, 0, 0)
LINE_THICKNESS = 2
#cam = Client('https://10.20.7.48', 'admin', 'AstartA_$ec')
#cam = Client('http://193.32.68.58', 'admin', 'AstartA_$ec')
#cam = hikvisionapi.Client('http://169.254.16.38', 'admin', 'AstartA_$ec')
#cam = hikvisionapi.Client('http://169.254.16.39', 'admin', 'AstartA_$ec')
#cam = hikvisionapi.Client('http://169.254.65.204', 'admin', 'AstartA_$ec')



logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


# handler = logging.FileHandler(f"{__name__}.log", mode='w')
handler = RotatingFileHandler(f"{__name__}.log", mode='w', maxBytes=20000000, backupCount=20)
formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s  - %(funcName)s: %(lineno)d -%(message)s")

# add formatter to the handler
handler.setFormatter(formatter)
# add handler to the logger
logger.addHandler(handler)
#logging.handlers.RotatingFileHandler(f"{__name__}.log", 2000, 20)
logger.info(f"Testing the custom logger for module {__name__}...")

cam = hikvisionapi.Client('http://10.20.60.194', 'admin', 'AstartA_$ec')


response = cam.System.deviceInfo(method='get')
logger.info(f"Camera: {response}")
#print(response)

#while True:
#    try:
#        response = cam.Event.notification.alertStream(method='get', type='stream')
#        if response:
#            print(response)
#    except Exception:
#        pass

# Model
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # or yolov5n - yolov5x6, custom
# model = torch.hub.load('../yolov5', 'custom', path='../inclusions/dataset_yolo5/best.pt', source='local')
#yolov5_path = 'E:\\ai\\sugarInclusionsClassifier\\sugarInclusionsClassifier\\yolov5'
yolov5_path = f'{os.getcwd()}/yolov5_train_detect'
#yolov5_model_path = 'E:\\ai\\sugarInclusionsClassifier\\sugarInclusionsClassifier\\images\\shem_yolov5\\best.pt'
yolov5_model_path = f'{os.getcwd()}/images/shem_yolov5/best.pt'

model = torch.hub.load(yolov5_path, 'custom', path=yolov5_model_path, source='local')

Main_Thread(cam_img_path,period_rotation).start()
Service_Thread(cam_img_path,path_for_detections,1).start()
Service_Thread(cam_img_path,path_for_detections,2).start()
Service_Thread(cam_img_path,path_for_detections,3).start()
SendResults_Thread(path_csv,csv_rotation).start()


# detect.py --source images/Start/test --weights inclusions/dataset_yolo5/best.pt --imgsz 640 640 --view_img true --save_txt true --save_conf true --save_crop true --visualize true

# python yolov5/detect.py --source images/Start/test --weights inclusions/dataset_yolo5/best.pt
# Images
# img_path = '../images/Start/Red/IMG20221013133136.jpg'  # or file, Path, PIL, OpenCV, numpy, list


#for img_path in glob.glob(img_paths + '/*.jpg'):
#    detect_trash(img_path)

#    try:
#        shutil.rmtree(detect_path)
#    except OSError as e:
#        print("Ошибка: %s : %s" % (detect_path, e.strerror))

    # print(results.xyxy[0])  # img1 predictions (tensor)
    # print(results.pandas().xywh[0])  # img1 predictions (pandas)
    # print(results.pandas().xywhn[0])  # img1 predictions (pandas)
    # print(results.pandas().xyxy[0])  # img1 predictions (pandas)
    # print(results.pandas().xyxyn[0])  # img1 predictions (pandas)
# print(crops.box)
# print(crops.conf)
# print(crops.cls)
# print(crops.im)
# print( crops[0])



# !python detect.py --weights {runs_directory}/weights/best.pt\
#                  --source ../inclusions/test/images\
#                  --conf 0.1\
#                  --data ../inclusions/data.yaml\
#                  --project {PROJECT}/{RUN_NAME}\
#                  --name test_data_detections

# python yolo5/detect.py --weights ../inclusions/dataset_yolo5/best.pt\  --source ../inclusions/dataset_yolo5/test/images\ --conf 0.1  --data inclusions/data.yaml\ --project inclusions/test/dataset_yolo5/detected\ --name trash+now()
