'''
def get_images_from_server():

    remote_dir = '/mnt/sd/snapshot/'
    if prod_release:
        hostname = 'http://10.112.0.143'
        port = '10084'
        username = 'admin'
        password = 'Admin12345'

        #cam = hikvisionapi.Client('http://10.112.0.143:10084', 'admin', 'Admin12345')
    else:
        hostname = 'http://10.20.60.99'
        port = '80'
        username = 'admin'
        password = 'Admin12345'

        #cam = hikvisionapi.Client('http://10.20.60.99', 'admin', 'Admin12345')

    """Функция для получения списка и загрузки новых изображений с сервера"""
    try:
        transport = paramiko.Transport((hostname, port))
        transport.connect(username=username, password=password)
        sftp = paramiko.SFTPClient.from_transport(transport)

        files = sftp.listdir(remote_dir)
        new_files = [f for f in files if f.endswith('.jpg')]  # Фильтруем только изображения
        for file in new_files:
            remote_file = os.path.join(remote_dir, file)
            local_file = os.path.join(cam_img_path, file)
            if not os.path.exists(local_file):
                sftp.get(remote_file, local_file)  # Загружаем файл
                logger.info(f"Загружено изображение: {local_file}")
                process_image(local_file)  # Обрабатываем изображение
            else:
                logger.info(f"Изображение уже загружено: {local_file}")

        sftp.close()
        transport.close()
    except Exception as e:
        logger.error(f"Ошибка при работе с сервером: {str(e)}")



def detect_trash_v9(img_path: str, num_thread, threshold=0.5):
    if os.path.isfile(img_path):
        file_name = os.path.basename(img_path).split('.')[0]
        file_date_time = file_name[6:]
        file_date = f"{file_date_time[:4]}-{file_date_time[4:6]}-{file_date_time[6:8]}"
        file_time = file_date_time[9:].replace("_", ":", 2).replace("_", ".")

    try:
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
    except OSError as e:
        logger.error(
            f"srvs_threads {num_thread}: detect_trash_v9: Ошибка чтения файла изображения: {img_path} : {e.strerror}")
        return

    # Создание маски и применение ее к изображению
    mask = np.zeros(img.shape, dtype=np.uint8)
    roi_corners = np.array([[(585, 995), (645, 1515),
                             (1210, 1351), (1615, 1120),
                             (1380, 770)]],
                           dtype=np.int32)
    ignore_mask_color = (255,) * img.shape[2]
    cv2.fillPoly(mask, roi_corners, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    #cv2.imwrite(img_path, masked_image)
    try:
        cv2.imwrite(img_path, masked_image)
    except Exception as e:
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка сохранения изображения: {img_path} : {str(e)}")
        return

    if os.path.isfile(img_path):
        try:
            img = cv2.imread(img_path)
        except OSError as e:
            logger.error(
                f"srvs_threads {num_thread}: detect_trash_v9: Ошибка чтения файла изображения: {img_path} : {e.strerror}")
            return
    else:
        return

    logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Start detection {h} {w}')

    try:
        print(f"srvs_threads {num_thread}: detect_trash_v9: Начало обработки изображения")
        masks, boxes, labels = get_outputs_v9(img, model, num_thread, threshold=threshold)
        print(f'srvs_threads {num_thread}: detect_trash_v9: Результаты модели: {masks}, {boxes}, {labels}')

        if not boxes and not labels:
            logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Модель не нашла ни одного объекта")
            return
        print(f'srvs_threads {num_thread}: detect_trash_v9: {masks} {boxes} {labels}')
    except Exception as e:
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка работы модели: {img_path} : {str(e)}")
        return



    isTrash = False
    s = f'зображення {1}: {w}x{h}, '

    im = img.copy()
    for box, label in zip(boxes, labels):
        print(f'{label} {box}')
        if label == "trash":  # Убедитесь, что метка мусора совпадает
            isTrash = True
            cv2.rectangle(im, box[0], box[1], (255, 0, 0), 6)
        elif label == "arm":  # Також позначаємо якщо знайдено руку
            cv2.rectangle(im, box[0], box[1], (0, 255, 0), 4)  # Зелений прямокутник для руки

    n_trash = labels.count("trash")
    n_arm = labels.count("arm")
    dict = {0: int(n_trash), 1: int(0)}
    #s += f"домішки: {n_trash}  "
    s += f"домішки: {n_trash}"
    #if n_arm > 0:
    #    s += f",\nрука: {n_arm}"

    if isTrash:
        save_to_csv(camera_key, path_csv, dict, file_date, file_time)
        if use_db_connection:
            save_to_db(dict, camera_key, f'{file_date} {file_time}', num_thread)

    # Добавляем текст на изображение
    text = s.replace(', ', '\n')
    img = draw_text_line(im, (5, 20), text, FONT_SCALE, FONT_FACE, FONT_THICKNESS, LINE_THICKNESS, LINE_COLOR, "custom")
    i = 0
    cv2.imwrite(f'{output_path}/images_{i:03d}.jpg', img)

    if isTrash:
        im_pil = Image.fromarray(im[..., ::-1])  # Конвертируем BGR в RGB для PIL
        im_pil.show()
    
    logger.info(f"DEBUG BEFORE SAVE: Reached for_marking save check")
    
    # Зберігаємо for_marking якщо знайдено trash або arm
    logger.info(f"DEBUG: saveResults_ForMarkings={saveResults_ForMarkings}, isTrash={isTrash}, n_arm={n_arm}, n_trash={n_trash}")
    if saveResults_ForMarkings and (isTrash or n_arm > 0):
        try:
            save_path = f"images/for_marking/{file_name}.jpg"
            cv2.imwrite(save_path, masked_image)
            logger.info(f"Saved for_marking: {file_name}.jpg (trash: {n_trash}, arm: {n_arm})")
            if os.path.exists(save_path):
                logger.info(f"File successfully saved to {save_path}")
            else:
                logger.error(f"Failed to save file to {save_path}")
        except Exception as e:
            logger.error(f"Error saving for_marking: {e}")
    else:
        logger.info(f"NOT saving for_marking: saveResults_ForMarkings={saveResults_ForMarkings}, condition={(isTrash or n_arm > 0)}")

    #if saveResults:
    #    logger.info(f'srvs_threads {num_thread}: detect_trash_v9: Save analyzed results')
    #    try:
    #        results.save(True)
    #    except OSError as e:
    #        logger.error(
    #            f"srvs_threads {num_thread}: detect_trash_v9: Ошибка сохранения результатов модели: {img_path} : {e.strerror}")

    try:
        if isTrash:
            sended = send_photo_telegram(token, channel_id, f'{output_path}/images_{i:03d}.jpg', text)
            #if prod_release:
            sended = send_photo_telegram(token, channel_id_kovnir, f'{output_path}/images_{i:03d}.jpg', text)

            print(f'Отправляем фото {output_path} :: {sended}')
    except OSError as e:
        logger.error(f"srvs_threads {num_thread}: detect_trash_v9: Ошибка отправки в бот: {output_path} : {e.strerror}")

    if not isTrash:
        logger.info(f"srvs_threads {num_thread}: detect_trash_v9: не нашли мусор")
'''