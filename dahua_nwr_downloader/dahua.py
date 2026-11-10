import requests

# Замените следующие значения на свои
camera_ip = "192.168.1.108"  # IP-адрес камеры
username = "admin"           # Имя пользователя
password = "AstartA_$ec"   # Пароль
api_endpoint = "/cgi-bin/snapshot.cgi"  # Пример эндпоинта API для снимка

# Формирование URL-адреса для запроса
url = f"http://{camera_ip}{api_endpoint}"

# Создание HTTP-заголовков для аутентификации
auth = requests.auth.HTTPDigestAuth(username, password)

# Выполнение GET запроса для получения снимка
response = requests.get(url, auth=auth)

# Проверка статуса ответа
if response.status_code == 200:
    # Сохранение полученного изображения
    with open("snapshot.jpg", "wb") as f:
        f.write(response.content)
    print("Снимок успешно сохранен")
else:
    print(f"Ошибка: {response.status_code}")



# Пример эндпоинта API для получения статуса
api_endpoint = "/cgi-bin/status.cgi"

# Формирование URL-адреса для запроса
url = f"http://{camera_ip}{api_endpoint}"

# Создание HTTP-заголовков для аутентификации
auth = requests.auth.HTTPDigestAuth(username, password)

# Выполнение GET запроса для получения статуса
response = requests.get(url, auth=auth)

# Проверка статуса ответа
if response.status_code == 200:
    status_data = response.json()  # Предполагается, что ответ в формате JSON
    print("Статус камеры:")
    print(status_data)
else:
    print(f"Ошибка: {response.status_code}")



# Пример эндпоинта API для настройки интервала сохранения
# api_endpoint = "/cgi-bin/configManager.cgi?action=setConfig&Encode[0].SnapFormat[0].Video.FPS=0.00166666667"


# Формирование URL-адреса для запроса
# url = f"http://{camera_ip}{api_endpoint}"

# Создание HTTP-заголовков для аутентификации
# auth = requests.auth.HTTPDigestAuth(username, password)

# Выполнение GET запроса для настройки
# response = requests.get(url, auth=auth)

# Проверка статуса ответа
# if response.status_code == 200:
#     print("Настройка успешно применена")
# else:
#     print(f"Ошибка: {response.status_code}")




# Пример эндпоинта API для получения списка файлов на SD-карте
# api_endpoint = "/cgi-bin/sdList.cgi"

# Формирование URL-адреса для запроса
# url = f"http://{camera_ip}{api_endpoint}"

# Создание HTTP-заголовков для аутентификации
# auth = requests.auth.HTTPDigestAuth(username, password)

# Выполнение GET запроса для получения списка файлов
# response = requests.get(url, auth=auth)

# Проверка статуса ответа
# if response.status_code == 200:
    #     file_list = response.json()  # Предполагается, что ответ в формате JSON
    #     for file_info in file_list:
    #        file_path = file_info["path"]
        # Пример URL для скачивания файла
    #        file_url = f"http://{camera_ip}/{file_path}"

        # Выполнение GET запроса для скачивания файла
    #         file_response = requests.get(file_url, auth=auth)

    #        if file_response.status_code == 200:
            # Сохранение файла
    #             with open(file_info["name"], "wb") as f:
    #                 f.write(file_response.content)
    #            print(f"Файл '{file_info['name']}' успешно скачан")
    #        else:
    #         print(f"Ошибка при скачивании файла '{file_info['name']}': {file_response.status_code}")
    #else:
#    print(f"Ошибка: {response.status_code}")