# Подключаем библиотеки
import httplib2
import apiclient.discovery
from oauth2client.service_account import ServiceAccountCredentials


CREDENTIALS_FILE = 'test-proj-for-habr-article-1ab131d98a6b.json'  # имя файла с закрытым ключом

credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, ['https://www.googleapis.com/auth/spreadsheets',
                                                                                  'https://www.googleapis.com/auth/drive'])
httpAuth = credentials.authorize(httplib2.Http())
service = apiclient.discovery.build('sheets', 'v4', http = httpAuth)

CREDENTIALS_FILE = 'seraphic-effect-248407-7ac2c44ec709.json'  # Имя файла с закрытым ключом, вы должны подставить свое

# Читаем ключи из файла
credentials = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive'])

httpAuth = credentials.authorize(httplib2.Http()) # Авторизуемся в системе
service = apiclient.discovery.build('sheets', 'v4', http = httpAuth) # Выбираем работу с таблицами и 4 версию API

spreadsheet = service.spreadsheets().create(body = {
    'properties': {'title': 'Первый тестовый документ', 'locale': 'ru_RU'},
    'sheets': [{'properties': {'sheetType': 'GRID',
                               'sheetId': 0,
                               'title': 'Лист номер один',
                               'gridProperties': {'rowCount': 100, 'columnCount': 15}}}]
}).execute()
spreadsheetId = spreadsheet['spreadsheetId'] # сохраняем идентификатор файла
print('https://docs.google.com/spreadsheets/d/' + spreadsheetId)

driveService = apiclient.discovery.build('drive', 'v3', http = httpAuth)
shareRes = driveService.permissions().create(
    fileId = spreadsheet['spreadsheetId'],
    body = {'type': 'anyone', 'role': 'reader'},  # доступ на чтение кому угодно
#{'type': 'user', 'role': 'writer', 'emailAddress': 'user@example.com'}
    fields = 'id'
).execute()

results = service.spreadsheets().batchUpdate(spreadsheetId = spreadsheet['spreadsheetId'], body = {
  "requests": [

    # Задать ширину столбца A: 317 пикселей
    {
      "updateDimensionProperties": {
        "range": {
          "sheetId": 0,
          "dimension": "COLUMNS",  # COLUMNS - потому что столбец
          "startIndex": 0,         # Столбцы нумеруются с нуля
          "endIndex": 1            # startIndex берётся включительно, endIndex - НЕ включительно,
                                   # т.е. размер будет применён к столбцам в диапазоне [0,1), т.е. только к столбцу A
        },
        "properties": {
          "pixelSize": 317     # размер в пикселях
        },
        "fields": "pixelSize"  # нужно задать только pixelSize и не трогать другие параметры столбца
      }
    },

    # Задать ширину столбца B: 200 пикселей
    {
      "updateDimensionProperties": {
        "range": {
          "sheetId": 0,
          "dimension": "COLUMNS",
          "startIndex": 1,
          "endIndex": 2
        },
        "properties": {
          "pixelSize": 200
        },
        "fields": "pixelSize"
      }
    },

    # Задать ширину столбцов C и D: 165 пикселей
    {
      "updateDimensionProperties": {
        "range": {
          "sheetId": 0,
          "dimension": "COLUMNS",
          "startIndex": 2,
          "endIndex": 4
        },
        "properties": {
          "pixelSize": 165
        },
        "fields": "pixelSize"
      }
    },

    # Задать ширину столбца E: 100 пикселей
    {
      "updateDimensionProperties": {
        "range": {
          "sheetId": 0,
          "dimension": "COLUMNS",
          "startIndex": 4,
          "endIndex": 5
        },
        "properties": {
          "pixelSize": 100
        },
        "fields": "pixelSize"
      }
    }
  ]
}).execute()