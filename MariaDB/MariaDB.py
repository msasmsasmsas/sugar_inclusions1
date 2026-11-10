#import mysql.connector
#from mysql.connector import Error

import pymysql

def create_connection(host_name, port_name, user_name, user_password, db_name=''):

    connection = None

    try:

        connection = pymysql.connect(
        host = host_name,
        port = port_name,
        user = user_name,
        passwd = user_password,
#        if db_name!='':
        database = db_name
        )
        print("Подключение к базе данных MySQL прошло успешно")
    except Exception  as e:

        print(f"Произошла ошибка '{e}'")

    return connection

    connection = create_connection("localhost", "root", "")

def create_database(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("База данных создана успешно")
    except Exception  as e:
        print(f"Произошла ошибка '{e}'")

def create_table(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Таблица создана успешно")
    except Exception  as e:
        print(f"Произошла ошибка '{e}'")

def insert_table(connection, query):
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("данных создана успешно")
    except Exception  as e:
        print(f"Произошла ошибка '{e}'")