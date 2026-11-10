"""
Централізований файл конфігурації змінних оточення
"""
import os
import logging

# Спробуємо завантажити .env файл якщо є
try:
    from dotenv import load_dotenv
    load_dotenv()  # Завантажує .env файл якщо він існує
except ImportError:
    pass  # python-dotenv не встановлено

logger = logging.getLogger(__name__)

# Рівень оточення (dev, prod, localhost)
ENV_LEVEL = os.getenv('ENV_LEVEL', 'dev')

# Використання бази даних
USE_DB = os.getenv('USE_DB', 'false').lower() == 'true'

# Версія моделі YOLO (5, 8, 9)
MODEL_VERSION = int(os.getenv('MODEL_VERSION', '9').replace('v', ''))

# Збереження результатів
SAVE_RESULTS_FOR_MARKINGS = os.getenv('SAVE_RESULTS_FOR_MARKINGS', 'true').lower() == 'true'
SAVE_DETECTED_IMAGES = os.getenv('SAVE_DETECTED_IMAGES', 'true').lower() == 'true'
SAVE_MASKED_IMAGES = os.getenv('SAVE_MASKED_IMAGES', 'false').lower() == 'true'

# Показ результатів
SHOW_RESULTS = os.getenv('SHOW_RESULTS', 'true').lower() == 'true'
SAVE_RESULTS = os.getenv('SAVE_RESULTS', 'false').lower() == 'true'

# Telegram налаштування
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '5694476991:AAFBBqkT_b1I8cwCdoxL-Q8mdN6kXNl6BWU')
TELEGRAM_CHANNEL_ID = os.getenv('TELEGRAM_CHANNEL_ID', '406058615')
TELEGRAM_CHANNEL_ID_KOVNIR = os.getenv('TELEGRAM_CHANNEL_ID_KOVNIR', '362669295')
TELEGRAM_CHANNEL_ID_JSZ = os.getenv('TELEGRAM_CHANNEL_ID_JSZ', '245458514')

# Періоди ротації
PERIOD_ROTATION = float(os.getenv('PERIOD_ROTATION', '0.9'))
CSV_ROTATION = int(os.getenv('CSV_ROTATION', '1800'))

# Кількість потоків детекції
NUM_DETECTION_THREADS = int(os.getenv('NUM_DETECTION_THREADS', '10'))

# Налаштування активності камер (формат: JDN=true,NRK=false,...)
CAMERAS_ACTIVE_STATUS = os.getenv('CAMERAS_ACTIVE_STATUS', '')

# Рівень логування (DEBUG, INFO, WARNING, ERROR, CRITICAL)
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# Логування конфігурації
logger.info("=== Environment Configuration ===")
logger.info(f"ENV_LEVEL: {ENV_LEVEL}")
logger.info(f"USE_DB: {USE_DB}")
logger.info(f"MODEL_VERSION: {MODEL_VERSION}")
logger.info(f"SAVE_RESULTS_FOR_MARKINGS: {SAVE_RESULTS_FOR_MARKINGS}")
logger.info(f"SAVE_DETECTED_IMAGES: {SAVE_DETECTED_IMAGES}")
logger.info(f"SAVE_MASKED_IMAGES: {SAVE_MASKED_IMAGES}")
logger.info(f"NUM_DETECTION_THREADS: {NUM_DETECTION_THREADS}")
logger.info("================================")

# Для зворотної сумісності
print(f"Environment Configuration loaded: ENV_LEVEL={ENV_LEVEL}, USE_DB={USE_DB}, MODEL_VERSION={MODEL_VERSION}")
