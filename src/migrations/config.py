import os
from pydantic_settings import BaseSettings, SettingsConfigDict
import logging

# Спробуємо імпортувати централізовану конфігурацію
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from env_config import ENV_LEVEL as ENV
except ImportError:
    # Fallback якщо env_config.py недоступний
    ENV = os.getenv('ENV_LEVEL','dev')

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print(f'welcome config.py')
logger.info(f'welcome config.py')
print(f'ENV = {ENV}')
logger.info(f'ENV = {ENV}')
dotenv_path = f"src/migrations/{ENV}.env"

class Settings(BaseSettings):
    # Спочатку намагаємось взяти з змінних оточення, потім з файлу
    DB_HOST: str = os.getenv('DB_HOST', '')
    DB_PORT: int = int(os.getenv('DB_PORT', '3306'))
    DB_USER: str = os.getenv('DB_USER', '')
    DB_PASS: str = os.getenv('DB_PASS', '')
    DB_NAME: str = os.getenv('DB_NAME', '')

    @property
    def DATABASE_URL_pymysql(self):
        return f"mysql+pymysql://{self.DB_USER}:{self.DB_PASS}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    model_config = SettingsConfigDict(env_file=dotenv_path if not os.getenv('DB_HOST') else None)

# Создаем экземпляр настроек
settings = Settings()

# Используем DATABASE_URL для подключения к базе данных
DATABASE_URL = settings.DATABASE_URL_pymysql

logger.info(f"DB_HOST: {settings.DB_HOST}")
logger.info(f"DB_PORT: {settings.DB_PORT}")
logger.info(f"DB_USER: {settings.DB_USER}")
#logger.info(f"DB_PASS: {settings.DB_PASS}")
logger.info(f"DB_NAME: {settings.DB_NAME}")
logger.info(DATABASE_URL)

print(f"DB_HOST: {settings.DB_HOST}")
print(f"DB_PORT: {settings.DB_PORT}")
print(f"DB_USER: {settings.DB_USER}")
#print(f"DB_PASS: {settings.DB_PASS}")
print(f"DB_NAME: {settings.DB_NAME}")
print(DATABASE_URL)
