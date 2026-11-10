from sqlalchemy import create_engine
from sqlalchemy_utils import database_exists, create_database
from src.migrations.config import settings, DATABASE_URL
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

print(f'welcome create_db.py')
logger.info(f'welcome create_db.py')



def create_db():
    engine = create_engine(settings.DATABASE_URL_pymysql)
    if not database_exists(engine.url):
        create_database(engine.url)
        print(f"Database {settings.DB_NAME} created.")
        logger.error(f"Database {settings.DB_NAME} created.")
    else:
        print(f"Database {settings.DB_NAME} already exists.")
        logger.error(f"Database {settings.DB_NAME} created.")

if __name__ == "__main__":
    create_db()
