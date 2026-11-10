from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from src.migrations.config import settings

# Укажите ваши данные для подключения к базе данных
SQLALCHEMY_DATABASE_URL = settings.DATABASE_URL_pymysql

#"mysql+pymysql://user:password@localhost:3306/testdb"

engine = create_engine(SQLALCHEMY_DATABASE_URL, echo=True, pool_size=20, max_overflow=20, pool_timeout=30, pool_recycle=1800)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
