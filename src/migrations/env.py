import os
from logging.config import fileConfig

from sqlalchemy import engine_from_config
from sqlalchemy import pool

from alembic import context
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


print("All environment variables:")
print(os.environ)
logger.info(f'welcome env.py')
logger.info(f'environ {os.environ}')

print(f'welcome env.py')
# Загрузка переменных окружения из файла .env


# Спробуємо імпортувати з env_config, якщо не вдасться - використаємо os.getenv
try:
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from env_config import ENV_LEVEL as ENV
    print(f'ENV loaded from env_config: {ENV}')
except ImportError:
    # Fallback - беремо з змінних оточення
    ENV = os.getenv('ENV_LEVEL', 'dev')
    print(f'ENV loaded from os.getenv: {ENV}')

logger.info(f'ENV = {ENV}')



dotenv_path = f"src/migrations/{ENV}.env"
load_dotenv(dotenv_path)

logger.info(f'Loading environment variables from {dotenv_path}')
print(f'ENV = {ENV}')
logger.info(f'ENV = {ENV}')
# Отладочный вывод для проверки загрузки переменных окружения

print(f"DB_HOST: {os.getenv('DB_HOST')}")
print(f"DB_PORT: {os.getenv('DB_PORT')}")
print(f"DB_USER: {os.getenv('DB_USER')}")
#print(f"DB_PASS: {os.getenv('DB_PASS')}")
print(f"DB_NAME: {os.getenv('DB_NAME')}")
logger.info(f"DB_HOST: {os.getenv('DB_HOST')}")
logger.info(f"DB_PORT: {os.getenv('DB_PORT')}")
logger.info(f"DB_USER: {os.getenv('DB_USER')}")
#logger.info(f"DB_PASS: {os.getenv('DB_PASS')}")
logger.info(f"DB_NAME: {os.getenv('DB_NAME')}")

from src.migrations.database import Base
import src.migrations.models
from src.migrations.config import DATABASE_URL
from src.migrations.create_db import create_db  # Импортируем создание базы данных

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
print(context.config)
logger.info(f'config alembic  {context.config}')

# Interpret the config file for Python logging.
# This line sets up loggers basically.
fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
# from myapp import mymodel
# target_metadata = mymodel.Base.metadata
target_metadata = Base.metadata

# Set the SQLAlchemy URL from the environment variable
config.set_main_option('sqlalchemy.url', DATABASE_URL)
logger.info(f'preceed complete  {DATABASE_URL}')


def run_migrations_offline():
    """Run migrations in 'offline' mode.
    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well. By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online():
    """Run migrations in 'online' mode.
    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    # Создание базы данных перед выполнением миграций
    create_db()

    connectable = engine_from_config(
        config.get_section(config.config_ini_section),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()

logger.info(f'start migration  {DATABASE_URL}')
if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
logger.info(f'finish migration  {DATABASE_URL}')