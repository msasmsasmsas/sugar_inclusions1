# Этап сборки
FROM python:3.12-slim AS builder

# Отключение записи байт-кода и буферизации
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Установка необходимых инструментов для сборки и зависимости
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libmariadb-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Установка рабочего каталога
WORKDIR /sugar

# Обновление pip и установка зависимостей
RUN pip install --upgrade pip
COPY requirements.txt /sugar/requirements.txt

# Создаём wheel пакеты для всех зависимостей БЕЗ установки PyTorch на этом этапе
# Установка numpy сначала для избежания конфликтов
RUN pip install --no-cache-dir "numpy>=2.0,<2.3"
RUN pip wheel --no-cache-dir --no-deps --wheel-dir /sugar/wheels -r requirements.txt

# Финальный этап
FROM python:3.12-slim

# Отключение записи байт-кода
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Установка необходимых библиотек для выполнения
RUN apt-get update && \
    apt-get install -y --no-install-recommends libmariadb3 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Копирование wheel пакетов из этапа сборки и установка зависимостей
COPY --from=builder /sugar/wheels /wheels
COPY --from=builder /sugar/requirements.txt .

# Установка PyTorch и зависимостей (только один раз)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir "numpy>=2.0,<2.3" && \
    pip install --no-cache-dir /wheels/* && \
    rm -rf /wheels && \
    rm -rf /root/.cache/pip

# Установка рабочего каталога
WORKDIR /sugar

# Копирование только необходимых файлов
COPY *.py ./
COPY requirements.txt ./

# Копирование необходимых Python модулей
COPY dir_files ./dir_files
COPY dahua_nwr_downloader ./dahua_nwr_downloader
COPY MariaDB ./MariaDB
COPY send_mail ./send_mail
COPY src ./src
COPY yolov5_train_detect ./yolov5_train_detect
COPY yolov8_train_detect ./yolov8_train_detect
COPY yolov9_train_detect ./yolov9_train_detect

# Копирование конфигурационных файлов
COPY input ./input

# Копирование моделей YOLO
COPY images/shem_yolov9 ./images/shem_yolov9

# Создание необходимых директорий
RUN mkdir -p images/for_detection images/for_marking images/marked images/source images/detected output/csv output/logs output/marked

# Команда запуска приложения
CMD ["python", "predict_yolo5.py"]


# docker build -t sugar-inclusions .
# docker run -it -p 80:80 --rm sugar-inclusions
# docker run -it -p 81:81 --rm -e PROD_RELEASE=false -e ENV_LEVEL=dev sugar-inclusions
# docker ps -a
# C:\Users\msts\Downloads\dive_0.9.2_windows_amd64\dive.exe sugar-inclusions
# Копирование из контейнера на диски системы
# docker cp  infallible_wing:/app/images/for_marking e:\\

# docker tag sugar-inclusions harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.36
# docker login harbor.agrichain.com.ua
#
#
# docker push harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.36
# docker rm harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.35

# alembic migration
#alembic -c src/migrations/alembic.ini revision --autogenerate -m "Added initial tables"
#alembic -c src/migrations/alembic.ini -x echo=true upgrade head
#alembic -c src/migrations/alembic.ini revision --autogenerate -m "Add model_files table"