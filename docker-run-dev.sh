#!/bin/bash
# Bash скрипт для запуску Docker контейнера з dev налаштуваннями

# Збірка образу (якщо потрібно)
echo -e "\033[32mBuilding Docker image...\033[0m"
docker build -t sugar-inclusions .

# Функція для читання .env файлу
load_env_file() {
    local env_file=$1
    while IFS= read -r line; do
        # Пропускаємо коментарі та порожні рядки
        if [[ ! "$line" =~ ^# ]] && [[ -n "$line" ]]; then
            # Парсимо формат DB_HOST = 'value'
            if [[ "$line" =~ ^([A-Z_]+)[[:space:]]*=[[:space:]]*\'([^\']*)\' ]]; then
                echo "-e ${BASH_REMATCH[1]}=${BASH_REMATCH[2]}"
            fi
        fi
    done < "$env_file"
}

# Читаємо змінні з dev.env
ENV_VARS=$(load_env_file "src/migrations/dev.env")

# Запуск контейнера
echo -e "\033[33mRunning container with dev settings...\033[0m"

docker run -it --rm \
    --name sugar-inclusions-dev \
    -v "$(pwd)/images:/sugar/images" \
    -v "$(pwd)/output:/sugar/output" \
    $ENV_VARS \
    -e ENV_LEVEL=dev \
    -e USE_DB=false \
    -e MODEL_VERSION=v9 \
    -e SAVE_RESULTS_FOR_MARKINGS=false \
    -e SAVE_DETECTED_IMAGES=true \
    -e SHOW_RESULTS=false \
    -e SAVE_RESULTS=false \
    -e NUM_DETECTION_THREADS=10 \
    -e PERIOD_ROTATION=10 \
    -e CSV_ROTATION=10 \
    -e TELEGRAM_TOKEN=YOUR_TOKEN \
    -e TELEGRAM_CHANNEL_ID=YOUR_CHANNEL \
    -e TELEGRAM_CHANNEL_ID_KOVNIR=YOUR_CHANNEL \
    -e TELEGRAM_CHANNEL_ID_JSZ=YOUR_CHANNEL \
    sugar-inclusions
