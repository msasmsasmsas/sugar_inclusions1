# PowerShell скрипт для запуску Docker контейнера з dev налаштуваннями

# Збірка образу (якщо потрібно)
Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t sugar-inclusions .

# Запуск контейнера з dev налаштуваннями
Write-Host "Running container with dev settings..." -ForegroundColor Green

# Читаємо dev.env файл та передаємо як змінні середовища
$devEnvPath = "src/migrations/dev.env"
$envVars = @()

# Читаємо файл та парсимо змінні
Get-Content $devEnvPath | ForEach-Object {
    if ($_ -match "^(\w+)\s*=\s*'([^']*)'") {
        $envVars += "-e `"$($matches[1])=$($matches[2])`""
    }
}

# Додаткові змінні з env_config
$additionalEnvVars = @(
    "-e ENV_LEVEL=dev",
    "-e USE_DB=false",
    "-e MODEL_VERSION=v9",
    "-e SAVE_RESULTS_FOR_MARKINGS=true",
    "-e SAVE_DETECTED_IMAGES=true",
    "-e SHOW_RESULTS=false",
    "-e SAVE_RESULTS=false",
    "-e NUM_DETECTION_THREADS=10",
    "-e PERIOD_ROTATION=10",
    "-e CSV_ROTATION=10",
    "-e TELEGRAM_TOKEN=YOUR_TOKEN",
    "-e TELEGRAM_CHANNEL_ID=YOUR_CHANNEL",
    "-e TELEGRAM_CHANNEL_ID_KOVNIR=YOUR_CHANNEL",
    "-e TELEGRAM_CHANNEL_ID_JSZ=YOUR_CHANNEL"
)

# Об'єднуємо всі змінні
$allEnvVars = $envVars + $additionalEnvVars -join " "

# Команда запуску
$volumeImages = "${PWD}/images:/sugar/images"
$volumeOutput = "${PWD}/output:/sugar/output"
$volumeModels = "${PWD}/images/shem_yolov9:/sugar/images/shem_yolov9"

Write-Host "Starting container with environment variables..." -ForegroundColor Yellow

# Виконуємо команду напряму без Invoke-Expression
$dockerArgs = @(
    "run", "-it", "--rm",
    "--name", "sugar-inclusions-dev",
    "-v", $volumeImages,
    "-v", $volumeOutput,
    "-v", $volumeModels
)

# Додаємо змінні середовища з dev.env
Get-Content $devEnvPath | ForEach-Object {
    if ($_ -match "^(\w+)\s*=\s*'([^']*)'") {
        $dockerArgs += "-e"
        $dockerArgs += "$($matches[1])=$($matches[2])"
    }
}

# Додаємо додаткові змінні
$dockerArgs += "-e", "ENV_LEVEL=dev"
$dockerArgs += "-e", "USE_DB=false"
$dockerArgs += "-e", "MODEL_VERSION=v9"
$dockerArgs += "-e", "SAVE_RESULTS_FOR_MARKINGS=false"
$dockerArgs += "-e", "SAVE_DETECTED_IMAGES=false"
$dockerArgs += "-e", "SAVE_MASKED_IMAGES=false"
$dockerArgs += "-e", "SHOW_RESULTS=false"
$dockerArgs += "-e", "SAVE_RESULTS=false"
$dockerArgs += "-e", "NUM_DETECTION_THREADS=10"
$dockerArgs += "-e", "PERIOD_ROTATION=10"
$dockerArgs += "-e", "CSV_ROTATION=10"
$dockerArgs += "-e", "TELEGRAM_TOKEN=YOUR_TOKEN"
$dockerArgs += "-e", "TELEGRAM_CHANNEL_ID=YOUR_CHANNEL"
$dockerArgs += "-e", "TELEGRAM_CHANNEL_ID_KOVNIR=YOUR_CHANNEL"
$dockerArgs += "-e", "TELEGRAM_CHANNEL_ID_JSZ=YOUR_CHANNEL"
$dockerArgs += "-e", "LOG_LEVEL=DEBUG"

# Приклад налаштування активності камер (розкоментуйте при потребі)
# $dockerArgs += "-e", "CAMERAS_ACTIVE_STATUS=NVO=false,GLO=false"

# Додаємо назву образу
$dockerArgs += "sugar-inclusions"

Write-Host "Running: docker $($dockerArgs -join ' ')" -ForegroundColor Cyan

# Запускаємо Docker
& docker $dockerArgs
