# PowerShell скрипт для запуску Docker контейнера з prod налаштуваннями

# Збірка образу (якщо потрібно)
Write-Host "Building Docker image..." -ForegroundColor Green
docker build -t sugar-inclusions .

# Запуск контейнера з prod налаштуваннями
Write-Host "Running container with PROD settings..." -ForegroundColor Green

# Читаємо prod.env файл
$prodEnvPath = "src/migrations/prod.env"

# Команда запуску
$volumeImages = "${PWD}/images:/sugar/images"
$volumeOutput = "${PWD}/output:/sugar/output"
$volumeModels = "${PWD}/images/shem_yolov9:/sugar/images/shem_yolov9"

Write-Host "Starting container with environment variables..." -ForegroundColor Yellow

# Виконуємо команду
$dockerArgs = @(
    "run", "-it", "--rm",
    "--name", "sugar-inclusions-prod",
    "-v", $volumeImages,
    "-v", $volumeOutput,
    "-v", $volumeModels
)

# Додаємо змінні середовища з prod.env (якщо файл існує)
if (Test-Path $prodEnvPath) {
    Get-Content $prodEnvPath | ForEach-Object {
        if ($_ -match "^(\w+)\s*=\s*'([^']*)'") {
            $dockerArgs += "-e"
            $dockerArgs += "$($matches[1])=$($matches[2])"
        }
    }
}

# Додаємо prod змінні
$dockerArgs += "-e", "ENV_LEVEL=prod"
$dockerArgs += "-e", "USE_DB=true"
$dockerArgs += "-e", "MODEL_VERSION=v9"
$dockerArgs += "-e", "SAVE_RESULTS_FOR_MARKINGS=false"  # На prod не зберігаємо
$dockerArgs += "-e", "SAVE_DETECTED_IMAGES=false"
$dockerArgs += "-e", "SAVE_MASKED_IMAGES=false"
$dockerArgs += "-e", "SHOW_RESULTS=false"
$dockerArgs += "-e", "SAVE_RESULTS=false"
$dockerArgs += "-e", "NUM_DETECTION_THREADS=10"
$dockerArgs += "-e", "PERIOD_ROTATION=2"  # Частіше на prod
$dockerArgs += "-e", "CSV_ROTATION=1800"

# ВАЖЛИВО: Встановіть правильні Telegram токени для prod!
$dockerArgs += "-e", "TELEGRAM_TOKEN=YOUR_PROD_TOKEN"
$dockerArgs += "-e", "TELEGRAM_CHANNEL_ID=YOUR_PROD_CHANNEL"
$dockerArgs += "-e", "TELEGRAM_CHANNEL_ID_KOVNIR=YOUR_PROD_CHANNEL_KOVNIR"
$dockerArgs += "-e", "TELEGRAM_CHANNEL_ID_JSZ=YOUR_PROD_CHANNEL_JSZ"
$dockerArgs += "-e", "LOG_LEVEL=INFO"

# Вимкнути недоступні камери (розкоментуйте при потребі)
$dockerArgs += "-e", "CAMERAS_ACTIVE_STATUS=JCZ=false,GLO=false,NVO=false"

# Додаємо назву образу
$dockerArgs += "sugar-inclusions"

Write-Host "Running: docker $($dockerArgs -join ' ')" -ForegroundColor Cyan

# Запускаємо Docker
& docker $dockerArgs

