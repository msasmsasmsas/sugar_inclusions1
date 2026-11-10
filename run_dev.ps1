# PowerShell скрипт для запуску в режимі розробки

# Встановлюємо змінні оточення
$env:ENV_LEVEL = "dev"
$env:USE_DB = "false"
$env:MODEL_VERSION = "9"
$env:SAVE_RESULTS_FOR_MARKINGS = "true"
$env:SAVE_DETECTED_IMAGES = "true"
$env:SHOW_RESULTS = "true"
$env:SAVE_RESULTS = "false"
$env:NUM_DETECTION_THREADS = "5"

# Виводимо конфігурацію
Write-Host "Starting Sugar Inclusions Classifier with configuration:" -ForegroundColor Green
Write-Host "ENV_LEVEL: $env:ENV_LEVEL"
Write-Host "USE_DB: $env:USE_DB"
Write-Host "MODEL_VERSION: $env:MODEL_VERSION"
Write-Host "NUM_DETECTION_THREADS: $env:NUM_DETECTION_THREADS"

# Активуємо віртуальне оточення якщо воно існує
if (Test-Path "venv\Scripts\Activate.ps1") {
    & "venv\Scripts\Activate.ps1"
}

# Запускаємо програму
python predict_yolo5.py
