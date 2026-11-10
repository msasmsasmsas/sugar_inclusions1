# Змінні оточення для Sugar Inclusions Classifier

## Опис змінних

| Змінна | Опис | Значення за замовчуванням | Приклади |
|--------|------|---------------------------|----------|
| `ENV_LEVEL` | Рівень оточення | `dev` | `dev`, `prod`, `local` |
| `USE_DB` | Використовувати базу даних | `false` | `true`, `false` |
| `MODEL_VERSION` | Версія YOLO моделі | `9` | `5`, `8`, `9` |
| `SAVE_RESULTS_FOR_MARKINGS` | Зберігати результати для розмітки | `true` | `true`, `false` |
| `SAVE_DETECTED_IMAGES` | Зберігати розмічені зображення | `true` | `true`, `false` |
| `SHOW_RESULTS` | Показувати результати | `true` | `true`, `false` |
| `SAVE_RESULTS` | Зберігати результати моделі | `false` | `true`, `false` |
| `TELEGRAM_TOKEN` | Токен Telegram бота | - | Ваш токен |
| `TELEGRAM_CHANNEL_ID` | ID основного каналу | - | ID каналу |
| `TELEGRAM_CHANNEL_ID_KOVNIR` | ID каналу Ковнір | - | ID каналу |
| `TELEGRAM_CHANNEL_ID_JSZ` | ID каналу ЖСЗ | - | ID каналу |
| `PERIOD_ROTATION` | Період захоплення фото (сек) | `0.9` | `0.5` - `5.0` |
| `CSV_ROTATION` | Період відправки CSV (сек) | `1800` | `600` - `3600` |
| `NUM_DETECTION_THREADS` | Кількість потоків детекції | `10` | `1` - `20` |
| `CAMERAS_ACTIVE_STATUS` | Налаштування активності камер | - | `JDN=true,NRK=false` |

## Способи встановлення

### 1. Через файл .env (рекомендовано для локальної розробки)

1. Встановіть python-dotenv:
   ```bash
   pip install python-dotenv
   ```

2. Створіть файл `.env` в кореневій директорії проекту:
   ```env
   ENV_LEVEL=dev
   USE_DB=false
   MODEL_VERSION=9
   SAVE_RESULTS_FOR_MARKINGS=true
   SAVE_DETECTED_IMAGES=true
   ```

### 2. Через PowerShell скрипти

Використовуйте готові скрипти:
```powershell
# Для розробки
.\run_dev.ps1

# Для production
.\run_prod.ps1
```

### 3. Вручну в терміналі

PowerShell:
```powershell
$env:ENV_LEVEL = "dev"
$env:USE_DB = "false"
$env:MODEL_VERSION = "9"
python predict_yolo5.py
```

Bash:
```bash
export ENV_LEVEL=dev
export USE_DB=false
export MODEL_VERSION=9
python predict_yolo5.py
```

### 4. Через VS Code / Cursor

Використовуйте файл `.vscode/launch.json` та виберіть потрібну конфігурацію в Debug панелі.

### 5. В Kubernetes

Застосуйте ConfigMap та Secrets:
```bash
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
```

## Логіка роботи залежно від ENV_LEVEL

### ENV_LEVEL=prod
- `prod_release = True`
- Використовуються production IP адреси камер
- Відправка в усі Telegram канали
- Не зберігаються файли для розмітки
- Рекомендовано `USE_DB=true`

### ENV_LEVEL=dev
- `prod_release = False`
- Використовуються dev IP адреси камер
- Відправка тільки в основний канал
- Зберігаються файли для розмітки
- Рекомендовано `USE_DB=false`

### ENV_LEVEL=local
- `prod_release = False`
- Використовуються локальні IP адреси камер (192.168.x.x)
- Мінімальне логування
- Максимальне збереження для тестування

## Налаштування активності камер

### CAMERAS_ACTIVE_STATUS
Дозволяє динамічно вмикати/вимикати окремі камери без зміни коду.

**Формат**: `CAMERA_KEY=true/false,CAMERA_KEY2=true/false,...`

**Приклади**:
- `JDN=false` - вимкнути тільки камеру JDN
- `JDN=false,NRK=false` - вимкнути камери JDN та NRK
- `GLO=false,NVO=false` - вимкнути камери GLO та NVO

**Доступні ключі камер**:
- `JDN` - Жданівка
- `NRK` - Наркевичі  
- `JCZ` - Яреськи
- `GLO` - Глобине
- `NVO` - Новооржиця

**Примітка**: За замовчуванням всі камери активні. Якщо змінна не встановлена, використовуються значення `is_active` з коду.
