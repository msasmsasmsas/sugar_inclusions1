# 🚀 Deployment Guide - Sugar Inclusions Classifier

## 📋 Огляд

Скрипт `deployment.ps1` автоматизує процес deployment Docker образу в Harbor Registry та виконання міграцій бази даних.

## 🎯 Що робить скрипт:

1. ✅ Збирає Docker образ
2. 🏷️ Тегує образ для Harbor
3. 🔐 Логується в Harbor Registry
4. 📤 Відправляє образ в Harbor
5. 🗄️ Виконує міграцію БД (опціонально)

---

## 📝 Використання

### Базове використання (без міграції):
```powershell
.\deployment.ps1 -Version "0.1.37"
```

### З міграцією бази даних:
```powershell
.\deployment.ps1 -Version "0.1.37" -RunMigration
```

### Пропустити збірку (використати існуючий образ):
```powershell
.\deployment.ps1 -Version "0.1.37" -SkipBuild
```

### Тільки збірка (без відправки в Harbor):
```powershell
.\deployment.ps1 -Version "0.1.37" -SkipPush
```

### Кастомний Harbor Registry:
```powershell
.\deployment.ps1 -Version "0.1.37" `
    -HarborRegistry "my-harbor.example.com" `
    -HarborProject "my-project" `
    -ImageName "my-app"
```

---

## 🔧 Параметри скрипта

| Параметр | Обов'язковий | За замовчуванням | Опис |
|----------|--------------|------------------|------|
| `-Version` | ✅ Так | - | Версія образу (напр. "0.1.37") |
| `-RunMigration` | ❌ Ні | `false` | Виконати міграцію БД після deployment |
| `-SkipBuild` | ❌ Ні | `false` | Пропустити збірку Docker образу |
| `-SkipPush` | ❌ Ні | `false` | Пропустити відправку в Harbor |
| `-HarborRegistry` | ❌ Ні | `harbor.agrichain.com.ua` | URL Harbor Registry |
| `-HarborProject` | ❌ Ні | `agrichainserver` | Назва проекту в Harbor |
| `-ImageName` | ❌ Ні | `ai-sugar-inclusions` | Назва Docker образу |

---

## 📦 Приклади сценаріїв

### 1️⃣ Повний deployment з міграцією (Production)
```powershell
# Збирає образ, відправляє в Harbor, виконує міграцію БД
.\deployment.ps1 -Version "0.1.37" -RunMigration
```

**Що відбувається:**
- ✅ Збирає Docker образ `ai-sugar-inclusions`
- ✅ Тегує як `harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.37`
- ✅ Відправляє в Harbor Registry
- ✅ Виконує `alembic upgrade head`

---

### 2️⃣ Швидкий deployment без змін у БД
```powershell
# Тільки збірка та відправка, без міграції
.\deployment.ps1 -Version "0.1.37"
```

**Використовуйте коли:**
- Змінили код, але не змінювали схему БД
- Виправляли баги
- Оновлювали логіку детекції

---

### 3️⃣ Тестування збірки локально
```powershell
# Збирає образ, але не відправляє в Harbor
.\deployment.ps1 -Version "0.1.37-test" -SkipPush
```

**Використовуйте для:**
- Перевірки що образ збирається без помилок
- Локального тестування
- Дебагу Dockerfile

---

### 4️⃣ Швидка відправка існуючого образу
```powershell
# Використовує існуючий локальний образ
.\deployment.ps1 -Version "0.1.37" -SkipBuild
```

**Використовуйте коли:**
- Образ вже зібраний
- Потрібно перевідправити в Harbor
- Збірка займає багато часу

---

## 🗄️ Міграції бази даних

### Створення нової міграції

**1. Після змін в `models.py`:**
```powershell
# Активувати venv
.\venv\Scripts\Activate.ps1

# Створити міграцію
alembic -c src/migrations/alembic.ini revision --autogenerate -m "Add new column to table"
```

**2. Перевірити згенеровану міграцію:**
```powershell
# Переглянути файл міграції
code src/migrations/versions/<generated_file>.py
```

**3. Виконати deployment з міграцією:**
```powershell
.\deployment.ps1 -Version "0.1.37" -RunMigration
```

### Команди Alembic

```powershell
# Поточна версія БД
alembic -c src/migrations/alembic.ini current

# Історія міграцій
alembic -c src/migrations/alembic.ini history

# Виконати міграцію
alembic -c src/migrations/alembic.ini upgrade head

# Відкотити останню міграцію
alembic -c src/migrations/alembic.ini downgrade -1

# Показати SQL без виконання
alembic -c src/migrations/alembic.ini upgrade head --sql
```

---

## 🔐 Автентифікація в Harbor

### Перший раз:
```powershell
docker login harbor.agrichain.com.ua
```

Введіть ваші Harbor credentials:
- **Username**: ваш Harbor username
- **Password**: ваш Harbor password

### Перевірка логіну:
```powershell
# Переглянути збережені credentials
cat ~/.docker/config.json
```

---

## ☸️ Оновлення Kubernetes Deployment

Після успішного deployment виконайте:

### 1. Оновити образ в deployment:
```bash
kubectl set image deployment/sugar-inclusions \
  ai-sugar-inclusions=harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.37
```

### 2. Моніторити rollout:
```bash
kubectl rollout status deployment/sugar-inclusions
```

### 3. Перевірити логи:
```bash
kubectl logs -f deployment/sugar-inclusions
```

### 4. Якщо щось пішло не так - rollback:
```bash
kubectl rollout undo deployment/sugar-inclusions
```

---

## 🧹 Очищення старих образів

### Видалити локальні образи:
```powershell
# Видалити конкретну версію
docker rmi ai-sugar-inclusions:0.1.36
docker rmi harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.36

# Видалити всі dangling образи
docker image prune
```

### Видалити з Harbor:
1. Відкрийте https://harbor.agrichain.com.ua
2. Перейдіть в проект `agrichainserver`
3. Виберіть репозиторій `ai-sugar-inclusions`
4. Видаліть старі теги

---

## ⚠️ Важливі примітки

### 🔒 Безпека (КРИТИЧНО!)
- **НЕ зберігайте credentials в скриптах!**
- **НЕ комітьте Harbor паролі в Git!**
- Всі секрети завантажуються з Azure Key Vault через managed identity
- `.env` файли додані в `.gitignore`

### 📊 Версіонування
- Використовуйте семантичне версіонування: `MAJOR.MINOR.PATCH`
- Приклад: `0.1.37` → `0.1.38` (patch)
- При зміні API: `0.1.38` → `0.2.0` (minor)
- Breaking changes: `0.2.0` → `1.0.0` (major)

### 🔄 Rollback стратегія
1. Завжди тримайте мінімум 2 останні версії в Harbor
2. Перед deployment перевірте rollback plan
3. Тестуйте нову версію на dev середовищі

---

## 🐛 Troubleshooting

### Помилка: "Docker build failed"
```powershell
# Перевірити Docker daemon
docker info

# Очистити build cache
docker builder prune

# Спробувати ще раз
.\deployment.ps1 -Version "0.1.37"
```

### Помилка: "Docker login failed"
```powershell
# Видалити старі credentials
docker logout harbor.agrichain.com.ua

# Залогінитись знову
docker login harbor.agrichain.com.ua
```

### Помилка: "Alembic migration failed"
```powershell
# Перевірити поточну версію
alembic -c src/migrations/alembic.ini current

# Перевірити історію
alembic -c src/migrations/alembic.ini history

# Спробувати відкотити і накатити знову
alembic -c src/migrations/alembic.ini downgrade -1
alembic -c src/migrations/alembic.ini upgrade head
```

### Помилка: "Image not found in Harbor"
```powershell
# Перевірити чи образ існує
docker images | Select-String "ai-sugar-inclusions"

# Перевірити чи push успішний
docker manifest inspect harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.37
```

---

## 📞 Контакти та підтримка

При виникненні проблем:
1. Перевірте логи: `kubectl logs -f deployment/sugar-inclusions`
2. Перевірте Harbor Registry: https://harbor.agrichain.com.ua
3. Перевірте стан БД: `alembic -c src/migrations/alembic.ini current`

---

## 📚 Додаткові ресурси

- [Dockerfile](./Dockerfile) - конфігурація образу
- [docker-run-prod.ps1](./docker-run-prod.ps1) - локальний запуск prod
- [docker-run-dev.ps1](./docker-run-dev.ps1) - локальний запуск dev
- [k8s/configmap.yaml](./k8s/configmap.yaml) - Kubernetes конфігурація
- [src/migrations/README](./src/migrations/README) - інструкції по міграціям

