# 🚀 Deployment Cheat Sheet

## Швидкі команди для deployment

### 📦 Базовий deployment
```powershell
# Тільки збірка та відправка в Harbor
.\deployment.ps1 -Version "0.1.37"
```

### 📦 + 🗄️ З міграцією БД
```powershell
# Збірка, відправка та міграція
.\deployment.ps1 -Version "0.1.37" -RunMigration
```

---

## 🔄 Workflow

### 1. Розробка → Тестування
```powershell
# Локальний тест
.\docker-run-dev.ps1

# Тест prod локально
.\docker-run-prod.ps1
```

### 2. Збірка → Harbor
```powershell
# Deployment в Harbor
.\deployment.ps1 -Version "0.1.37"
```

### 3. Harbor → Kubernetes
```bash
# Оновити K8s deployment
kubectl set image deployment/sugar-inclusions \
  ai-sugar-inclusions=harbor.agrichain.com.ua/agrichainserver/ai-sugar-inclusions:0.1.37

# Перевірити статус
kubectl rollout status deployment/sugar-inclusions

# Переглянути логи
kubectl logs -f deployment/sugar-inclusions
```

---

## 🗄️ Міграції БД

```powershell
# Створити нову міграцію
alembic -c src/migrations/alembic.ini revision --autogenerate -m "Description"

# Поточна версія БД
alembic -c src/migrations/alembic.ini current

# Застосувати міграції
alembic -c src/migrations/alembic.ini upgrade head

# Відкотити останню міграцію
alembic -c src/migrations/alembic.ini downgrade -1
```

---

## 🔍 Моніторинг

```bash
# Логи pod
kubectl logs -f deployment/sugar-inclusions

# Статус deployment
kubectl get deployment sugar-inclusions

# Детальна інформація
kubectl describe deployment sugar-inclusions

# Переглянути всі pods
kubectl get pods | grep sugar-inclusions
```

---

## ⚡ Швидкі дії

```bash
# Перезапустити deployment
kubectl rollout restart deployment/sugar-inclusions

# Rollback на попередню версію
kubectl rollout undo deployment/sugar-inclusions

# Rollback на конкретну версію
kubectl rollout undo deployment/sugar-inclusions --to-revision=2

# Історія rollout
kubectl rollout history deployment/sugar-inclusions
```

---

## 🧹 Очищення

```powershell
# Видалити локальні образи
docker rmi ai-sugar-inclusions:0.1.36

# Очистити build cache
docker builder prune

# Очистити всі unused образи
docker image prune -a
```

---

## 📊 Версіонування

- **Patch** (баг фікси): `0.1.36` → `0.1.37`
- **Minor** (нові функції): `0.1.37` → `0.2.0`
- **Major** (breaking changes): `0.2.0` → `1.0.0`

