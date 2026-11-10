# Розгортання в Hetzner Cloud Kubernetes

## Передумови

1. Hetzner Cloud Kubernetes кластер
2. kubectl налаштований для доступу до кластера
3. Docker Registry для зберігання образів

## Крок 1: Створення секретів

### 1.1 Створення секрету для БД

```bash
# Опція 1: Використовуйте готовий файл (НЕ комітьте його в git!)
kubectl apply -f k8s/db-secret.yaml

# Опція 2: Створіть секрет напряму
kubectl create secret generic db-credentials \
  --from-literal=db-host="10.47.55.177" \
  --from-literal=db-port="3306" \
  --from-literal=db-user="root" \
  --from-literal=db-pass="YOUR_SECURE_PASSWORD" \
  --from-literal=db-name="ai_sugar_inclusions"
```

### 1.2 Створення секрету для Telegram

```bash
kubectl apply -f k8s/secrets.yaml

# Або
kubectl create secret generic sugar-inclusions-secrets \
  --from-literal=telegram-token="YOUR_BOT_TOKEN" \
  --from-literal=telegram-channel-id="YOUR_CHANNEL_ID" \
  --from-literal=telegram-channel-id-kovnir="KOVNIR_CHANNEL_ID" \
  --from-literal=telegram-channel-id-jsz="JSZ_CHANNEL_ID"
```

## Крок 2: Створення ConfigMap

```bash
kubectl apply -f k8s/configmap.yaml
```

## Крок 3: Створення Persistent Volumes

Hetzner Cloud автоматично провізує volumes через StorageClass:

```yaml
# k8s/pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sugar-images-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 10Gi
  storageClassName: hcloud-volumes
---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sugar-output-pvc
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 5Gi
  storageClassName: hcloud-volumes
```

```bash
kubectl apply -f k8s/pvc.yaml
```

## Крок 4: Розгортання додатку

```bash
kubectl apply -f k8s/deployment.yaml
```

## Крок 5: Перевірка

```bash
# Перевірка статусу
kubectl get pods
kubectl get secrets
kubectl get configmaps

# Перегляд логів
kubectl logs -f deployment/sugar-inclusions

# Перевірка змінних оточення в поді
kubectl exec -it deployment/sugar-inclusions -- env | grep DB_
```

## Безпека

### Використання Sealed Secrets (рекомендовано)

1. Встановіть sealed-secrets controller:
```bash
kubectl apply -f https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/controller.yaml
```

2. Встановіть kubeseal CLI:
```bash
wget https://github.com/bitnami-labs/sealed-secrets/releases/download/v0.18.0/kubeseal-0.18.0-linux-amd64.tar.gz
tar -xvzf kubeseal-0.18.0-linux-amd64.tar.gz
sudo install -m 755 kubeseal /usr/local/bin/kubeseal
```

3. Створіть sealed secret:
```bash
# Створіть звичайний секрет
kubectl create secret generic db-credentials \
  --from-literal=db-host="mariadb" \
  --from-literal=db-pass="SECURE_PASSWORD" \
  --dry-run=client -o yaml > db-secret-temp.yaml

# Зашифруйте його
kubeseal -o yaml < db-secret-temp.yaml > db-sealed-secret.yaml

# Видаліть тимчасовий файл
rm db-secret-temp.yaml

# Застосуйте sealed secret
kubectl apply -f db-sealed-secret.yaml
```

## Налаштування MariaDB в Kubernetes

Якщо БД також в Kubernetes:

```yaml
# k8s/mariadb.yaml
apiVersion: v1
kind: Service
metadata:
  name: mariadb
spec:
  ports:
  - port: 3306
  selector:
    app: mariadb
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mariadb
spec:
  selector:
    matchLabels:
      app: mariadb
  template:
    metadata:
      labels:
        app: mariadb
    spec:
      containers:
      - name: mariadb
        image: mariadb:10.6
        env:
        - name: MYSQL_ROOT_PASSWORD
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: db-pass
        - name: MYSQL_DATABASE
          valueFrom:
            secretKeyRef:
              name: db-credentials
              key: db-name
        ports:
        - containerPort: 3306
        volumeMounts:
        - name: mariadb-storage
          mountPath: /var/lib/mysql
      volumes:
      - name: mariadb-storage
        persistentVolumeClaim:
          claimName: mariadb-pvc
```

## Приклад повного розгортання

```bash
# 1. Створити namespace (опціонально)
kubectl create namespace sugar-inclusions

# 2. Перемкнутись на namespace
kubectl config set-context --current --namespace=sugar-inclusions

# 3. Створити секрети
kubectl create secret generic db-credentials \
  --from-literal=db-host="mariadb" \
  --from-literal=db-port="3306" \
  --from-literal=db-user="root" \
  --from-literal=db-pass="$(openssl rand -base64 32)" \
  --from-literal=db-name="ai_sugar_inclusions"

# 4. Застосувати всі конфігурації
kubectl apply -f k8s/

# 5. Перевірити статус
kubectl get all
```

## Оновлення секретів

```bash
# Видалити старий секрет
kubectl delete secret db-credentials

# Створити новий
kubectl create secret generic db-credentials \
  --from-literal=db-host="NEW_HOST" \
  --from-literal=db-port="3306" \
  --from-literal=db-user="NEW_USER" \
  --from-literal=db-pass="NEW_PASSWORD" \
  --from-literal=db-name="ai_sugar_inclusions"

# Перезапустити deployment
kubectl rollout restart deployment/sugar-inclusions
```


