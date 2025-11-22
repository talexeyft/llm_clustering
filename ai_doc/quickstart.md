# Quick Start

## Установка

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Настройка

```bash
cp env.example .env
# Убедитесь, что Ollama установлена и доступна локально
```

По умолчанию проект настроен на использование локальной Ollama с моделью `qwen3:30b`.
Перед запуском выполните `ollama serve` и `ollama pull qwen3:30b`.

### Обязательные переменные окружения

- `DEFAULT_LLM_PROVIDER`, `DEFAULT_MODEL`, `DEFAULT_TEMPERATURE`, `DEFAULT_MAX_TOKENS`
- `CLUSTERING_BATCH_SIZE`, `MAX_CLUSTERS_PER_BATCH`, `MIN_REQUESTS_PER_CLUSTER`
- `LOG_LEVEL`, `LOG_FILE`, `LOG_PROMPTS`, `LOG_PROMPT_DIR`

Все переменные перечислены в `env.example`. Любые дополнительные ключи/URL копируйте туда, затем запускайте `source venv/bin/activate && export $(cat .env | xargs)`.

### Требования к данным

Мы работаем только с `pandas.DataFrame`. На вход Pipeline Runner подается таблица со столбцами:

- `request_id` — строковый идентификатор обращения (уникален в рамках batch_id)
- `text` — очищенный или сырой текст обращения
- Доп. контекст (`channel`, `priority`, `metadata`) передается в промпты как опциональные столбцы

При подготовке батчей сохраняйте как минимум два столбца с текстом:

1. `text_raw` — исходный текст
2. `text_clean` — нормализованная версия, которую читают LLM

Сырые/очищенные выгрузки складываются в `ai_data/batches/`, результаты присвоений — в `ai_data/results/`.

### Тестовый датасет из ~/data/subs

```bash
# Собрать первые 1000 сообщений в parquet/csv
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.data.subs_dataset --limit 1000
```

После выполнения появятся файлы `ai_data/subs_sample_1000.parquet` и `.csv`.
Их можно использовать в качестве стандартного входа для пайплайна.

## Запуск MVP пайплайна

```bash
# пример: parquet c тестовыми обращениями subs
INPUT=ai_data/subs_sample_1000.parquet make run

# указать batch_id и альтернативное имя текстового столбца
INPUT=ai_data/support.csv BATCH=batch-20241121 TEXT_COL=message make run
```

После выполнения:

- готовые батчи и снапшоты лежат в `ai_data/batches/`
- результаты решений Assignment Judge — `ai_data/results/<batch_id>.parquet` и `.csv`
- логи промптов/ответов — `ai_data/prompts/`
- отчеты QA (cohesion) — `ai_data/reports/`

### Использование локальной Ollama

Локальная Ollama является дефолтным провайдером.

1. Проверьте, что сервис запущен:
   ```bash
   ollama serve
   ```
2. Убедитесь, что модель скачана:
   ```bash
   ollama pull qwen3:30b
   ```
3. В `.env` уже выставлены:
   ```
   DEFAULT_LLM_PROVIDER=ollama
   OLLAMA_MODEL=qwen3:30b
   ```

### Использование Triton провайдера

Для использования Triton Inference Server провайдера:

1. Убедитесь, что Triton сервер запущен (см. раздел "Использование Triton Inference Server" выше)

2. В `.env` установите:
   ```
   DEFAULT_LLM_PROVIDER=triton
   TRITON_API_URL=http://localhost:8000
   TRITON_MODEL=qwen3_30b_4bit
   ```

### Использование Triton Inference Server

Для использования Triton Inference Server:

1. Убедитесь, что установлен Docker и nvidia-container-toolkit:
   ```bash
   # Установка nvidia-container-toolkit (если не установлен)
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
   curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
   sudo apt-get update
   sudo apt-get install -y nvidia-container-toolkit
   sudo nvidia-ctk runtime configure --runtime=docker
   sudo systemctl restart docker
   ```

2. Запустите Triton сервер:
   ```bash
   cd /home/alex/tradeML/llm_clustering
   docker run -d --name triton-server --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
     -v $(pwd)/ai_data/triton_models:/models \
     nvcr.io/nvidia/tritonserver:24.09-py3 \
     tritonserver --model-repository=/models
   ```

3. Проверьте статус:
   ```bash
   curl http://localhost:8000/v2/health/ready
   docker logs triton-server
   ```

4. Остановка сервера:
   ```bash
   docker stop triton-server
   docker rm triton-server
   ```

**Примечание**: Triton сервер работает на портах:
- 8000 - HTTP API
- 8001 - gRPC API
- 8002 - Metrics API

### Настройка модели Qwen 3 30B 4-bit в Triton

Модель Qwen 3 30B 4-bit уже настроена в `ai_data/triton_models/qwen3_30b_4bit/`.

Для работы модели необходимо установить зависимости. См. подробные инструкции в `ai_data/triton_models/qwen3_30b_4bit/README.md`.

**Быстрый старт** (с кастомным Docker образом):

```bash
# Создать кастомный образ с зависимостями
docker build -f ai_data/Dockerfile.triton -t triton-qwen:latest .

# Остановить текущий сервер
docker stop triton-server && docker rm triton-server

# Запустить с кастомным образом
cd /home/alex/tradeML/llm_clustering
docker run -d --name triton-server --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/ai_data/triton_models:/models \
  triton-qwen:latest \
  tritonserver --model-repository=/models --model-control-mode=poll
```

После запуска модель автоматически загрузится (может занять несколько минут при первом запуске).

## Базовое использование

```python
from llm_clustering.clustering import Clusterer
import pandas as pd

# Загрузка данных
inquiries = pd.read_csv("data/inquiries.csv")

# Кластеризация
clusterer = Clusterer()
clustered = clusterer.cluster_inquiries(inquiries, text_column="text")

# Описание кластеров
descriptions = clusterer.describe_clusters(clustered)
```

## Запуск тестов

```bash
pytest
```

## Тестирование Ollama

Для тестирования работы Ollama:

```bash
source venv/bin/activate
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_ollama_success.py
```

**Примечание**: Провайдер Ollama использует `urllib3` вместо `requests` из-за проблем совместимости с Ollama сервером.

## Тестирование OpenRouter

Для тестирования работы OpenRouter:

```bash
source venv/bin/activate
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_openrouter_simple.py
```

Тест автоматически выберет доступную простую модель (исключая бесплатные rate-limited модели).

## Тестирование Triton

Для тестирования работы Triton провайдера:

```bash
source venv/bin/activate
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_triton.py
```

**Примечание**: Убедитесь, что Triton сервер запущен и модель загружена перед запуском теста.

