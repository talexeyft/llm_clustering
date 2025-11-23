# Quick Start

> **Примечание**: Основная документация находится в [README.md](../README.md) в корне проекта.

## Установка

```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Или установить как библиотеку в editable mode
pip install -e .

# Проверить установку
python -c "import llm_clustering; print(llm_clustering.__version__)"
llm-clustering --help
```

## Демо-данные

В репозитории включен файл `ai_data/demo_sample.csv.zip` с 1000 случайных сэмплов для демонстрации работы кластеризации.

```bash
# Распаковать демо-данные
unzip ai_data/demo_sample.csv.zip -d ai_data/

# Запустить кластеризацию на демо-данных (20 сэмплов)
source venv/bin/activate
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/demo_sample.csv \
  --limit 20 \
  --batch-id demo_test
```

Этот файл предназначен для быстрого старта и демонстрации возможностей без необходимости настройки полного датасета.

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
# Собрать первые 1000 сообщений в единый parquet/csv файл
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.data.subs_dataset --limit 1000
```

После выполнения появятся файлы `ai_data/subs_sample.parquet` и `.csv` (единый файл для всех тестов).
Используйте параметр `--limit` при запуске для ограничения количества обращений.

**Примечание:** Старые файлы с суффиксами размера (`subs_sample_20.parquet`, `subs_sample_100.parquet` и т.д.) больше не создаются. Если они есть в `ai_data/`, можно их удалить:
```bash
rm -f ai_data/subs_sample_*.parquet ai_data/subs_sample_*.csv
```

## Запуск MVP пайплайна

```bash
# пример: parquet c тестовыми обращениями subs (все сообщения из файла)
INPUT=ai_data/subs_sample.parquet make run

# ограничить количество обращений через --limit
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main --input ai_data/subs_sample.parquet --limit 100

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

## Использование как библиотека

### Установка в любое ядро/окружение

```bash
# Установка в editable mode для разработки
pip install -e /path/to/llm_clustering

# Или установка как обычный пакет
pip install /path/to/llm_clustering

# Проверка установки
python -c "import llm_clustering; print(llm_clustering.__version__)"
```

### Базовое использование

```python
from llm_clustering import ClusteringPipeline
import pandas as pd

# Создать pipeline с настройками по умолчанию
pipeline = ClusteringPipeline()

# Загрузить данные
df = pd.DataFrame({
    "text": ["Не могу войти", "Забыл пароль", "Товар не пришел"]
})

# Выполнить кластеризацию
result = pipeline.fit(df, text_column="text")

# Результаты
print(f"Coverage: {result.coverage:.1f}%")
print(f"Clusters found: {len(result.clusters)}")
print(result.assignments.head())
```

### Использование с кастомной LLM (УПРОЩЕНО!)

```python
from llm_clustering import ClusteringPipeline, SimpleLLMProvider

class MyCustomLLM(SimpleLLMProvider):
    """Только 1 метод нужен - chat_completion()!"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.your-llm.com"
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        # Ваша реализация вызова LLM
        import requests
        response = requests.post(
            f"{self.api_url}/chat",
            json={"messages": messages, "temperature": temperature},
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["content"]
    
    # describe_cluster() работает автоматически!
    # embed() и cluster() опциональны (NotImplementedError по умолчанию)

# Использование кастомной LLM
custom_llm = MyCustomLLM(api_key="your-key")
pipeline = ClusteringPipeline(llm_provider=custom_llm)
result = pipeline.fit(df, text_column="text")
```

**Преимущества SimpleLLMProvider:**
- ✅ Только 1 метод: `chat_completion()`
- ✅ `describe_cluster()` работает автоматически
- ✅ `embed()` и `cluster()` опциональны
- ✅ В 4 раза проще чем `BaseLLMProvider`

Подробнее: см. `doc/adding_custom_provider.md` и `examples.py` (Example 2)

### Итеративная обработка

```python
# Обработка большого датасета по частям
for partial in pipeline.fit_partial(df, batch_size=50, start_from=0):
    print(f"Batch {partial.batch_number}: {partial.processed_rows}/{partial.total_rows}")
    print(f"New clusters: {len(partial.new_clusters)}")
    
    # Можно остановиться в любой момент
    if partial.processed_rows >= 200:
        break
```

### Доразметка данных

```python
# Первоначальная разметка
result1 = pipeline.fit(df_part1, text_column="text")

# Позже: доразметка с учетом найденных кластеров
result2 = pipeline.refit(
    df_part2,
    previous_assignments=result1.assignments,
    text_column="text"
)

# Кластеры обновлены и уточнены
all_clusters = pipeline.get_clusters()
```

### Сохранение и загрузка кластеров

```python
from pathlib import Path

# Сохранить кластеры
pipeline.save_clusters(Path("my_clusters.json"))

# Загрузить в новый pipeline
new_pipeline = ClusteringPipeline()
new_pipeline.load_clusters(Path("my_clusters.json"))

# Использовать загруженные кластеры
result = new_pipeline.fit(new_df, text_column="text")
```

### Использование бизнес-контекста

```python
# Добавление контекста для специфичной кластеризации
business_context = """
Разметка обращений для улучшения бота поддержки.
Разделяй проблемы по сложности доработки:
- Простые: FAQ, типовые вопросы (автоматизируемые)
- Средние: требуют проверки данных
- Сложные: нестандартные ситуации
"""

pipeline = ClusteringPipeline(business_context=business_context)
result = pipeline.fit(df, text_column="text")

# LLM будет учитывать бизнес-контекст при кластеризации
```

### Полный пример с настройками

```python
from llm_clustering import ClusteringPipeline, Settings
from pathlib import Path

# Кастомные настройки
settings = Settings(
    clustering_batch_size=30,
    max_clusters_per_batch=5,
    default_temperature=0.1,
    default_llm_provider="ollama",
    ollama_model="qwen3:30b",
)

# Инициализация с настройками
pipeline = ClusteringPipeline(
    settings=settings,
    business_context="Разметка для бота поддержки",
    registry_path=Path("ai_data/my_clusters.json")
)

# Обработка
result = pipeline.fit(df, text_column="text", limit=100)

# Работа с результатами
for cluster in result.clusters:
    print(f"{cluster.name}: {cluster.count} requests")
    print(f"  Summary: {cluster.summary}")

# Сохранение результатов
result.assignments.to_csv("results.csv", index=False)
pipeline.save_clusters(Path("final_clusters.json"))
```

Больше примеров смотрите в `examples.py`.

## Запуск тестов

```bash
PYTHONPATH=src:$PYTHONPATH pytest
```

## Запуск на тестовых данных через Ollama

```bash
# Убедитесь, что Ollama запущен
ollama serve

# Запуск на 20 записях (быстрый тест)
source venv/bin/activate
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/subs_sample.parquet \
  --limit 20 \
  --batch-id test_quick

# Запуск на 100 записях (батчи по 20)
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/subs_sample.parquet \
  --limit 100 \
  --batch-id test_100
```

**Рекомендуемые модели для Ollama:**
- `qwen3:30b-a3b` (30B MoE) - быстрая, 100% стабильность после улучшений парсера ✅
- `gemma3:latest` (4B) - быстрая, хорошо работает с JSON
- `qwen3:32b` - более качественные результаты, но медленнее

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

## Шпаргалка команд

### Основные команды запуска

```bash
# Базовая кластеризация
llm-clustering --input ai_data/demo_sample.csv --limit 20

# С указанием batch-id
llm-clustering --input ai_data/demo_sample.csv --limit 100 --batch-id test_100

# С кастомным текстовым столбцом
llm-clustering --input data.csv --text-column message --limit 50

# Использование через Python модуль
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main --input ai_data/demo_sample.csv --limit 20
```

### Тестовые команды

```bash
# Standalone пример: полная кластеризация 1000 обращений с Ollama и кастомным промптом
python examples/standalone_ollama_example.py

# Тест библиотечного API
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_library_api.py

# Тест Ollama
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_ollama_success.py

# Тест OpenRouter
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_openrouter_simple.py

# Тест Triton
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_triton.py

# Тест параллельной обработки
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_parallel_inference.py

# Запуск unit-тестов
PYTHONPATH=src:$PYTHONPATH pytest
```

### Подготовка данных

```bash
# Создать тестовый датасет из ~/data/subs
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.data.subs_dataset --limit 1000

# Распаковать демо-данные
unzip ai_data/demo_sample.csv.zip -d ai_data/
```

### Работа с Docker (Triton)

```bash
# Запуск Triton сервера
docker run -d --name triton-server --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/ai_data/triton_models:/models \
  nvcr.io/nvidia/tritonserver:24.09-py3 \
  tritonserver --model-repository=/models

# Проверка статуса
curl http://localhost:8000/v2/health/ready

# Логи
docker logs triton-server

# Остановка
docker stop triton-server && docker rm triton-server
```

### Переменные окружения

```bash
# Экспорт переменных из .env
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)

# Или с активированным venv
source venv/bin/activate
export $(cat .env | grep -v '^#' | grep -v '^$' | xargs)
```

### Make команды

```bash
# Запуск с demo данными
INPUT=ai_data/demo_sample.csv make run

# С ограничением
INPUT=ai_data/demo_sample.csv LIMIT=100 make run

# С кастомными параметрами
INPUT=ai_data/support.csv BATCH=batch-20241121 TEXT_COL=message make run
```

