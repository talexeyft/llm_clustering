# LLM Clustering Library Usage Guide

Эта библиотека предоставляет высокоуровневый API для кластеризации текстовых данных с использованием LLM-моделей.

## Установка

```bash
# Установка в editable mode (для разработки)
pip install -e /path/to/llm_clustering

# Или обычная установка
pip install /path/to/llm_clustering

# Проверка установки
python -c "import llm_clustering; print(llm_clustering.__version__)"
```

## Быстрый старт

```python
from llm_clustering import ClusteringPipeline
import pandas as pd

# Создать pipeline
pipeline = ClusteringPipeline()

# Загрузить данные
df = pd.DataFrame({"text": ["Запрос 1", "Запрос 2", ...]})

# Кластеризовать
result = pipeline.fit(df, text_column="text")

# Результаты
print(f"Coverage: {result.coverage:.1f}%")
print(f"Clusters: {len(result.clusters)}")
```

## Основные возможности

### 1. Передача кастомной LLM

```python
from llm_clustering import ClusteringPipeline, BaseLLMProvider

class MyLLM(BaseLLMProvider):
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        # Ваша реализация
        return "JSON response"
    
    # Остальные методы можно не реализовывать
    def embed(self, texts): raise NotImplementedError()
    def cluster(self, texts, num_clusters=None): raise NotImplementedError()
    def describe_cluster(self, texts): raise NotImplementedError()

pipeline = ClusteringPipeline(llm_provider=MyLLM())
result = pipeline.fit(df, text_column="text")
```

### 2. Итеративная обработка

```python
# Обработка большого датасета по частям
for partial in pipeline.fit_partial(df, batch_size=50):
    print(f"Batch {partial.batch_number}: {partial.processed_rows}/{partial.total_rows}")
    print(f"New clusters: {len(partial.new_clusters)}")
    
    # Можно остановиться в любой момент
    if partial.processed_rows >= 200:
        break
```

### 3. Доразметка с существующими кластерами

```python
# Первоначальная разметка
result1 = pipeline.fit(df_part1, text_column="text")

# Доразметка новых данных с учетом найденных кластеров
result2 = pipeline.refit(
    df_part2,
    previous_assignments=result1.assignments,
    text_column="text"
)
```

### 4. Сохранение и загрузка кластеров

```python
from pathlib import Path

# Сохранить
pipeline.save_clusters(Path("my_clusters.json"))

# Загрузить в новый pipeline
new_pipeline = ClusteringPipeline()
new_pipeline.load_clusters(Path("my_clusters.json"))

# Использовать загруженные кластеры
result = new_pipeline.fit(new_df, text_column="text")
```

### 5. Получение найденных кластеров

```python
# Получить все кластеры (отсортированы по частоте)
clusters = pipeline.get_clusters()

for cluster in clusters:
    print(f"{cluster.name}: {cluster.count} requests")
    print(f"  Summary: {cluster.summary}")
    print(f"  Criteria: {cluster.criteria}")
```

### 6. Бизнес-контекст для промптов

```python
business_context = """
Разметка обращений для улучшения бота поддержки.
Разделяй проблемы по сложности:
- Простые: FAQ (автоматизируемые)
- Средние: требуют проверки данных
- Сложные: нестандартные ситуации
"""

pipeline = ClusteringPipeline(business_context=business_context)
result = pipeline.fit(df, text_column="text")
```

### 7. Кастомные настройки

```python
from llm_clustering import Settings

settings = Settings(
    clustering_batch_size=30,
    max_clusters_per_batch=5,
    default_temperature=0.1,
    default_llm_provider="ollama",
    ollama_model="qwen3:30b",
)

pipeline = ClusteringPipeline(settings=settings)
```

## Структура результатов

### ClusteringResult

```python
result = pipeline.fit(df)

# Атрибуты
result.batch_id              # ID батча
result.assignments           # DataFrame с назначениями
result.clusters              # Список найденных кластеров
result.coverage              # Процент покрытия (0-100)
result.metrics               # Словарь с метриками
result.total_requests        # Всего запросов
result.assigned_requests     # Назначено в кластеры
```

### PartialResult

```python
for partial in pipeline.fit_partial(df, batch_size=50):
    partial.batch_number        # Номер батча
    partial.batch_id            # ID батча
    partial.assignments         # DataFrame с назначениями
    partial.new_clusters        # Новые кластеры в этом батче
    partial.processed_rows      # Обработано строк
    partial.total_rows          # Всего строк
```

### ClusterRecord

```python
clusters = pipeline.get_clusters()

for cluster in clusters:
    cluster.cluster_id          # Уникальный ID
    cluster.name                # Название
    cluster.summary             # Описание
    cluster.criteria            # Критерии отнесения
    cluster.sample_requests     # Примеры запросов (ID)
    cluster.count               # Количество запросов
    cluster.status              # Статус (active/tentative)
    cluster.created_at          # Время создания
    cluster.updated_at          # Время обновления
```

## Требования к данным

Входной DataFrame должен содержать:

- **Обязательный столбец**: текст для кластеризации (по умолчанию `text`)
- **Опциональные столбцы**: любые дополнительные данные для контекста

Библиотека автоматически добавит `request_id` если его нет.

## Примеры

Полные примеры использования смотрите в:
- `examples.py` - демонстрация всех возможностей
- `ai_doc/quickstart.md` - детальное руководство
- `ai_experiments/test_library_api.py` - тесты API

## Настройка LLM провайдера

### Встроенные провайдеры

Библиотека поддерживает:
- `ollama` - локальная Ollama (по умолчанию)
- `openrouter` - OpenRouter API
- `openai` - OpenAI API
- `anthropic` - Anthropic API
- `triton` - Triton Inference Server

### Конфигурация через Settings

```python
settings = Settings(
    default_llm_provider="ollama",
    ollama_model="qwen3:30b",
    ollama_api_url="http://localhost:11434/api",
    default_temperature=0.0,
    default_max_tokens=4096,
)

pipeline = ClusteringPipeline(settings=settings)
```

### Конфигурация через .env

```bash
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:30b
OLLAMA_API_URL=http://localhost:11434/api
DEFAULT_TEMPERATURE=0.0
DEFAULT_MAX_TOKENS=4096
```

## Продвинутое использование

### Параллельная обработка

Библиотека автоматически использует параллельную обработку для ускорения inference:

```python
settings = Settings(
    parallel_inference_batch_size=10,  # 10 параллельных запросов
)
pipeline = ClusteringPipeline(settings=settings)
```

### Кастомный путь к реестру кластеров

```python
from pathlib import Path

pipeline = ClusteringPipeline(
    registry_path=Path("my_project/clusters.json")
)
```

### Ограничение количества данных

```python
# Обработать только первые 100 строк
result = pipeline.fit(df, text_column="text", limit=100)
```

### Возобновление обработки с определенной позиции

```python
# Начать с 100-й строки
for partial in pipeline.fit_partial(df, batch_size=50, start_from=100):
    # Обработка
    pass
```

## Интеграция в Jupyter Notebook

```python
# Установите в kernel
!pip install -e /path/to/llm_clustering

# Используйте как обычно
from llm_clustering import ClusteringPipeline
import pandas as pd

pipeline = ClusteringPipeline()
result = pipeline.fit(df, text_column="text")

# Визуализация
import matplotlib.pyplot as plt

cluster_counts = [c.count for c in result.clusters]
cluster_names = [c.name[:30] for c in result.clusters]

plt.barh(cluster_names, cluster_counts)
plt.xlabel("Number of requests")
plt.title("Cluster Distribution")
plt.tight_layout()
plt.show()
```

## Troubleshooting

### ModuleNotFoundError

Если получаете ошибку импорта:
```bash
pip install -e /path/to/llm_clustering
```

### LLM не отвечает

Проверьте:
1. Запущен ли LLM сервер (для Ollama: `ollama serve`)
2. Правильно ли указаны URL и модель в настройках
3. Доступна ли модель (`ollama list`)

### Ошибки парсинга JSON

Если LLM возвращает невалидный JSON:
- Попробуйте другую модель
- Установите `temperature=0.0` для более детерминированных ответов
- Проверьте промпты в `src/llm_clustering/llm/prompts/`

## Дополнительная информация

- **Репозиторий**: `/home/alex/tradeML/llm_clustering`
- **Документация**: `ai_doc/`
- **Примеры**: `examples.py`
- **Тесты**: `ai_experiments/test_library_api.py`

