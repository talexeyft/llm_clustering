# LLM Clustering

Библиотека для кластеризации обращений клиентов в контакт-центр с использованием LLM-моделей.

## 🎯 Основные возможности

- 🚀 **Высокоуровневый API** для быстрого старта
- 🔌 **Поддержка кастомных LLM-моделей** через простой интерфейс
- 📊 **Итеративная обработка** больших датасетов
- 🔄 **Доразметка** с учетом существующих кластеров
- 💾 **Сохранение и загрузка** кластеров
- 🎯 **Бизнес-контекст** для специфичной кластеризации
- 📦 **Легкая установка** в любое окружение
- ⚡ **Параллельная обработка** для ускорения inference
- 🏭 **Поддержка различных LLM-провайдеров**: Ollama, OpenRouter, OpenAI, Anthropic, Triton

## 📦 Установка

### Как библиотека (рекомендуется)

```bash
# Установка в editable mode (для разработки)
pip install -e /path/to/llm_clustering

# Или обычная установка
pip install /path/to/llm_clustering

# Проверка установки
python -c "import llm_clustering; print(llm_clustering.__version__)"
llm-clustering --help
```

### Для разработки

```bash
# Создать виртуальное окружение
python3 -m venv venv
source venv/bin/activate

# Установить зависимости
pip install -r requirements.txt
# или с dev-зависимостями
pip install -e ".[dev]"
```

## ⚙️ Конфигурация

Скопируйте `env.example` в `.env` и заполните необходимые параметры:

```bash
cp env.example .env
```

**По умолчанию**: Проект настроен на использование **Ollama** с моделью **`qwen3:30b`**.

### Настройка Ollama (по умолчанию)

```bash
# Запустить Ollama
ollama serve

# Скачать модель
ollama pull qwen3:30b

# В .env убедитесь что:
DEFAULT_LLM_PROVIDER=ollama
OLLAMA_MODEL=qwen3:30b
OLLAMA_API_URL=http://localhost:11434/api
```

### Другие провайдеры

Библиотека поддерживает:
- **ollama** - локальная Ollama (по умолчанию)
- **openrouter** - OpenRouter API
- **openai** - OpenAI API
- **anthropic** - Anthropic API
- **triton** - Triton Inference Server

Подробнее см. в разделе [Кастомные LLM-провайдеры](#кастомные-llm-провайдеры).

## 🚀 Быстрый старт

### Базовое использование

```python
from llm_clustering import ClusteringPipeline
import pandas as pd

# Создать pipeline
pipeline = ClusteringPipeline()

# Загрузить данные
df = pd.DataFrame({
    "text": ["Не могу войти", "Забыл пароль", "Товар не пришел"]
})

# Кластеризовать
result = pipeline.fit(df, text_column="text")

# Результаты
print(f"Coverage: {result.coverage:.1f}%")
print(f"Clusters: {len(result.clusters)}")

for cluster in result.clusters:
    print(f"  - {cluster.name}: {cluster.count} requests")
```

### С кастомной LLM

```python
from llm_clustering import ClusteringPipeline, BaseLLMProvider

class MyLLM(BaseLLMProvider):
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        # Ваша реализация
        return json_response
    
    # Остальные методы можно не реализовывать
    def embed(self, texts): raise NotImplementedError()
    def cluster(self, texts, num_clusters=None): raise NotImplementedError()
    def describe_cluster(self, texts): raise NotImplementedError()

pipeline = ClusteringPipeline(llm_provider=MyLLM())
result = pipeline.fit(df, text_column="text")
```

### Использование CLI

```bash
# Кластеризация файла
llm-clustering --input data.csv --text-column text --limit 100

# Справка
llm-clustering --help
```

### Демо-данные

В репозитории включен файл `ai_data/demo_sample.csv.zip` с 1000 образцов для демонстрации:

```bash
# Распаковать
unzip ai_data/demo_sample.csv.zip -d ai_data/

# Запустить на 20 сэмплах
llm-clustering --input ai_data/demo_sample.csv --limit 20
```

## 🔌 Кастомные LLM-провайдеры

### Использование встроенных провайдеров

```python
from llm_clustering import ClusteringPipeline, Settings

# Через Settings
settings = Settings(
    default_llm_provider="ollama",
    ollama_model="qwen3:30b",
    default_temperature=0.0,
)
pipeline = ClusteringPipeline(settings=settings)

# Или через .env
# DEFAULT_LLM_PROVIDER=ollama
# OLLAMA_MODEL=qwen3:30b
```

### Создание собственного провайдера

Минимальная реализация требует только метод `chat_completion`:

```python
from llm_clustering import BaseLLMProvider
import requests

class MyCustomProvider(BaseLLMProvider):
    def __init__(self):
        self.api_url = "https://my-llm-api.com/v1/chat"
        self.api_key = "your-api-key"
        self.temperature = 0.0
        self.max_tokens = 4096
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        """
        Главный метод для генерации ответов.
        
        Args:
            messages: список словарей [{"role": "user", "content": "text"}]
            temperature: температура генерации (опционально)
            max_tokens: макс. кол-во токенов (опционально)
            
        Returns:
            str: JSON-строка с ответом модели
        """
        response = requests.post(
            self.api_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "messages": messages,
                "temperature": temperature or self.temperature,
                "max_tokens": max_tokens or self.max_tokens,
            }
        )
        return response.json()["choices"][0]["message"]["content"]
    
    # Эти методы можно не реализовывать
    def embed(self, texts): raise NotImplementedError()
    def cluster(self, texts, num_clusters=None): raise NotImplementedError()
    def describe_cluster(self, texts): raise NotImplementedError()

# Использование
pipeline = ClusteringPipeline(llm_provider=MyCustomProvider())
result = pipeline.fit(df, text_column="text")
```

### Регистрация в фабрике (опционально)

Для удобного использования через `get_llm_provider()`:

1. Создайте файл `src/llm_clustering/llm/my_provider.py`
2. Добавьте в `src/llm_clustering/llm/factory.py`:

```python
from llm_clustering.llm.my_provider import MyProvider

class LLMFactory:
    _providers = {
        # ... существующие ...
        "myprovider": MyProvider,
    }
```

3. Используйте через фабрику:

```python
from llm_clustering.llm.factory import get_llm_provider

provider = get_llm_provider("myprovider")
pipeline = ClusteringPipeline(llm_provider=provider)
```

Подробное руководство: [doc/adding_custom_provider.md](doc/adding_custom_provider.md)

## 🎯 Кастомизация промптов и бизнес-контекст

### Использование бизнес-контекста

Бизнес-контекст добавляется в промпты для специфичной кластеризации:

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

### Кастомизация промптов

Промпты находятся в `src/llm_clustering/clustering/`:
- `judge.py` - промпт для назначения запросов в кластеры
- `proposer.py` - промпт для создания новых кластеров

Вы можете:
1. Модифицировать промпты напрямую в коде
2. Передать `business_context` при создании pipeline
3. Создать собственные классы `Judge` и `Proposer` с кастомными промптами

```python
from llm_clustering.clustering import Judge, Proposer

class MyJudge(Judge):
    def build_prompt(self, ...):
        # Ваш кастомный промпт
        return custom_prompt

pipeline = ClusteringPipeline()
# Используйте кастомные компоненты через низкоуровневый API
```

## 📚 API и возможности

### 1. Итеративная обработка

Обработка больших датасетов по частям:

```python
for partial in pipeline.fit_partial(df, batch_size=50):
    print(f"Batch {partial.batch_number}: {partial.processed_rows}/{partial.total_rows}")
    print(f"New clusters: {len(partial.new_clusters)}")
    
    # Можно остановиться в любой момент
    if partial.processed_rows >= 200:
        break
```

### 2. Доразметка данных

Доразметка новых данных с учетом найденных кластеров:

```python
# Первоначальная разметка
result1 = pipeline.fit(df_part1, text_column="text")

# Доразметка с учетом существующих кластеров
result2 = pipeline.refit(
    df_part2,
    previous_assignments=result1.assignments,
    text_column="text"
)
```

### 3. Сохранение и загрузка кластеров

```python
from pathlib import Path

# Сохранить
pipeline.save_clusters(Path("clusters.json"))

# Загрузить в новый pipeline
new_pipeline = ClusteringPipeline()
new_pipeline.load_clusters(Path("clusters.json"))

# Использовать загруженные кластеры
result = new_pipeline.fit(new_df, text_column="text")
```

### 4. Получение кластеров

```python
# Получить все кластеры (отсортированы по частоте)
clusters = pipeline.get_clusters()

for cluster in clusters:
    print(f"{cluster.name}: {cluster.count} requests")
    print(f"  Summary: {cluster.summary}")
    print(f"  Criteria: {cluster.criteria}")
```

### 5. Ограничение данных

```python
# Обработать только первые 100 строк
result = pipeline.fit(df, text_column="text", limit=100)

# Начать с 100-й строки
for partial in pipeline.fit_partial(df, batch_size=50, start_from=100):
    pass
```

### 6. Параллельная обработка

```python
settings = Settings(
    parallel_inference_batch_size=10,  # 10 параллельных запросов
)
pipeline = ClusteringPipeline(settings=settings)
```

## 📊 Структура результатов

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

## 📁 Структура проекта

```
llm_clustering/
├── src/
│   └── llm_clustering/
│       ├── __init__.py         # Публичный API
│       ├── api.py              # ClusteringPipeline
│       ├── config/             # Конфигурация и настройки
│       ├── llm/                # LLM провайдеры
│       │   ├── base.py         # BaseLLMProvider
│       │   ├── factory.py      # LLMFactory
│       │   ├── ollama_provider.py
│       │   ├── openrouter_provider.py
│       │   └── ...
│       ├── clustering/         # Логика кластеризации
│       │   ├── clusterer.py    # Основной движок
│       │   ├── judge.py        # Назначение в кластеры
│       │   ├── proposer.py     # Создание новых кластеров
│       │   ├── registry.py     # Управление кластерами
│       │   └── utils.py
│       ├── pipeline/           # Pipeline runner
│       ├── data/               # Работа с данными
│       └── utils/              # Утилиты
├── tests/                      # Юнит-тесты
├── ai_experiments/             # Эксперименты и тесты
├── ai_doc/                     # Документация
├── ai_data/                    # Данные, логи, результаты
│   ├── batches/                # Батчи обработки
│   ├── results/                # Результаты назначений
│   ├── prompts/                # Логи промптов
│   └── reports/                # Отчеты QA
├── doc/                        # Дополнительная документация
├── examples.py                 # Примеры использования
└── README.md                   # Этот файл
```

## 📖 Требования к данным

Входной DataFrame должен содержать:

- **Обязательный столбец**: текст для кластеризации (по умолчанию `text`)
- **Опциональные столбцы**: любые дополнительные данные для контекста

**Автоматическая обработка:**
- Если `request_id` отсутствует, библиотека автоматически сгенерирует уникальные идентификаторы
- Формат автогенерации: `req-{batch_id}-{index}`
- Вы можете предоставить свои `request_id` для отслеживания через вашу систему

## 💡 Примеры использования

### Интеграция в Jupyter Notebook

```python
# Установите в kernel
!pip install -e /path/to/llm_clustering

from llm_clustering import ClusteringPipeline
import pandas as pd
import matplotlib.pyplot as plt

pipeline = ClusteringPipeline()
result = pipeline.fit(df, text_column="text")

# Визуализация
cluster_counts = [c.count for c in result.clusters]
cluster_names = [c.name[:30] for c in result.clusters]

plt.barh(cluster_names, cluster_counts)
plt.xlabel("Number of requests")
plt.title("Cluster Distribution")
plt.tight_layout()
plt.show()
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

# Инициализация
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

# Сохранение
result.assignments.to_csv("results.csv", index=False)
pipeline.save_clusters(Path("final_clusters.json"))
```

Больше примеров в [examples.py](examples.py).

## 🔧 Troubleshooting

### ModuleNotFoundError

```bash
pip install -e /path/to/llm_clustering
```

### LLM не отвечает

1. Проверьте, что LLM сервер запущен (для Ollama: `ollama serve`)
2. Проверьте URL и модель в настройках
3. Проверьте доступность модели (`ollama list`)

### Ошибки парсинга JSON

- Попробуйте другую модель
- Установите `temperature=0.0`
- Проверьте промпты в `src/llm_clustering/clustering/`

## 📚 Документация

- **[Руководство по API](ai_doc/library_usage.md)** - полное описание всех возможностей
- **[Быстрый старт](ai_doc/quickstart.md)** - установка и первые шаги
- **[Добавление кастомного провайдера](doc/adding_custom_provider.md)** - создание своих провайдеров
- **[Примеры](examples.py)** - демонстрация всех возможностей
- **[Демо-данные](ai_doc/demo_data.md)** - описание тестовых данных

## 🧪 Тестирование

```bash
# Запуск unit-тестов
PYTHONPATH=src:$PYTHONPATH pytest

# Тест библиотечного API
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_library_api.py

# Тест Ollama
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_ollama_success.py

# Тест OpenRouter
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_openrouter_simple.py
```

## 🚀 Запуск через CLI

```bash
# MVP пайплайн
INPUT=ai_data/demo_sample.csv make run

# С ограничением
llm-clustering --input ai_data/demo_sample.csv --limit 100

# С кастомными параметрами
llm-clustering \
  --input data.csv \
  --text-column message \
  --batch-id batch-001 \
  --limit 200
```

После выполнения:
- Батчи в `ai_data/batches/`
- Результаты в `ai_data/results/`
- Логи промптов в `ai_data/prompts/`
- Отчеты QA в `ai_data/reports/`

## 📄 Лицензия

MIT

## 👥 Авторы

Alex - alex@example.com

## 🤝 Вклад

Вклад приветствуется! Создавайте issue или pull request.

