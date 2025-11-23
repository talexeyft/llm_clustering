# Добавление собственного провайдера LLM

Эта инструкция описывает **упрощенный** процесс добавления нового провайдера LLM в проект `llm_clustering`.

> **Примечание о рефакторинге (ноябрь 2024):** Архитектура была упрощена:
> - Создан `BaseLLMComponent` для устранения дублирования кода
> - Добавлена Pydantic-валидация LLM ответов
> - Settings теперь имеет плоскую структуру (без вложенных config классов)
> - `PipelineRunner` объединен с `ClusteringPipeline`

## 🎯 Быстрый старт: SimpleLLMProvider

**Хорошая новость:** Теперь для добавления своего LLM нужно реализовать **всего 1 метод** - `chat_completion()`!

### Минимальный пример

```python
from llm_clustering import SimpleLLMProvider, ClusteringPipeline
import requests

class MyCustomLLM(SimpleLLMProvider):
    """Ваш кастомный LLM провайдер - всего 1 метод!"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.api_url = "https://api.your-llm.com"
        self.model = "your-model-name"
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        """Единственный метод, который нужно реализовать."""
        response = requests.post(
            f"{self.api_url}/chat/completions",
            json={
                "model": self.model,
                "messages": messages,
                "temperature": temperature or 0.7,
                "max_tokens": max_tokens or 2000,
            },
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()["choices"][0]["message"]["content"]

# Использование
llm = MyCustomLLM(api_key="your-api-key")
pipeline = ClusteringPipeline(llm_provider=llm)
result = pipeline.fit(df, text_column="text")
```

**Вот и всё!** SimpleLLMProvider автоматически предоставит:
- ✅ `describe_cluster()` - через ваш `chat_completion()`
- ✅ `embed()` и `cluster()` - как `NotImplementedError` (они опциональны)

---

## Обзор архитектуры

### Два базовых класса на выбор

#### 1. SimpleLLMProvider (⭐ Рекомендуется)
Для большинства случаев - нужен только `chat_completion()`.

```python
from llm_clustering import SimpleLLMProvider

class MyLLM(SimpleLLMProvider):
    def chat_completion(self, messages, temperature=None, max_tokens=None) -> str:
        # Ваша реализация
        pass
```

**Автоматически получаете:**
- `describe_cluster()` - работает через `chat_completion()`
- `embed()` и `cluster()` - заглушки (опциональные методы)

#### 2. BaseLLMProvider (Продвинутый)
Если нужен полный контроль над всеми методами.

```python
from llm_clustering import BaseLLMProvider

class AdvancedLLM(BaseLLMProvider):
    def chat_completion(self, messages, ...) -> str: pass
    def describe_cluster(self, texts) -> str: pass
    def embed(self, texts) -> list[list[float]]: pass
    def cluster(self, texts, num_clusters) -> list[int]: pass
```

---

## Шаг за шагом: Добавление провайдера

### Вариант A: Простое использование (без регистрации в фабрике)

Если вам нужен провайдер только для вашего проекта - просто используйте его напрямую:

```python
# my_llm.py
from llm_clustering import SimpleLLMProvider

class MyLLM(SimpleLLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        # Ваша реализация
        pass

# Использование
from llm_clustering import ClusteringPipeline
from my_llm import MyLLM

pipeline = ClusteringPipeline(llm_provider=MyLLM(api_key="xxx"))
```

### Вариант B: Полная интеграция (регистрация в фабрике)

Если хотите добавить провайдер в библиотеку:

#### 1. Создайте файл провайдера

`src/llm_clustering/llm/my_provider.py`:

```python
"""My custom LLM provider."""

from llm_clustering.llm.simple_provider import SimpleLLMProvider
from llm_clustering.config import get_settings
from loguru import logger


class MyProvider(SimpleLLMProvider):
    """My custom LLM provider."""
    
    def __init__(self) -> None:
        """Initialize provider from settings."""
        settings = get_settings()
        self.api_key = settings.my_api_key
        self.api_url = settings.my_api_url or "https://api.example.com"
        self.model = settings.my_model or "default-model"
        
        if not self.api_key:
            raise ValueError("MY_API_KEY is required in .env file")
    
    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Chat completion implementation."""
        import requests
        
        try:
            response = requests.post(
                f"{self.api_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature or 0.7,
                    "max_tokens": max_tokens or 2000,
                },
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"API error: {e}")
            raise
```

#### 2. Добавьте настройки

`src/llm_clustering/config/settings.py`:

```python
class Settings(BaseSettings):
    # ... существующие настройки ...
    
    # My Provider
    my_api_key: str = ""
    my_api_url: str = "https://api.example.com"
    my_model: str = "default-model"
```

`env.example`:

```bash
# My Provider
MY_API_KEY=your_api_key_here
MY_API_URL=https://api.example.com
MY_MODEL=default-model
```

#### 3. Зарегистрируйте в фабрике

`src/llm_clustering/llm/factory.py`:

```python
from llm_clustering.llm.my_provider import MyProvider

class LLMFactory:
    _providers: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "my_provider": MyProvider,  # <-- Добавьте здесь
    }
```

#### 4. Экспортируйте (опционально)

`src/llm_clustering/llm/__init__.py`:

```python
from llm_clustering.llm.my_provider import MyProvider

__all__ = [
    # ...
    "MyProvider",
]
```

---

## Примеры реализации

### Пример 1: REST API с requests

```python
from llm_clustering import SimpleLLMProvider
import requests
from loguru import logger


class RESTAPIProvider(SimpleLLMProvider):
    """Provider для стандартного REST API."""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        try:
            response = requests.post(
                f"{self.base_url}/v1/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature or 0.7,
                    "max_tokens": max_tokens or 2000,
                },
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
```

### Пример 2: Локальный Ollama (без SDK)

```python
from llm_clustering import SimpleLLMProvider
import urllib3
import json


class LocalOllamaProvider(SimpleLLMProvider):
    """Provider для локального Ollama."""
    
    def __init__(self, model: str = "llama3", api_url: str = "http://localhost:11434"):
        self.model = model
        self.api_url = api_url
        self.http = urllib3.PoolManager()
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature or 0.7,
                "num_predict": max_tokens or 2000,
            }
        }
        
        response = self.http.request(
            "POST",
            f"{self.api_url}/api/chat",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        
        return json.loads(response.data)["message"]["content"]
```

### Пример 3: OpenAI-совместимый SDK

```python
from llm_clustering import SimpleLLMProvider
from openai import OpenAI  # или другой SDK


class OpenAICompatibleProvider(SimpleLLMProvider):
    """Provider для OpenAI-совместимых API."""
    
    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature or 0.7,
            max_tokens=max_tokens or 2000,
        )
        return response.choices[0].message.content
```

### Пример 4: Кастомизация describe_cluster

Если хотите изменить стандартное поведение `describe_cluster()`:

```python
from llm_clustering import SimpleLLMProvider


class CustomDescriptionProvider(SimpleLLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    def chat_completion(self, messages, temperature=None, max_tokens=None):
        # Ваша реализация
        pass
    
    def describe_cluster(self, texts: list[str]) -> str:
        """Кастомное описание кластера на английском."""
        prompt = f"""Analyze these customer requests and provide a brief summary:

{chr(10).join(f"- {text}" for text in texts[:15])}

Provide a brief 1-sentence summary of the main theme."""
        
        return self.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=100,
        )
```

---

## Использование провайдера

### Способ 1: Напрямую

```python
from my_llm import MyCustomLLM
from llm_clustering import ClusteringPipeline

llm = MyCustomLLM(api_key="xxx")
pipeline = ClusteringPipeline(llm_provider=llm)
result = pipeline.fit(df, text_column="text")
```

### Способ 2: Через фабрику (если зарегистрирован)

```python
from llm_clustering.llm.factory import get_llm_provider

# Использование по имени
llm = get_llm_provider("my_provider")
```

### Способ 3: Через настройки .env

```bash
DEFAULT_LLM_PROVIDER=my_provider
```

```python
# Автоматически использует my_provider из .env
pipeline = ClusteringPipeline()
```

---

## Тестирование

Создайте тестовый файл в `ai_experiments/`:

```python
"""Test custom LLM provider."""

from my_llm import MyCustomLLM
import pandas as pd


def test_provider():
    """Простой тест провайдера."""
    
    # Тест chat_completion
    llm = MyCustomLLM(api_key="xxx")
    response = llm.chat_completion([
        {"role": "user", "content": "Привет! Как дела?"}
    ])
    print(f"Response: {response}")
    
    # Тест describe_cluster (автоматически через chat_completion)
    texts = [
        "Не могу войти в аккаунт",
        "Забыл пароль",
        "Проблема с авторизацией"
    ]
    description = llm.describe_cluster(texts)
    print(f"Cluster description: {description}")
    
    # Тест в pipeline
    from llm_clustering import ClusteringPipeline
    df = pd.DataFrame({"text": texts})
    pipeline = ClusteringPipeline(llm_provider=llm)
    result = pipeline.fit(df, text_column="text")
    print(f"Clusters found: {len(result.clusters)}")


if __name__ == "__main__":
    test_provider()
```

---

## Рекомендации

### ✅ Хорошие практики

1. **Используйте SimpleLLMProvider** для большинства случаев
2. **Обрабатывайте ошибки** и логируйте их с помощью `loguru.logger`
3. **Устанавливайте таймауты** для HTTP-запросов (обычно 60 секунд)
4. **Валидируйте параметры** в `__init__` (API ключи, URL)
5. **Добавляйте docstrings** для методов класса
6. **Используйте типизацию** из `typing`

### ❌ Частые ошибки

1. **Не переопределяйте** `describe_cluster()` без необходимости - стандартная реализация отлично работает
2. **Не реализуйте** `embed()` и `cluster()` если не нужно - они опциональные
3. **Не забывайте** про обработку ошибок API
4. **Не храните** секреты в коде - используйте `.env`

---

## Существующие провайдеры для справки

Посмотрите реализации в `src/llm_clustering/llm/`:

- **`simple_provider.py`** - базовый класс SimpleLLMProvider (⭐ начните с него)
- **`openrouter_provider.py`** - REST API с requests
- **`ollama_provider.py`** - локальный провайдер с urllib3
- **`triton_provider.py`** - Triton Inference Server

---

## FAQ

### Q: Какие методы обязательны?

**A:** Только `chat_completion()` если используете `SimpleLLMProvider`.

### Q: Нужны ли embeddings?

**A:** Нет, метод `embed()` опционален и большинство пайплайнов его не используют.

### Q: Как кастомизировать описание кластеров?

**A:** Переопределите `describe_cluster()` в своём классе (см. Пример 4).

### Q: Что делать если мой LLM API не совместим с OpenAI?

**A:** Просто адаптируйте формат в `chat_completion()` - конвертируйте `messages` в нужный вам формат.

### Q: Можно ли использовать несколько провайдеров одновременно?

**A:** Да, создайте разные инстансы и передавайте их в разные пайплайны.

---

## Заключение

С `SimpleLLMProvider` добавление своего LLM стало **в 4 раза проще**:
- Раньше: 4 абстрактных метода
- Теперь: 1 метод (`chat_completion`)

Удачи! 🚀
