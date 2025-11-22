# Добавление собственного провайдера LLM

Эта инструкция описывает процесс добавления нового провайдера LLM в проект `llm_clustering`.

## Обзор архитектуры

Все провайдеры LLM наследуются от базового класса `BaseLLMProvider` и должны реализовать три обязательных метода:

- `embed()` - генерация эмбеддингов для списка текстов
- `cluster()` - кластеризация текстов с использованием LLM
- `describe_cluster()` - генерация описания кластера текстов

Провайдеры регистрируются в `LLMFactory` и могут быть использованы через фабрику или напрямую.

## Шаги добавления провайдера

### 1. Создание класса провайдера

Создайте новый файл в директории `src/llm_clustering/llm/` с именем `{provider_name}_provider.py`.

Пример структуры:

```python
"""Описание провайдера LLM."""

from typing import Any

from loguru import logger

from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider


class CustomProvider(BaseLLMProvider):
    """Описание провайдера для эмбеддингов и кластеризации."""

    def __init__(self) -> None:
        """Инициализация провайдера."""
        settings = get_settings()
        # Загрузите необходимые настройки из конфигурации
        self.api_key = settings.custom_api_key
        self.api_url = settings.custom_api_url or "https://api.example.com"
        self.model = settings.custom_model or "default-model"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

        # Валидация обязательных параметров
        if not self.api_key:
            raise ValueError(
                "Custom API key is required. Set CUSTOM_API_KEY in .env file"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Генерация эмбеддингов для списка текстов."""
        # Реализуйте логику генерации эмбеддингов
        embeddings = []
        for text in texts:
            # Ваша логика здесь
            embedding = self._get_embedding(text)
            embeddings.append(embedding)
        return embeddings

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Кластеризация текстов с использованием LLM."""
        # Реализуйте логику кластеризации
        # Возвращает список индексов кластеров для каждого текста
        raise NotImplementedError("Clustering logic to be implemented")

    def describe_cluster(self, texts: list[str]) -> str:
        """Генерация описания кластера текстов."""
        # Создайте промпт для описания кластера
        prompt = f"""Проанализируй следующие обращения клиентов в контакт-центр и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in texts[:10])}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""

        # Выполните запрос к API
        response = self._make_request(prompt)
        
        # Извлеките и верните описание
        description = self._extract_description(response)
        return description.strip()

    def _make_request(self, prompt: str) -> dict[str, Any]:
        """Вспомогательный метод для выполнения запросов к API."""
        # Реализуйте логику HTTP-запросов
        # Используйте requests, httpx, urllib3 или клиентскую библиотеку провайдера
        pass

    def _get_embedding(self, text: str) -> list[float]:
        """Вспомогательный метод для получения эмбеддинга одного текста."""
        # Реализуйте логику получения эмбеддинга
        pass

    def _extract_description(self, response: dict[str, Any]) -> str:
        """Вспомогательный метод для извлечения описания из ответа API."""
        # Реализуйте логику парсинга ответа
        pass
```

### 2. Добавление настроек в конфигурацию

Добавьте необходимые настройки в `src/llm_clustering/config/settings.py`:

```python
class Settings(BaseSettings):
    # ... существующие настройки ...
    
    # Custom Provider specific
    custom_api_key: str = ""
    custom_api_url: str = "https://api.example.com"
    custom_model: str = "default-model"
```

Также добавьте соответствующие переменные окружения в `env.example`:

```bash
# Custom Provider
CUSTOM_API_KEY=your_api_key_here
CUSTOM_API_URL=https://api.example.com
CUSTOM_MODEL=default-model
```

### 3. Регистрация провайдера в фабрике

Добавьте импорт и регистрацию провайдера в `src/llm_clustering/llm/factory.py`:

```python
from llm_clustering.llm.custom_provider import CustomProvider

class LLMFactory:
    """Factory for creating LLM providers."""

    _providers: dict[str, type[BaseLLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "openrouter": OpenRouterProvider,
        "ollama": OllamaProvider,
        "custom": CustomProvider,  # Добавьте ваш провайдер
    }
```

### 4. Экспорт провайдера (опционально)

Если нужно, добавьте экспорт в `src/llm_clustering/llm/__init__.py`:

```python
from llm_clustering.llm.custom_provider import CustomProvider

__all__ = [
    # ... существующие экспорты ...
    "CustomProvider",
]
```

## Примеры реализации

### Пример 1: REST API провайдер (как OpenRouter)

```python
"""Пример провайдера с REST API."""

import requests
from loguru import logger
from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider


class RESTProvider(BaseLLMProvider):
    """Провайдер с REST API."""

    API_URL = "https://api.example.com/v1"

    def __init__(self) -> None:
        """Инициализация провайдера."""
        settings = get_settings()
        self.api_key = settings.rest_api_key
        self.model = settings.rest_model or "default-model"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

        if not self.api_key:
            raise ValueError("REST API key is required")

    def _make_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Выполнение запроса к REST API."""
        url = f"{self.API_URL}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"REST API error: {e}")
            if hasattr(e, "response") and e.response is not None:
                logger.error(f"Response: {e.response.text}")
            raise

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Генерация эмбеддингов."""
        embeddings = []
        for text in texts:
            payload = {"model": self.model, "input": text}
            response = self._make_request("embeddings", payload)
            embeddings.append(response.get("data", [{}])[0].get("embedding", []))
        return embeddings

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Кластеризация текстов."""
        # Реализуйте логику кластеризации
        raise NotImplementedError

    def describe_cluster(self, texts: list[str]) -> str:
        """Генерация описания кластера."""
        prompt = f"""Проанализируй следующие обращения клиентов в контакт-центр и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in texts[:10])}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""

        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        response = self._make_request("chat/completions", payload)
        return response["choices"][0]["message"]["content"].strip()
```

### Пример 2: Локальный провайдер (как Ollama)

```python
"""Пример локального провайдера."""

import json
import urllib3
from loguru import logger
from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider

urllib3.disable_warnings()


class LocalProvider(BaseLLMProvider):
    """Локальный провайдер для моделей, запущенных локально."""

    def __init__(self) -> None:
        """Инициализация провайдера."""
        settings = get_settings()
        self.api_url = settings.local_api_url
        self.model = settings.local_model or "local-model"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

    def _make_request(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Выполнение запроса к локальному API."""
        url = f"{self.api_url}/{endpoint}"
        # Реализуйте логику запроса (аналогично OllamaProvider)
        # ...
        pass

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Генерация эмбеддингов."""
        # Реализуйте логику
        pass

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Кластеризация текстов."""
        raise NotImplementedError

    def describe_cluster(self, texts: list[str]) -> str:
        """Генерация описания кластера."""
        # Реализуйте логику
        pass
```

### Пример 3: Провайдер с использованием клиентской библиотеки

```python
"""Пример провайдера с использованием клиентской библиотеки."""

from vendor_sdk import Client  # Пример клиентской библиотеки
from loguru import logger
from llm_clustering.config import get_settings
from llm_clustering.llm.base import BaseLLMProvider


class SDKProvider(BaseLLMProvider):
    """Провайдер с использованием SDK."""

    def __init__(self) -> None:
        """Инициализация провайдера."""
        settings = get_settings()
        self.api_key = settings.sdk_api_key
        self.model = settings.sdk_model or "default-model"
        self.temperature = settings.default_temperature
        self.max_tokens = settings.default_max_tokens

        if not self.api_key:
            raise ValueError("SDK API key is required")

        # Инициализация клиента
        self.client = Client(api_key=self.api_key)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Генерация эмбеддингов."""
        embeddings = []
        for text in texts:
            embedding = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embeddings.append(embedding.data[0].embedding)
        return embeddings

    def cluster(self, texts: list[str], num_clusters: int | None = None) -> list[int]:
        """Кластеризация текстов."""
        raise NotImplementedError

    def describe_cluster(self, texts: list[str]) -> str:
        """Генерация описания кластера."""
        prompt = f"""Проанализируй следующие обращения клиентов в контакт-центр и создай краткое описание общей темы или проблемы:

{chr(10).join(f"- {text}" for text in texts[:10])}

Создай краткое описание (1-2 предложения) общей темы этих обращений на русском языке."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content.strip()
```

## Использование нового провайдера

После регистрации провайдера его можно использовать несколькими способами:

### Способ 1: Через фабрику

```python
from llm_clustering.llm.factory import get_llm_provider

# Использование провайдера по умолчанию
provider = get_llm_provider()

# Использование конкретного провайдера
provider = get_llm_provider("custom")
```

### Способ 2: Напрямую

```python
from llm_clustering.llm.custom_provider import CustomProvider

provider = CustomProvider()
embeddings = provider.embed(["текст 1", "текст 2"])
```

### Способ 3: Через настройки

Установите провайдера по умолчанию в `.env`:

```bash
DEFAULT_LLM_PROVIDER=custom
```

## Тестирование

Создайте тестовый файл в `ai_experiments/` для проверки функциональности:

```python
"""Тест нового провайдера."""

from llm_clustering.llm.factory import get_llm_provider

def test_custom_provider():
    """Тест провайдера."""
    provider = get_llm_provider("custom")
    
    # Тест эмбеддингов
    texts = ["Тестовый текст 1", "Тестовый текст 2"]
    embeddings = provider.embed(texts)
    print(f"Embeddings shape: {[len(e) for e in embeddings]}")
    
    # Тест описания кластера
    description = provider.describe_cluster(texts)
    print(f"Cluster description: {description}")

if __name__ == "__main__":
    test_custom_provider()
```

## Рекомендации

1. **Обработка ошибок**: Всегда обрабатывайте ошибки API и логируйте их с помощью `logger` из `loguru`.

2. **Таймауты**: Устанавливайте разумные таймауты для HTTP-запросов (обычно 60 секунд).

3. **Валидация**: Проверяйте наличие обязательных параметров (API ключи, URL) в `__init__`.

4. **Типизация**: Используйте типы из `typing` для лучшей читаемости и поддержки IDE.

5. **Документация**: Добавляйте docstrings для всех методов класса.

6. **Конфигурация**: Выносите все настраиваемые параметры в `Settings` и `.env`.

## Существующие провайдеры для справки

- `OpenRouterProvider` (`openrouter_provider.py`) - пример REST API провайдера
- `OllamaProvider` (`ollama_provider.py`) - пример локального провайдера с urllib3
- `OpenAIProvider` (`openai_provider.py`) - базовая структура (не реализован)
- `AnthropicProvider` (`anthropic_provider.py`) - базовая структура (не реализован)

Изучите эти файлы для понимания различных подходов к реализации.

