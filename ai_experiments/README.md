# AI Experiments

Эта папка содержит различные тесты и эксперименты для библиотеки llm_clustering.

## Standalone пример (перенесен в examples/)

### 🌟 См. examples/standalone_ollama_example.py
**Полноценный автономный пример кластеризации с Ollama**

**Запуск:**
```bash
python examples/standalone_ollama_example.py
```

📖 Подробная документация: [ai_doc/standalone_example.md](../ai_doc/standalone_example.md)

---

## Тесты провайдеров

### test_ollama_success.py
Проверка работы Ollama провайдера

**Запуск:**
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_ollama_success.py
```

### test_openrouter_simple.py
Проверка работы OpenRouter провайдера

**Запуск:**
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_openrouter_simple.py
```

### test_triton.py
Проверка работы Triton Inference Server провайдера

**Запуск:**
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_triton.py
```

---

## Тесты библиотечного API

### test_library_api.py
Полный тест публичного API библиотеки

**Запуск:**
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_library_api.py
```

### test_simple_provider_*.py
Тесты SimpleLLMProvider (упрощенный базовый класс)

---

## Тесты производительности

### test_parallel_inference.py
Тест параллельной обработки батчей

**Запуск:**
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_parallel_inference.py
```

### benchmark_parallel.py
Бенчмарк производительности параллельной обработки

**Запуск:**
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/benchmark_parallel.py
```

### visualize_benchmark.py
Визуализация результатов бенчмарков

---

## Тесты функциональности

### test_batch_mvp.py
Тест MVP батчевой обработки

### test_limit_*.py
Тесты параметра limit в pipeline

### test_examples_live.py
Живое тестирование примеров из examples.py

---

## Структура экспериментов

```
ai_experiments/
├── README.md                           # Этот файл
├── standalone_ollama_example.py        # 🌟 Полный автономный пример
│
├── test_ollama_*.py                    # Тесты Ollama
├── test_openrouter_*.py                # Тесты OpenRouter
├── test_triton.py                      # Тесты Triton
│
├── test_library_api.py                 # Тесты публичного API
├── test_simple_provider*.py            # Тесты SimpleLLMProvider
│
├── test_parallel_inference.py          # Параллельная обработка
├── benchmark_parallel.py               # Бенчмарки
├── visualize_benchmark.py              # Визуализация
│
└── test_*.py                           # Прочие тесты
```

---

## Быстрый старт для новичков

1. **Установите зависимости:**
   ```bash
   pip install -e .
   ```

2. **Настройте Ollama:**
   ```bash
   ollama serve                # В одном терминале
   ollama pull qwen3:30b      # В другом терминале
   ```

3. **Запустите standalone пример:**
   ```bash
   python ai_experiments/standalone_ollama_example.py
   ```

4. **Изучите результаты:**
   ```bash
   ls -lh ai_data/standalone_example_results/
   ```

---

## Полезные ссылки

- 📚 [Основная документация](../README.md)
- 🚀 [Быстрый старт](../ai_doc/quickstart.md)
- 📖 [Использование библиотеки](../ai_doc/library_usage.md)
- 💡 [Все примеры API](../examples.py)
- 🌟 [Standalone пример (подробно)](../ai_doc/standalone_example.md)

