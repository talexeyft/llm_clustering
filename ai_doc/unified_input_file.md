# Единый входной файл с параметром --limit

## Проблема

Ранее для разных размеров выборки создавались отдельные файлы:
- `subs_sample_20.parquet` - для быстрых тестов
- `subs_sample_100.parquet` - для средних тестов
- `subs_sample_1000.parquet` - для полных тестов

Это приводило к:
- Дублированию данных
- Необходимости пересоздавать файлы для каждого размера
- Захламлению папки `ai_data/`

## Решение

Теперь используется единый файл `subs_sample.parquet` с максимальным количеством сообщений (по умолчанию 1000), а ограничение размера выборки задается через параметр `--limit` при запуске pipeline.

## Изменения в коде

### 1. `src/llm_clustering/data/subs_dataset.py`

Функция `materialize_subs_sample` теперь создает файл без суффикса размера:
```python
# Было:
parquet_path = target_dir / f"{file_stem}_{sample_size}.parquet"

# Стало:
parquet_path = target_dir / f"{file_stem}.parquet"
```

### 2. `src/llm_clustering/main.py`

Добавлен новый параметр `--limit`:
```python
parser.add_argument(
    "--limit",
    type=int,
    default=None,
    help="Limit the number of input messages to process."
)
```

В функции `main()` добавлена логика ограничения:
```python
if args.limit is not None and args.limit > 0:
    original_count = len(dataframe)
    dataframe = dataframe.head(args.limit)
    print(f"Limited input from {original_count} to {len(dataframe)} messages")
```

### 3. `ai_doc/quickstart.md`

Обновлены все примеры команд для использования единого файла с параметром `--limit`.

## Использование

### Создание единого файла

```bash
# Создать единый файл с 1000 сообщениями
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.data.subs_dataset --limit 1000
```

Результат:
- `ai_data/subs_sample.parquet`
- `ai_data/subs_sample.csv`

### Запуск с ограничением

```bash
# Быстрый тест на 20 сообщениях
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/subs_sample.parquet \
  --limit 20 \
  --batch-id test_quick

# Средний тест на 100 сообщениях
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/subs_sample.parquet \
  --limit 100 \
  --batch-id test_100

# Полный тест на всех сообщениях файла
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.main \
  --input ai_data/subs_sample.parquet \
  --batch-id test_full
```

## Миграция

Если у вас есть старые файлы с размерами в имени, их можно удалить:

```bash
rm -f ai_data/subs_sample_*.parquet ai_data/subs_sample_*.csv
```

Новый единый файл создается командой:
```bash
PYTHONPATH=src:$PYTHONPATH python -m llm_clustering.data.subs_dataset --limit 1000
```

## Преимущества

✅ **Меньше дублирования данных** - один файл вместо нескольких  
✅ **Гибкость** - можно задать любой лимит через `--limit`  
✅ **Меньше мусора** - не нужно хранить множество файлов разных размеров  
✅ **Проще поддержка** - все примеры используют один файл  

## Тестирование

Запустите тест для проверки функциональности:
```bash
PYTHONPATH=src:$PYTHONPATH python ai_experiments/test_limit_parameter.py
```

Ожидаемый результат:
```
✅ Full file loaded: 1000 messages
✅ Limited to 20: 20 messages
✅ Limited to 100: 100 messages
✅ Limited to 500: 500 messages
✅ All tests passed!
```

